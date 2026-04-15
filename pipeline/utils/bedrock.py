import json
import logging
import re
import sys
import threading
import time

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, ReadTimeoutError

from pipeline.utils.config import get_config

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimate at ~4 characters per token."""
    return len(text) // 4


class _Spinner:
    """Inline spinner that shows elapsed time without adding log lines.

    Automatically suppresses output when tqdm progress bars are active to
    avoid corrupting their display.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None

    def _tqdm_active(self) -> bool:
        """Check if any tqdm progress bars are currently being displayed."""
        try:
            from tqdm import tqdm
            return len(getattr(tqdm, "_instances", set())) > 0
        except ImportError:
            return False

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        if not self._tqdm_active():
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()

    def _spin(self):
        start = time.time()
        i = 0
        while not self._stop.is_set():
            if not self._tqdm_active():
                elapsed = int(time.time() - start)
                frame = self.FRAMES[i % len(self.FRAMES)]
                sys.stderr.write(f"\r  {frame} Waiting for Bedrock... {elapsed}s")
                sys.stderr.flush()
            i += 1
            self._stop.wait(0.1)


class BedrockClient:
    """Wrapper around AWS Bedrock invoke_model with retry and JSON parsing."""

    def __init__(self, model_id: str = None, region: str = None, max_tokens: int = None):
        config = get_config()
        self.model_id = model_id or config["BEDROCK_MODEL_ID"]
        self.max_tokens = max_tokens or config["MAX_TOKENS"]
        self._last_stop_reason = None
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region or config["AWS_REGION"],
            config=BotoConfig(read_timeout=300),
        )

    def invoke(
        self,
        messages: list[dict],
        system: str = None,
        max_tokens: int = None,
    ) -> str:
        """Send a request to Bedrock and return the text response."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self.max_tokens,
            "messages": messages,
        }
        if system:
            body["system"] = system

        spinner = _Spinner()
        spinner.start()
        try:
            response = self._invoke_with_retry(body)
        finally:
            spinner.stop()
        result = json.loads(response["body"].read())
        self._last_stop_reason = result.get("stop_reason")
        return result["content"][0]["text"]

    def invoke_with_json(
        self,
        messages: list[dict],
        system: str = None,
        max_tokens: int = None,
    ) -> dict | list:
        """Send a request and parse the response as JSON.

        Strips markdown fences and retries once on parse failure.
        """
        text = self.invoke(messages, system=system, max_tokens=max_tokens)
        parsed = self._try_parse_json(text)
        if parsed is not None:
            return parsed

        # If output was truncated, retrying won't help — same input produces same length
        if self._last_stop_reason == "max_tokens":
            raise ValueError(
                f"Response truncated at max_tokens ({max_tokens or self.max_tokens}). "
                f"Output too long for the token budget. Response: {text[:300]}"
            )

        # Retry: ask Claude to fix its JSON
        logger.warning("JSON parse failed, retrying with correction prompt")
        retry_messages = messages + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. "
                    "Please respond with ONLY the valid JSON, no markdown fences or commentary."
                ),
            },
        ]
        text = self.invoke(retry_messages, system=system, max_tokens=max_tokens)
        parsed = self._try_parse_json(text)
        if parsed is not None:
            return parsed

        raise ValueError(f"Failed to parse JSON after retry. Response: {text[:500]}")

    def _invoke_with_retry(
        self,
        body: dict,
        max_retries: int = 5,
        base_delay: float = 2.0,
    ) -> dict:
        """Invoke with exponential backoff on throttling/transient errors."""
        for attempt in range(max_retries + 1):
            try:
                return self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body),
                )
            except ReadTimeoutError:
                if attempt == max_retries:
                    raise
                delay = min(base_delay * (2 ** attempt), 60)
                logger.warning(
                    "Bedrock read timeout (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                retryable = error_code in (
                    "ThrottlingException",
                    "TooManyRequestsException",
                    "ServiceUnavailableException",
                    "ModelTimeoutException",
                )
                if not retryable or attempt == max_retries:
                    raise

                delay = min(base_delay * (2 ** attempt), 60)
                logger.warning(
                    "Bedrock %s (attempt %d/%d), retrying in %.1fs",
                    error_code, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)

    @staticmethod
    def _try_parse_json(text: str) -> dict | list | None:
        """Try to parse JSON, stripping markdown fences if present."""
        cleaned = text.strip()
        # Strip ```json ... ``` fences
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
