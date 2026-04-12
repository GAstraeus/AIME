import json
import logging
import re
import time

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, ReadTimeoutError

from pipeline.utils.config import get_config

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimate at ~4 characters per token."""
    return len(text) // 4


class BedrockClient:
    """Wrapper around AWS Bedrock invoke_model with retry and JSON parsing."""

    def __init__(self, model_id: str = None, region: str = None, max_tokens: int = None):
        config = get_config()
        self.model_id = model_id or config["BEDROCK_MODEL_ID"]
        self.max_tokens = max_tokens or config["MAX_TOKENS"]
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

        response = self._invoke_with_retry(body)
        result = json.loads(response["body"].read())
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
