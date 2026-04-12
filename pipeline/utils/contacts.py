import logging
import re

logger = logging.getLogger(__name__)


def _normalize_phone(raw: str) -> str:
    """Strip formatting from a phone number, keeping only digits.

    Removes the leading '1' for US numbers so that '+14155551234'
    and '(415) 555-1234' both normalise to '4155551234'.
    """
    digits = re.sub(r"\D", "", raw)
    # Strip leading country code '1' for US numbers (11 digits starting with 1)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits


class ContactResolver:
    """Resolve iMessage handle IDs (phone/email) to display names via macOS Contacts."""

    def __init__(self):
        self._phone_to_name: dict[str, str] = {}
        self._email_to_name: dict[str, str] = {}
        self._loaded = False
        self._load()

    def _load(self):
        try:
            from Contacts import (
                CNContactStore,
                CNContactFetchRequest,
                CNContactGivenNameKey,
                CNContactFamilyNameKey,
                CNContactPhoneNumbersKey,
                CNContactEmailAddressesKey,
            )
        except ImportError:
            logger.warning(
                "pyobjc-framework-Contacts not available; "
                "contact names will fall back to raw handle IDs"
            )
            return

        store = CNContactStore.alloc().init()

        keys = [
            CNContactGivenNameKey,
            CNContactFamilyNameKey,
            CNContactPhoneNumbersKey,
            CNContactEmailAddressesKey,
        ]
        request = CNContactFetchRequest.alloc().initWithKeysToFetch_(keys)

        def process_contact(contact, stop):
            given = contact.givenName() or ""
            family = contact.familyName() or ""
            name = f"{given} {family}".strip()
            if not name:
                return

            for phone_value in (contact.phoneNumbers() or []):
                raw = phone_value.value().stringValue()
                normalised = _normalize_phone(raw)
                if normalised:
                    self._phone_to_name[normalised] = name

            for email_value in (contact.emailAddresses() or []):
                email = email_value.value().lower().strip()
                if email:
                    self._email_to_name[email] = name

        success, error = store.enumerateContactsWithFetchRequest_error_usingBlock_(
            request, None, process_contact,
        )

        if not success:
            logger.warning("Failed to enumerate contacts: %s", error)
            return

        self._loaded = True
        logger.info(
            "Loaded %d phone and %d email mappings from Contacts",
            len(self._phone_to_name),
            len(self._email_to_name),
        )

    def resolve(self, handle_id: str) -> str:
        """Resolve a handle ID to a display name, or return the raw ID."""
        if not self._loaded:
            return handle_id

        # Try phone lookup
        normalised = _normalize_phone(handle_id)
        if normalised and normalised in self._phone_to_name:
            return self._phone_to_name[normalised]

        # Try email lookup
        lower = handle_id.lower().strip()
        if lower in self._email_to_name:
            return self._email_to_name[lower]

        return handle_id

    def resolve_all(self, handle_ids: list[str]) -> dict[str, str]:
        """Batch resolve handle IDs to display names."""
        return {h: self.resolve(h) for h in handle_ids}
