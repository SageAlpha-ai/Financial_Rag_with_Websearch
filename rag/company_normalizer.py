import re
from typing import Optional


def normalize_company_name(name: str) -> Optional[str]:
    """Normalize a company name into a Chroma-safe collection slug.

    Pure function — **never raises**.  Returns ``None`` when the input is
    empty, ``None``, or normalizes to an empty string.  Safe for use
    inside generator expressions and ranking logic where an exception
    would collapse the entire retrieval pipeline.

    Examples::

        normalize_company_name("Tata Motors Ltd.")  # → "tata_motors"
        normalize_company_name("")                  # → None
        normalize_company_name(None)                # → None
        normalize_company_name("Ltd.")              # → None  (suffix-only)
    """
    if not name:
        return None

    name = name.lower()

    # Remove common suffixes
    name = re.sub(
        r"\b(ltd|limited|inc|corp|corporation|plc|technologies|technologies_limited)\b",
        "",
        name,
    )

    # Remove special characters (keep only alphanumeric and spaces for next step)
    name = re.sub(r"[^a-z0-9\s]", "", name)

    # Replace spaces with underscore and strip
    name = re.sub(r"\s+", "_", name.strip())

    if not name:
        return None

    return name
