import re

def normalize_company_name(name: str) -> str:
    """
    Normalizes a company name into a Chroma-safe collection name.
    Example: "Tata Motors Ltd." -> "tata_motors"
    """
    if not name:
        raise ValueError("Company name could not be resolved.")

    name = name.lower()

    # Remove common suffixes
    name = re.sub(r"\b(ltd|limited|inc|corp|corporation|plc|technologies|technologies_limited)\b", "", name)

    # Remove special characters (keep only alphanumeric and spaces for next step)
    name = re.sub(r"[^a-z0-9\s]", "", name)

    # Replace spaces with underscore and strip
    name = re.sub(r"\s+", "_", name.strip())

    if not name:
        raise ValueError("Company name could not be resolved.")

    return name
