"""Config module."""
from .settings import get_config, load_config, validate_config, AppConfig
from .settings import PRIORITY_WEB_DOMAINS, ALLOWED_FALLBACK_DOMAINS, BLOCKED_DOMAINS
from .settings import DEBUG_MODE