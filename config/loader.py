"""
Configuration Loader Module
Loads settings, secrets, and watchlists with environment variable substitution.

Supports:
- settings.yaml: Application settings
- secrets.yaml: API keys and credentials
- watchlist.yaml: Dynamic symbol lists for trading
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from box import Box

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """
    Raised when configuration file is corrupted or contains invalid values.

    This error is raised explicitly instead of being silently swallowed,
    so users know exactly what went wrong with their config file.
    """
    pass


class ConfigLoader:
    """Load and manage application configuration including watchlists."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.settings: Optional[Dict] = None
        self.secrets: Optional[Dict] = None
        self.watchlist: Optional[Dict] = None
        self._config: Optional[Box] = None
        self._watchlist_config: Optional[Box] = None

    def load(self) -> Box:
        """
        Load configuration from YAML files.

        Raises:
            ConfigurationError: If settings.yaml is missing or any config file is malformed.
            FileNotFoundError: If settings.yaml doesn't exist.
        """
        # Load settings (required)
        settings_path = self.config_dir / "settings.yaml"
        try:
            with open(settings_path, "r") as f:
                settings_raw = f.read()
                settings_substituted = self._substitute_env_vars(settings_raw)
                self.settings = yaml.safe_load(settings_substituted)
        except FileNotFoundError:
            raise ConfigurationError(
                f"Required configuration file not found: {settings_path}\n"
                f"Please create it from settings.yaml.example"
            )
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML syntax in {settings_path}: {e}\n"
                f"Please fix the file or restore from settings.yaml.example"
            )

        # Load secrets (optional file, but must be valid if present)
        # NOTE: For credentials, prefer using config.config.settings (Pydantic)
        # which loads from environment variables / .env file.
        secrets_path = self.config_dir / "secrets.yaml"
        if secrets_path.exists():
            try:
                with open(secrets_path, "r") as f:
                    self.secrets = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                # FAIL FAST: Malformed secrets file is a serious error
                raise ConfigurationError(
                    f"Invalid YAML syntax in {secrets_path}: {e}\n"
                    f"The secrets file exists but is corrupted. "
                    f"Please fix it or delete it to use environment variables instead."
                )
            except PermissionError as e:
                raise ConfigurationError(
                    f"Permission denied reading {secrets_path}: {e}\n"
                    f"Check file permissions."
                )
        else:
            # Missing secrets.yaml is OK - use env vars via config.config.settings
            logger.info(
                f"No secrets.yaml found at {secrets_path}. "
                f"Using environment variables for credentials (recommended)."
            )
            self.secrets = {}

        # Merge configurations
        merged_config = self._merge_configs(self.settings, self.secrets)

        # Convert to Box for dot notation access
        self._config = Box(merged_config, frozen_box=False)

        return self._config

    def _substitute_env_vars(self, text: str) -> str:
        """
        Substitute environment variables in format ${VAR_NAME} or ${VAR_NAME:-default}

        Examples:
            ${ENV} -> value of ENV
            ${ENV:-development} -> value of ENV, or 'development' if not set
        """
        pattern = r'\$\{([^}:]+)(?::[-]([^}]+))?\}'

        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.environ.get(var_name, default_value)

        return re.sub(pattern, replacer, text)

    def _merge_configs(self, settings: Dict, secrets: Dict) -> Dict:
        """Deep merge settings and secrets dictionaries"""
        merged = settings.copy()

        for key, value in secrets.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key_path: Dot-separated path (e.g., 'zerodha.api_key')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if self._config is None:
            self.load()

        try:
            keys = key_path.split('.')
            value: Any = self._config
            for key in keys:
                if value is None:
                    return default
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def reload(self):
        """Reload configuration from files"""
        self.load()
        self.load_watchlist()

    def load_watchlist(self) -> Box:
        """
        Load watchlist configuration from YAML file.

        Returns:
            Box object with watchlist configuration
        """
        watchlist_path = self.config_dir / "watchlist.yaml"

        if not watchlist_path.exists():
            logger.warning(f"Watchlist file not found: {watchlist_path}")
            self.watchlist = {"active_watchlist": "custom", "custom": []}
            self._watchlist_config = Box(self.watchlist, frozen_box=False)
            return self._watchlist_config

        try:
            with open(watchlist_path, "r") as f:
                self.watchlist = yaml.safe_load(f)

            self._watchlist_config = Box(self.watchlist, frozen_box=False)
            logger.info(f"Loaded watchlist from {watchlist_path}")
            return self._watchlist_config

        except Exception as e:
            logger.error(f"Failed to load watchlist: {e}")
            self.watchlist = {"active_watchlist": "custom", "custom": []}
            self._watchlist_config = Box(self.watchlist, frozen_box=False)
            return self._watchlist_config

    def get_active_symbols(self) -> List[str]:
        """
        Get the list of symbols from the active watchlist.

        Returns:
            List of symbol strings (e.g., ["NSE:RELIANCE", "NSE:TCS"])
        """
        if self._watchlist_config is None:
            self.load_watchlist()

        active_name = self._watchlist_config.get("active_watchlist", "custom")

        # Handle "all" - combine all watchlists
        if active_name == "all":
            all_symbols = set()
            for key in ["nifty50", "banknifty", "custom", "fno"]:
                if key in self._watchlist_config:
                    symbols = self._watchlist_config.get(key, [])
                    if symbols:
                        all_symbols.update(symbols)
            return list(all_symbols)

        # Get specific watchlist
        symbols = self._watchlist_config.get(active_name, [])
        if not symbols:
            logger.warning(f"Watchlist '{active_name}' is empty or not found")
            return []

        return list(symbols)

    def get_instrument_tokens(self) -> Dict[str, int]:
        """
        Get the instrument token mapping for WebSocket subscriptions.

        Returns:
            Dict mapping "EXCHANGE:SYMBOL" to instrument token
        """
        if self._watchlist_config is None:
            self.load_watchlist()

        return dict(self._watchlist_config.get("instrument_tokens", {}))

    def get_symbol_overrides(self, symbol: str) -> Dict[str, Any]:
        """
        Get strategy parameter overrides for a specific symbol.

        Args:
            symbol: Symbol string (e.g., "NSE:RELIANCE")

        Returns:
            Dict of parameter overrides
        """
        if self._watchlist_config is None:
            self.load_watchlist()

        overrides = self._watchlist_config.get("symbol_overrides", {})
        return dict(overrides.get(symbol, {}))

    def get_scanner_config(self) -> Dict[str, Any]:
        """
        Get scanner configuration settings.

        Returns:
            Dict with scanner settings
        """
        if self._watchlist_config is None:
            self.load_watchlist()

        return dict(self._watchlist_config.get("scanner", {}))


# Global configuration instance
_config_loader = ConfigLoader()


def get_config() -> Box:
    """
    Get the global configuration instance.

    DEPRECATED: Use 'from config.config import settings' instead.

    The Pydantic-based settings provides:
    - Type safety and validation
    - Automatic .env file loading
    - Better IDE support

    Example migration:
        # Old (deprecated):
        from config import get_config
        config = get_config()
        api_key = config.zerodha.api_key

        # New (recommended):
        from config.config import settings
        api_key = settings.ZERODHA_API_KEY
    """
    import warnings
    warnings.warn(
        "get_config() is deprecated. Use 'from config.config import settings' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if _config_loader._config is None:
        return _config_loader.load()
    return _config_loader._config


def reload_config():
    """Reload configuration from files"""
    _config_loader.reload()
    return _config_loader._config


# =============================================================================
# Watchlist Helper Functions
# =============================================================================

def get_watchlist() -> Box:
    """Get the watchlist configuration."""
    if _config_loader._watchlist_config is None:
        return _config_loader.load_watchlist()
    return _config_loader._watchlist_config


def get_active_symbols() -> List[str]:
    """
    Get symbols from the active watchlist.

    Returns:
        List of symbols (e.g., ["NSE:RELIANCE", "NSE:TCS"])

    Example:
        >>> from config.loader import get_active_symbols
        >>> symbols = get_active_symbols()
        >>> print(symbols[:3])
        ['NSE:RELIANCE', 'NSE:TCS', 'NSE:HDFCBANK']
    """
    return _config_loader.get_active_symbols()


def get_instrument_tokens() -> Dict[str, int]:
    """
    Get instrument token mapping for WebSocket subscriptions.

    Returns:
        Dict mapping symbol to token (e.g., {"NSE:RELIANCE": 738561})
    """
    return _config_loader.get_instrument_tokens()


def get_symbol_overrides(symbol: str) -> Dict[str, Any]:
    """
    Get strategy parameter overrides for a specific symbol.

    Args:
        symbol: Symbol string (e.g., "NSE:RELIANCE")

    Returns:
        Dict of parameter overrides
    """
    return _config_loader.get_symbol_overrides(symbol)


def get_scanner_config() -> Dict[str, Any]:
    """Get scanner configuration."""
    return _config_loader.get_scanner_config()


def reload_watchlist() -> Box:
    """Reload watchlist from file (useful for hot-reloading)."""
    return _config_loader.load_watchlist()
