# Config package
"""
Configuration Management for AlgoTrader Pro
============================================

ARCHITECTURE (Configuration Redundancy Fix):
--------------------------------------------
This project has TWO distinct configuration systems for different purposes:

1. INFRASTRUCTURE SETTINGS (config/config.py - Pydantic)
   -------------------------------------------------------
   Use for: Database URLs, Redis, API credentials, feature flags
   Import:  from config.config import settings

   Features:
   - Type-safe with Pydantic validation
   - Automatic .env file loading
   - Environment variable support
   - Production credential validation

   Example:
       from config.config import settings
       db_url = settings.DATABASE_URL
       api_key = settings.ZERODHA_API_KEY

2. WATCHLIST MANAGEMENT (config/loader.py - YAML+Box)
   ---------------------------------------------------
   Use for: Trading symbols, instrument tokens, scanner config
   Import:  from config import get_active_symbols, get_watchlist

   Features:
   - Dynamic symbol lists (nifty50, banknifty, custom)
   - Instrument token mapping for WebSocket
   - Symbol-specific parameter overrides
   - Hot-reload support

   Example:
       from config import get_active_symbols, get_watchlist
       symbols = get_active_symbols()  # ['NSE:RELIANCE', 'NSE:TCS', ...]

DEPRECATED:
-----------
- config_module/ - DELETED (was unused duplicate)
- get_config() from loader.py - Use settings from config.config instead

For UI/Trading settings, see: app/config.py (TradingConfig, UIConfig)
"""

from config.loader import (
    # Error class for configuration failures
    ConfigurationError,
    # DEPRECATED: Use 'from config.config import settings' instead
    get_config,
    reload_config,
    # Watchlist functions (primary use of this module)
    get_watchlist,
    get_active_symbols,
    get_instrument_tokens,
    get_symbol_overrides,
    get_scanner_config,
    reload_watchlist,
    ConfigLoader,
)

# Also expose the Pydantic settings for convenience
from config.config import settings, Settings

__all__ = [
    # Pydantic settings (RECOMMENDED for infrastructure config)
    'settings',
    'Settings',
    # Error class for configuration failures
    'ConfigurationError',
    # Legacy YAML config (DEPRECATED - use settings instead)
    'get_config',
    'reload_config',
    # Watchlist management (primary use case)
    'get_watchlist',
    'get_active_symbols',
    'get_instrument_tokens',
    'get_symbol_overrides',
    'get_scanner_config',
    'reload_watchlist',
    'ConfigLoader',
]
