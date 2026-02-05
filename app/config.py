"""
Configuration Management
Simple settings that anyone can understand.

SECURITY NOTE:
==============
Zerodha API credentials (api_key, api_secret, access_token) should NEVER be
stored in configuration files. Always use environment variables:

    export ZERODHA_API_KEY="your_api_key"
    export ZERODHA_API_SECRET="your_api_secret"
    export ZERODHA_ACCESS_TOKEN="your_access_token"

Or create a .env file (which should be in .gitignore):

    ZERODHA_API_KEY=your_api_key
    ZERODHA_API_SECRET=your_api_secret
    ZERODHA_ACCESS_TOKEN=your_access_token

This prevents accidental credential exposure through git commits or file sharing.
"""

import os
import yaml
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Configure logger
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """
    Raised when configuration file is corrupted or contains invalid values.

    This error is raised explicitly instead of being silently swallowed,
    so users know exactly what went wrong with their config file.
    """
    pass

# Base directory
BASE_DIR = Path(__file__).parent.parent

@dataclass
class ZerodhaConfig:
    """Zerodha API Configuration"""
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""

@dataclass
class TradingConfig:
    """Trading Settings - Easy to understand"""
    # Money Settings
    capital: float = 100000.0          # How much money to trade with
    risk_per_trade: float = 2.0        # Max 2% loss per trade
    max_daily_loss: float = 5.0        # Stop if lose 5% in a day

    # Position Settings
    max_positions: int = 5             # Max 5 stocks at once
    default_quantity: int = 1          # Default shares to buy

    # Time Settings
    market_start: str = "09:15"        # Market opens
    market_end: str = "15:30"          # Market closes

    # Safety Settings
    paper_trading: bool = True         # Practice mode (no real money)
    auto_stop_loss: bool = True        # Always use stop-loss

@dataclass
class UIConfig:
    """User Interface Settings"""
    theme: str = "dark"                # dark or light
    font_size: int = 12                # Text size
    show_tooltips: bool = True         # Help tips
    sound_alerts: bool = True          # Beep on trades

@dataclass
class AppConfig:
    """Main Application Configuration"""
    zerodha: ZerodhaConfig = field(default_factory=ZerodhaConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'AppConfig':
        """
        Load configuration from file.

        SECURITY: Credentials are loaded ONLY from environment variables,
        not from the config file. Any credentials in the config file are
        ignored and a security warning is issued.
        """
        if config_path is None:
            config_path = BASE_DIR / "config" / "settings.yaml"

        config = cls()

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f) or {}

                # SECURITY: Check for credentials in file and warn
                if 'zerodha' in data:
                    zerodha_data = data['zerodha']
                    has_creds = any(
                        zerodha_data.get(k)
                        for k in ('api_key', 'api_secret', 'access_token')
                    )
                    if has_creds:
                        warnings.warn(
                            f"\n"
                            f"╔══════════════════════════════════════════════════════════════╗\n"
                            f"║  SECURITY WARNING: Credentials found in {config_path.name}!  ║\n"
                            f"╠══════════════════════════════════════════════════════════════╣\n"
                            f"║  Credentials in config files are IGNORED for security.       ║\n"
                            f"║  Please remove them and use environment variables instead:   ║\n"
                            f"║                                                              ║\n"
                            f"║    export ZERODHA_API_KEY='your_key'                         ║\n"
                            f"║    export ZERODHA_API_SECRET='your_secret'                   ║\n"
                            f"║    export ZERODHA_ACCESS_TOKEN='your_token'                  ║\n"
                            f"║                                                              ║\n"
                            f"║  Or use a .env file (add to .gitignore!)                     ║\n"
                            f"╚══════════════════════════════════════════════════════════════╝\n",
                            SecurityWarning,
                            stacklevel=2
                        )
                    # NOTE: We intentionally do NOT load zerodha credentials from file

                # Load Trading settings
                if 'trading' in data:
                    try:
                        config.trading = TradingConfig(**data['trading'])
                    except TypeError as e:
                        raise ConfigurationError(
                            f"Invalid trading configuration in {config_path}: {e}\n"
                            f"Check that all values in the 'trading' section are valid."
                        ) from e

                # Load UI settings
                if 'ui' in data:
                    try:
                        config.ui = UIConfig(**data['ui'])
                    except TypeError as e:
                        raise ConfigurationError(
                            f"Invalid UI configuration in {config_path}: {e}\n"
                            f"Check that all values in the 'ui' section are valid."
                        ) from e

            except yaml.YAMLError as e:
                # YAML syntax error - config file is corrupted
                raise ConfigurationError(
                    f"Failed to parse config file {config_path}: {e}\n"
                    f"The YAML syntax is invalid. Please fix the file or delete it to use defaults."
                ) from e
            except PermissionError as e:
                # File exists but can't be read
                raise ConfigurationError(
                    f"Permission denied reading config file {config_path}: {e}\n"
                    f"Check file permissions or run with appropriate privileges."
                ) from e
            except ConfigurationError:
                # Re-raise our own errors
                raise
            except Exception as e:
                # Unexpected error - log and re-raise with context
                logger.error(f"Unexpected error loading config from {config_path}: {e}")
                raise ConfigurationError(
                    f"Unexpected error loading config from {config_path}: {e}\n"
                    f"This may indicate a bug in the configuration loader."
                ) from e

        # Load credentials ONLY from environment variables
        config.zerodha.api_key = os.getenv('ZERODHA_API_KEY', '')
        config.zerodha.api_secret = os.getenv('ZERODHA_API_SECRET', '')
        config.zerodha.access_token = os.getenv('ZERODHA_ACCESS_TOKEN', '')

        return config

    def save(self, config_path: Optional[Path] = None) -> None:
        """
        Save configuration to file.

        SECURITY: Credentials (api_key, api_secret, access_token) are NEVER saved
        to disk. They must be provided via environment variables:
          - ZERODHA_API_KEY
          - ZERODHA_API_SECRET
          - ZERODHA_ACCESS_TOKEN

        This prevents accidental credential exposure through:
          - Git commits
          - File sharing
          - Backup systems
          - Log files
        """
        if config_path is None:
            config_path = BASE_DIR / "config" / "settings.yaml"

        config_path.parent.mkdir(parents=True, exist_ok=True)

        # SECURITY FIX: Do NOT save zerodha credentials to file
        # Credentials should ONLY come from environment variables
        data = {
            # NOTE: 'zerodha' section intentionally omitted - use env vars for credentials
            'trading': {
                'capital': self.trading.capital,
                'risk_per_trade': self.trading.risk_per_trade,
                'max_daily_loss': self.trading.max_daily_loss,
                'max_positions': self.trading.max_positions,
                'default_quantity': self.trading.default_quantity,
                'market_start': self.trading.market_start,
                'market_end': self.trading.market_end,
                'paper_trading': self.trading.paper_trading,
                'auto_stop_loss': self.trading.auto_stop_loss,
            },
            'ui': {
                'theme': self.ui.theme,
                'font_size': self.ui.font_size,
                'show_tooltips': self.ui.show_tooltips,
                'sound_alerts': self.ui.sound_alerts,
            }
        }

        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


# Global config instance
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get the global configuration"""
    global _config
    if _config is None:
        _config = AppConfig.load()
    return _config

def save_config() -> None:
    """Save the global configuration"""
    global _config
    if _config is not None:
        _config.save()
