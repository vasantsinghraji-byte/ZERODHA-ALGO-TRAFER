"""
Configuration Management
Simple settings that anyone can understand.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

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
        """Load configuration from file"""
        if config_path is None:
            config_path = BASE_DIR / "config" / "settings.yaml"

        config = cls()

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f) or {}

                # Load Zerodha settings
                if 'zerodha' in data:
                    config.zerodha = ZerodhaConfig(**data['zerodha'])

                # Load Trading settings
                if 'trading' in data:
                    config.trading = TradingConfig(**data['trading'])

                # Load UI settings
                if 'ui' in data:
                    config.ui = UIConfig(**data['ui'])

            except Exception as e:
                print(f"Warning: Could not load config: {e}")

        # Override with environment variables
        config.zerodha.api_key = os.getenv('ZERODHA_API_KEY', config.zerodha.api_key)
        config.zerodha.api_secret = os.getenv('ZERODHA_API_SECRET', config.zerodha.api_secret)

        return config

    def save(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to file"""
        if config_path is None:
            config_path = BASE_DIR / "config" / "settings.yaml"

        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'zerodha': {
                'api_key': self.zerodha.api_key,
                'api_secret': self.zerodha.api_secret,
                'access_token': self.zerodha.access_token,
            },
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
