import os
from pathlib import Path
from typing import Dict, Optional
import yaml

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    # Environment
    ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://trader:trader123@localhost:5432/algotrader"
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0"
    )
    
    # Zerodha API
    ZERODHA_API_KEY: Optional[str] = Field(default=None)
    ZERODHA_API_SECRET: Optional[str] = Field(default=None)
    ZERODHA_ACCESS_TOKEN: Optional[str] = Field(default=None)
    
    # Feature flags
    ENABLE_PAPER_TRADING: bool = True
    ENABLE_LIVE_TRADING: bool = False
    
    # Trading parameters
    MAX_POSITION_SIZE: float = 100000.0
    RISK_PER_TRADE: float = 0.01

    @field_validator('ZERODHA_API_KEY', 'ZERODHA_API_SECRET', 'ZERODHA_ACCESS_TOKEN')
    @classmethod
    def validate_zerodha_credentials(cls, v: Optional[str], info) -> Optional[str]:
        field_name = info.field_name
        # Use Pydantic's validation context, not os.getenv which may not reflect .env file values
        env_value = info.data.get('ENV', 'development')
        if v is None and env_value == 'production':
            raise ValueError(f'{field_name} must be set in production environment')
        return v

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "Settings":
        """Load settings from YAML file"""
        with open(yaml_path) as f:
            config_data: Dict = yaml.safe_load(f)
        return cls(**config_data)

settings = Settings()
