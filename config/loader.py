"""
Configuration Loader Module
Loads settings and secrets with environment variable substitution
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from box import Box


class ConfigLoader:
    """Load and manage application configuration"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.settings: Optional[Dict] = None
        self.secrets: Optional[Dict] = None
        self._config: Optional[Box] = None

    def load(self) -> Box:
        """Load configuration from YAML files"""
        # Load settings
        settings_path = self.config_dir / "settings.yaml"
        with open(settings_path, "r") as f:
            settings_raw = f.read()
            settings_substituted = self._substitute_env_vars(settings_raw)
            self.settings = yaml.safe_load(settings_substituted)

        # Load secrets (if exists)
        secrets_path = self.config_dir / "secrets.yaml"
        if secrets_path.exists():
            with open(secrets_path, "r") as f:
                self.secrets = yaml.safe_load(f)
        else:
            print(f"Warning: {secrets_path} not found. Using example template.")
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


# Global configuration instance
_config_loader = ConfigLoader()


def get_config() -> Box:
    """Get the global configuration instance"""
    if _config_loader._config is None:
        return _config_loader.load()
    return _config_loader._config


def reload_config():
    """Reload configuration from files"""
    _config_loader.reload()
    return _config_loader._config
