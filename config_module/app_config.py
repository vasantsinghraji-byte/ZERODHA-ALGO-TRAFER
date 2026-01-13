"""Application configuration and secrets management"""

from pathlib import Path
import yaml

# Constants
CONFIG_DIR = Path(__file__).parent.parent / "config"
BOT_API_URL = "http://localhost:8000"


def load_secrets():
    """Load API secrets from config/secrets.yaml

    Returns:
        dict: Dictionary containing api_key, api_secret, and bot_api_key
    """
    try:
        with open(CONFIG_DIR / 'secrets.yaml', 'r') as f:
            secrets = yaml.safe_load(f)
        return {
            'api_key': secrets.get('zerodha', {}).get('api_key', ''),
            'api_secret': secrets.get('zerodha', {}).get('api_secret', ''),
            'bot_api_key': secrets.get('api_secret_key', '')
        }
    except Exception as e:
        print(f"⚠️ Error loading secrets: {e}")
        return {'api_key': '', 'api_secret': '', 'bot_api_key': ''}


def load_settings():
    """Load application settings from config/settings.yaml

    Returns:
        dict: Application settings
    """
    try:
        with open(CONFIG_DIR / 'settings.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {
            'capital': 100000,
            'risk_per_trade': 1.0,
            'features': {
                'paper_trading': True,
                'live_trading': False
            }
        }


# Load secrets at module import
SECRETS = load_secrets()
API_KEY = SECRETS['api_key']
API_SECRET = SECRETS['api_secret']
BOT_API_KEY = SECRETS['bot_api_key']
