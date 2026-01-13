# Zerodha Algorithmic Trading Bot

A personal algorithmic trading system for Zerodha Kite Connect API with real-time market data processing, risk management, and automated execution.

## Features

- Real-time market data streaming via WebSocket
- Multiple trading strategies (Opening Range Breakout, Mean Reversion, etc.)
- Sophisticated risk management with position sizing
- Order management system (OMS)
- Time-series database for OHLC data
- Redis caching for real-time state
- Telegram notifications
- Backtesting framework
- Paper trading mode

## Architecture

```
zerodha-algo-trader/
├── data/           # Market data processing and storage
├── research/       # Strategy research and backtesting
├── oms/            # Order Management System
├── risk/           # Risk management module
├── ui/             # Dashboard and monitoring
├── config/         # Configuration and secrets
└── infrastructure/ # Docker, database schemas
```

## Tech Stack

- **Language**: Python 3.11+
- **Database**: PostgreSQL with TimescaleDB extension
- **Cache**: Redis
- **API**: Zerodha Kite Connect
- **Deployment**: Docker + Docker Compose

## Quick Start

### Phase 0 Status: Setting Up Infrastructure

Currently implementing foundational architecture:
- Basic project structure
- Docker infrastructure (PostgreSQL + TimescaleDB, Redis)
- Configuration management
- Database schemas
- Logging setup

Some features mentioned above will be implemented in later phases.

### Current Status: Phase 1 - Market Data Infrastructure

Completed:
- Basic project structure ✅
- Docker infrastructure ✅
- Database schemas ✅
- Zerodha API connection ✅

In Progress:
- Real-time market data streaming
- Historical data collection
- Data validation and storage
- Redis caching implementation

Next Steps:
1. Implement WebSocket connection for real-time data
2. Set up tick data processing pipeline
3. Create historical data downloader
4. Implement data validation and cleanup
5. Add Redis caching layer
6. Create basic market data APIs

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Zerodha Kite Connect API credentials

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd zerodha-algo-trader
```

2. Create environment configuration:
```bash
cp .env.example .env
# Edit .env with your API credentials

# Generate Zerodha access token:
# 1. Go to https://kite.zerodha.com/connect/login?api_key=YOUR_API_KEY
# 2. Login and authorize the app
# 3. Copy the request_token from the redirect URL
# 4. Run: python scripts/generate_token.py --request_token=YOUR_REQUEST_TOKEN
```

3. Start infrastructure:
```bash
docker-compose up -d
```

4. Install Python dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

5. Initialize database:
```bash
python scripts/init_db.py
```

6. Run the bot:
```bash
python main.py
```

## Installation

#### Automated Installation (Recommended)

On Unix-like systems:
```bash
./scripts/install.sh
```

On Windows:
```bash
scripts\install.bat
```

#### Manual Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install package:
```bash
pip install -e ".[dev]"
```

3. Start infrastructure:
```bash
docker-compose up -d
```

4. Initialize database:
```bash
python scripts/init_db.py
```

## Configuration

See [config/README.md](config/README.md) for detailed configuration options.

## Environment Variables

- `ENV`: Environment (development/staging/production)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `ZERODHA_API_KEY`: Kite Connect API key
- `ZERODHA_API_SECRET`: Kite Connect API secret
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
isort .
```

### Linting
```bash
flake8 .
mypy .
```

### GitHub Copilot Integration

This project is configured to use GitHub Copilot for AI-assisted development. Configuration is available in `.vscode/settings.json`.

To use Copilot:
1. Install GitHub Copilot extension in VSCode
2. Sign in with your GitHub account
3. Ensure you have an active Copilot subscription

## Safety & Risk Warnings

- Start with paper trading mode
- Never risk more than 1-2% per trade
- Set daily loss limits
- Monitor bot actively during market hours
- Keep detailed logs for compliance
- Understand SEBI regulations

## License

MIT License - For personal use only

## Disclaimer

This software is for educational and personal use only. Trading involves substantial risk. Past performance does not guarantee future results. Use at your own risk.
