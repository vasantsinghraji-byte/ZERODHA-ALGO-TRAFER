# Zerodha Algo Trader Codebase Overview

## Core Components

### Data Layer (`/data`)
- Market data collection via Zerodha WebSocket
- OHLCV data processing and storage
- Real-time tick data handling
- Redis caching for performance

### Strategy Layer (`/strategy`)
- Base strategy framework
- Signal generation system
- Strategy manager for multiple strategies
- Example implementations (Moving Average)

### Order Management (`/oms`)
- Order execution engine
- Position tracking
- Transaction management
- Paper trading simulation

### Risk Management (`/risk`)
- Position sizing
- Risk per trade calculation
- Daily loss limits
- Exposure management

### Infrastructure (`/infrastructure`)
- TimescaleDB setup
- Redis connection pooling
- Logging configuration
- Database migrations

## Key Implementation Details

### Database Schema
```sql
market_data.instruments  # Instrument master data
market_data.ohlcv       # Time-series price data
trading.orders          # Order management
trading.positions       # Position tracking
trading.signals        # Strategy signals
analytics.*            # Performance metrics
```

### Caching Strategy
- Real-time market data: 5-minute TTL
- Instrument data: 1-day TTL
- Order status: No TTL, explicit invalidation
- Position cache: Real-time sync with DB

### Performance Considerations
- Connection pooling for DB and Redis
- Batch operations for tick data
- Indexed time-series queries
- WebSocket connection management

### Testing Strategy
```
tests/
├── unit/             # Unit tests
├── integration/      # Integration tests
└── conftest.py      # Test fixtures
```

## Development Guidelines

### Code Style
- Type hints required
- Async where appropriate
- Exception handling required
- Logging for key operations

### Documentation
- Docstrings for public APIs
- README for each module
- Configuration examples
- Inline comments for complexity

### Best Practices
- Use dependency injection
- Follow SOLID principles
- Write idempotent operations
- Handle edge cases explicitly

## Deployment

### Local Development
```bash
docker-compose up -d
python -m venv venv
pip install -e ".[dev]"
```

### Production
- Use Docker containers
- Configure environment variables
- Set up monitoring
- Enable error reporting

## Future Improvements

1. Add more strategy examples
2. Implement backtesting engine
3. Add web dashboard
4. Enhance risk management
5. Add more unit tests
