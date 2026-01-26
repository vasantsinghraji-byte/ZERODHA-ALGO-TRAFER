# Legacy Code - DO NOT USE

This folder contains the **old** architecture that has been replaced by the Event-Driven Architecture.

## Why This Was Deprecated

The old `src/zerodha_trader/` package:
- Used a separate DI container (`AppContainer`)
- Had its own `TradingBot` class with callback-based architecture
- Was disconnected from the new Event-Driven Engine

## The New Architecture

The unified Event-Driven Architecture is now in:
- `core/` - Event bus, trading engine, data sources
- `ui/` - GUI connected to EventBus
- `strategies/` - Strategies that work with events
- `run.py` - Single entry point

## Do NOT Import From This Folder

These files are kept for reference only. They are not maintained and may be deleted in future versions.

If you need functionality from here, migrate it to the new architecture:
1. Create components in `core/`
2. Use `EventBus` for communication
3. Subscribe to events instead of using callbacks
