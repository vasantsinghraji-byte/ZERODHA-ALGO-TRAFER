-- Initial Database Setup
-- This runs first in docker-entrypoint-initdb.d

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Set search path
SET search_path TO public, market_data, trading, analytics;

-- Create enum types
CREATE TYPE order_transaction_type AS ENUM ('BUY', 'SELL');
CREATE TYPE order_type AS ENUM ('MARKET', 'LIMIT', 'SL', 'SL-M');
CREATE TYPE order_status AS ENUM ('PENDING', 'OPEN', 'COMPLETE', 'CANCELLED', 'REJECTED', 'MODIFIED');
CREATE TYPE order_variety AS ENUM ('regular', 'amo', 'co', 'iceberg');
CREATE TYPE product_type AS ENUM ('CNC', 'MIS', 'NRML');
CREATE TYPE position_type AS ENUM ('LONG', 'SHORT');
CREATE TYPE signal_type AS ENUM ('BUY', 'SELL', 'HOLD');

-- Instruments table (master data)
CREATE TABLE IF NOT EXISTS market_data.instruments (
    instrument_token BIGINT PRIMARY KEY,
    exchange_token BIGINT NOT NULL,
    tradingsymbol VARCHAR(50) NOT NULL,
    name VARCHAR(100),
    exchange VARCHAR(10) NOT NULL,
    segment VARCHAR(20),
    instrument_type VARCHAR(10),
    expiry DATE,
    strike DECIMAL(10, 2),
    tick_size DECIMAL(10, 4),
    lot_size INTEGER,
    last_price DECIMAL(12, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(exchange, tradingsymbol)
);

CREATE INDEX idx_instruments_symbol ON market_data.instruments(tradingsymbol);
CREATE INDEX idx_instruments_exchange ON market_data.instruments(exchange);

-- Watchlist table
CREATE TABLE IF NOT EXISTS market_data.watchlist (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    instrument_tokens BIGINT[] NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE IF NOT EXISTS trading.orders (
    order_id VARCHAR(50) PRIMARY KEY,
    parent_order_id VARCHAR(50),
    exchange_order_id VARCHAR(50),
    exchange_update_timestamp TIMESTAMP,

    -- Order details
    tradingsymbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(10) NOT NULL,
    instrument_token BIGINT,

    -- Order parameters
    transaction_type order_transaction_type NOT NULL,
    order_type order_type NOT NULL,
    order_variety order_variety NOT NULL,
    product product_type NOT NULL,

    -- Quantities
    quantity INTEGER NOT NULL,
    filled_quantity INTEGER DEFAULT 0,
    pending_quantity INTEGER,
    cancelled_quantity INTEGER DEFAULT 0,

    -- Pricing
    price DECIMAL(12, 2),
    trigger_price DECIMAL(12, 2),
    average_price DECIMAL(12, 2),

    -- Status
    status order_status NOT NULL DEFAULT 'PENDING',
    status_message TEXT,

    -- Metadata
    tag VARCHAR(50),  -- Strategy name or custom tag
    validity VARCHAR(10) DEFAULT 'DAY',
    disclosed_quantity INTEGER,

    -- Timestamps
    placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_orders_status ON trading.orders(status);
CREATE INDEX idx_orders_tradingsymbol ON trading.orders(tradingsymbol);
CREATE INDEX idx_orders_placed_at ON trading.orders(placed_at DESC);
CREATE INDEX idx_orders_tag ON trading.orders(tag);

-- Trades table (executions)
CREATE TABLE IF NOT EXISTS trading.trades (
    trade_id VARCHAR(50) PRIMARY KEY,
    order_id VARCHAR(50) REFERENCES trading.orders(order_id),
    exchange_order_id VARCHAR(50),

    tradingsymbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(10) NOT NULL,
    instrument_token BIGINT,

    transaction_type order_transaction_type NOT NULL,
    product product_type NOT NULL,

    quantity INTEGER NOT NULL,
    price DECIMAL(12, 2) NOT NULL,

    -- Calculated fields
    trade_value DECIMAL(15, 2) GENERATED ALWAYS AS (quantity * price) STORED,

    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_trades_order_id ON trading.trades(order_id);
CREATE INDEX idx_trades_executed_at ON trading.trades(executed_at DESC);

-- Positions table (current positions)
CREATE TABLE IF NOT EXISTS trading.positions (
    id SERIAL PRIMARY KEY,
    tradingsymbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(10) NOT NULL,
    instrument_token BIGINT,
    product product_type NOT NULL,

    -- Position details
    position_type position_type NOT NULL,
    quantity INTEGER NOT NULL,
    average_price DECIMAL(12, 2) NOT NULL,

    -- P&L
    last_price DECIMAL(12, 2),
    pnl DECIMAL(15, 2),
    pnl_percent DECIMAL(8, 4),

    -- Risk parameters
    stop_loss DECIMAL(12, 2),
    target DECIMAL(12, 2),

    -- Metadata
    strategy_name VARCHAR(50),
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    is_open BOOLEAN DEFAULT true,

    UNIQUE(tradingsymbol, exchange, product, is_open) WHERE is_open = true
);

CREATE INDEX idx_positions_is_open ON trading.positions(is_open);
CREATE INDEX idx_positions_strategy ON trading.positions(strategy_name);

-- Trading signals table
CREATE TABLE IF NOT EXISTS trading.signals (
    id SERIAL PRIMARY KEY,
    tradingsymbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(10) NOT NULL,
    instrument_token BIGINT,

    strategy_name VARCHAR(50) NOT NULL,
    signal_type signal_type NOT NULL,

    -- Signal parameters
    entry_price DECIMAL(12, 2),
    stop_loss DECIMAL(12, 2),
    target DECIMAL(12, 2),
    quantity INTEGER,

    -- Signal metadata
    confidence DECIMAL(5, 4),  -- 0 to 1
    indicators JSONB,  -- Store technical indicator values

    -- Execution status
    is_executed BOOLEAN DEFAULT false,
    order_id VARCHAR(50),

    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP,
    expired_at TIMESTAMP
);

CREATE INDEX idx_signals_generated_at ON trading.signals(generated_at DESC);
CREATE INDEX idx_signals_strategy ON trading.signals(strategy_name);
CREATE INDEX idx_signals_is_executed ON trading.signals(is_executed);

-- Portfolio snapshots (daily portfolio state)
CREATE TABLE IF NOT EXISTS analytics.portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL UNIQUE,

    -- Capital
    total_capital DECIMAL(15, 2) NOT NULL,
    available_margin DECIMAL(15, 2),
    utilized_margin DECIMAL(15, 2),

    -- P&L
    realized_pnl DECIMAL(15, 2) DEFAULT 0,
    unrealized_pnl DECIMAL(15, 2) DEFAULT 0,
    total_pnl DECIMAL(15, 2),

    -- Metrics
    num_trades INTEGER DEFAULT 0,
    num_winning_trades INTEGER DEFAULT 0,
    num_losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 4),

    -- Risk metrics
    max_drawdown DECIMAL(15, 2),
    sharpe_ratio DECIMAL(8, 4),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_portfolio_snapshots_date ON analytics.portfolio_snapshots(snapshot_date DESC);

-- Strategy performance table
CREATE TABLE IF NOT EXISTS analytics.strategy_performance (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(50) NOT NULL,
    evaluation_date DATE NOT NULL,

    -- Trade statistics
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 4),

    -- P&L
    total_pnl DECIMAL(15, 2) DEFAULT 0,
    avg_win DECIMAL(12, 2),
    avg_loss DECIMAL(12, 2),
    profit_factor DECIMAL(8, 4),

    -- Risk metrics
    max_drawdown DECIMAL(15, 2),
    sharpe_ratio DECIMAL(8, 4),

    -- Configuration
    parameters JSONB,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strategy_name, evaluation_date)
);

CREATE INDEX idx_strategy_performance_name ON analytics.strategy_performance(strategy_name);
CREATE INDEX idx_strategy_performance_date ON analytics.strategy_performance(evaluation_date DESC);

-- System logs table
CREATE TABLE IF NOT EXISTS public.system_logs (
    id SERIAL PRIMARY KEY,
    level VARCHAR(10) NOT NULL,  -- DEBUG, INFO, WARNING, ERROR, CRITICAL
    logger_name VARCHAR(100),
    message TEXT NOT NULL,

    -- Context
    module VARCHAR(100),
    function_name VARCHAR(100),

    -- Additional data
    extra_data JSONB,
    exception_info TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_system_logs_level ON public.system_logs(level);
CREATE INDEX idx_system_logs_created_at ON public.system_logs(created_at DESC);
CREATE INDEX idx_system_logs_module ON public.system_logs(module);

-- OHLCV data table
CREATE TABLE market_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(30) NOT NULL,
    open NUMERIC(10,2) NOT NULL,
    high NUMERIC(10,2) NOT NULL,
    low NUMERIC(10,2) NOT NULL,
    close NUMERIC(10,2) NOT NULL,
    volume INTEGER NOT NULL
);

-- Convert to hypertable
SELECT create_hypertable('market_data', 'timestamp');

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at
CREATE TRIGGER update_instruments_updated_at BEFORE UPDATE ON market_data.instruments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_watchlist_updated_at BEFORE UPDATE ON market_data.watchlist
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON trading.orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON trading.positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA market_data TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA market_data TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO trader;
