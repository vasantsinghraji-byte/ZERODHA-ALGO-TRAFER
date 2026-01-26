# -*- coding: utf-8 -*-
"""
FastAPI Application - Remote Control & Monitoring API
Provides REST endpoints for monitoring and controlling the trading bot
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from zerodha_trader.core.trading_bot import TradingBot
from zerodha_trader.core.container import AppContainer
from zerodha_trader.core.config import Settings
from .security import create_api_key_dependency

logger = logging.getLogger(__name__)

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class StatusResponse(BaseModel):
    """Bot status response"""
    is_running: bool
    uptime_seconds: Optional[float] = None
    environment: str
    live_trading: bool
    subscribed_instruments: List[int]
    active_orders_count: int
    open_positions_count: int
    signals_generated: int
    trades_executed: int


class PositionResponse(BaseModel):
    """Position information"""
    symbol: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    pnl_percentage: float


class OrderResponse(BaseModel):
    """Order information"""
    order_id: str
    symbol: str
    transaction_type: str
    quantity: int
    price: Optional[float]
    order_type: str
    status: str
    timestamp: str


class PnLResponse(BaseModel):
    """P&L metrics"""
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: float
    average_loss: float


class RiskMetricsResponse(BaseModel):
    """Risk metrics"""
    account_balance: float
    current_exposure: float
    exposure_percentage: float
    max_position_size: float
    max_risk_per_trade: float
    max_daily_loss: float
    daily_pnl: float
    open_positions: int
    max_open_positions: int
    trades_today: int
    max_trades_per_day: int


class MessageResponse(BaseModel):
    """Generic message response"""
    message: str
    success: bool


# ============================================================================
# API APPLICATION
# ============================================================================

def create_app(bot: TradingBot, container: AppContainer) -> FastAPI:
    """
    Create FastAPI application with bot endpoints

    Args:
        bot: TradingBot instance to monitor and control
        container: AppContainer with all services

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Zerodha Algo Trader API",
        description="Remote monitoring and control for the trading bot",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS middleware - allow all origins in development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store references
    app.state.bot = bot
    app.state.container = container
    app.state.start_time = datetime.now()

    # Create API key dependency (if configured)
    api_key_auth = create_api_key_dependency(container.settings)

    # Log security status
    if container.settings.api_secret_key:
        logger.info("API authentication enabled - all endpoints protected with API key")
    else:
        logger.warning("API authentication disabled - consider setting API_SECRET_KEY for security")

    # ========================================================================
    # MONITORING ENDPOINTS
    # ========================================================================

    @app.get("/", response_model=MessageResponse)
    async def root():
        """API root endpoint"""
        return MessageResponse(
            message="Zerodha Algo Trader API - Use /docs for API documentation",
            success=True
        )

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    @app.get("/status", response_model=StatusResponse)
    async def get_status(api_key: str = Depends(api_key_auth)):
        """Get bot status and statistics"""
        uptime = None
        if bot.is_running:
            uptime = (datetime.now() - app.state.start_time).total_seconds()

        # Get active orders and positions
        active_orders = bot.order_manager.get_active_orders()

        # Get signal and trade counts from storage
        signals_count = 0
        trades_count = 0
        try:
            if hasattr(bot.storage, 'get_signals'):
                signals = bot.storage.get_signals(limit=10000)  # type: ignore
                signals_count = len(signals)

            if hasattr(bot.storage, 'get_trades'):
                trades = bot.storage.get_trades(limit=10000)  # type: ignore
                trades_count = len(trades)
        except Exception as e:
            logger.warning(f"Failed to get counts from storage: {e}")

        return StatusResponse(
            is_running=bot.is_running,
            uptime_seconds=uptime,
            environment=container.settings.environment.value,
            live_trading=container.settings.live_trading,
            subscribed_instruments=list(bot.symbols.keys()) if bot.symbols else [],
            active_orders_count=len(active_orders),
            open_positions_count=len(bot.order_manager.positions),
            signals_generated=signals_count,
            trades_executed=trades_count
        )

    @app.get("/positions", response_model=List[PositionResponse])
    async def get_positions(api_key: str = Depends(api_key_auth)):
        """Get current open positions"""
        positions = []

        for symbol, pos_data in bot.order_manager.positions.items():
            # Calculate P&L (simplified - would need current market price)
            avg_price = pos_data.get('average_price', 0)
            quantity = pos_data.get('quantity', 0)
            last_price = pos_data.get('last_price', avg_price)  # Use avg if last not available

            pnl = (last_price - avg_price) * quantity
            pnl_pct = ((last_price - avg_price) / avg_price * 100) if avg_price > 0 else 0

            positions.append(PositionResponse(
                symbol=symbol,
                quantity=quantity,
                average_price=avg_price,
                last_price=last_price,
                pnl=pnl,
                pnl_percentage=pnl_pct
            ))

        return positions

    @app.get("/orders", response_model=List[OrderResponse])
    async def get_orders(active_only: bool = False, limit: int = 100, api_key: str = Depends(api_key_auth)):
        """
        Get orders (from database for historical, from memory for active)

        Args:
            active_only: If True, only return active orders (pending/open)
            limit: Maximum number of orders to return (default: 100)
        """
        order_responses = []

        # Try to get orders from database (preferred for historical data)
        if hasattr(bot.storage, 'get_orders'):
            try:
                db_orders = bot.storage.get_orders(active_only=active_only, limit=limit)  # type: ignore

                for order_dict in db_orders:
                    order_responses.append(OrderResponse(
                        order_id=order_dict.get('order_id', 'N/A'),
                        symbol=order_dict.get('symbol', 'N/A'),
                        transaction_type=order_dict.get('transaction_type', 'N/A'),
                        quantity=order_dict.get('quantity', 0),
                        price=order_dict.get('price', 0.0),
                        order_type=order_dict.get('order_type', 'N/A'),
                        status=order_dict.get('status', 'N/A'),
                        timestamp=order_dict.get('timestamp', datetime.now().isoformat())
                    ))

                return order_responses
            except Exception as e:
                logger.warning(f"Failed to get orders from database: {e}")

        # Fallback: Get from memory (only active orders available)
        if active_only:
            orders = bot.order_manager.get_active_orders()
        else:
            # If database not available, only active orders are in memory
            orders = bot.order_manager.get_active_orders()

        for order in orders:
            # Convert OrderStatus enum to string if needed
            status_str = order.status.value if hasattr(order.status, 'value') else str(order.status)
            # Get timestamp - use created_at if timestamp doesn't exist
            timestamp_str = (
                order.timestamp.isoformat() if hasattr(order, 'timestamp')
                else datetime.now().isoformat()
            )

            order_responses.append(OrderResponse(
                order_id=order.order_id,
                symbol=str(order.symbol),  # Convert to string if needed
                transaction_type=order.transaction_type,
                quantity=order.quantity,
                price=order.price,
                order_type=order.order_type,
                status=status_str,
                timestamp=timestamp_str
            ))

        return order_responses[:limit]  # Apply limit

    @app.get("/pnl", response_model=PnLResponse)
    async def get_pnl(api_key: str = Depends(api_key_auth)):
        """Get P&L metrics"""
        # Get trades from storage
        trades = []
        if hasattr(bot.storage, 'get_trades'):
            trades = bot.storage.get_trades(limit=10000)  # type: ignore

        realized_pnl = 0.0
        winning_trades = 0
        losing_trades = 0
        wins_total = 0.0
        losses_total = 0.0

        for trade in trades:
            if trade.pnl is not None:
                realized_pnl += trade.pnl
                if trade.pnl > 0:
                    winning_trades += 1
                    wins_total += trade.pnl
                elif trade.pnl < 0:
                    losing_trades += 1
                    losses_total += abs(trade.pnl)

        # Calculate unrealized P&L from open positions
        unrealized_pnl = 0.0
        for symbol, pos_data in bot.order_manager.positions.items():
            avg_price = pos_data.get('average_price', 0)
            quantity = pos_data.get('quantity', 0)
            last_price = pos_data.get('last_price', avg_price)
            unrealized_pnl += (last_price - avg_price) * quantity

        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_win = (wins_total / winning_trades) if winning_trades > 0 else 0
        avg_loss = (losses_total / losing_trades) if losing_trades > 0 else 0

        return PnLResponse(
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=realized_pnl + unrealized_pnl,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            average_win=avg_win,
            average_loss=avg_loss
        )

    @app.get("/risk", response_model=RiskMetricsResponse)
    async def get_risk_metrics(api_key: str = Depends(api_key_auth)):
        """Get risk metrics and limits"""
        settings = container.settings

        # Calculate current exposure
        current_exposure = 0.0
        for symbol, pos_data in bot.order_manager.positions.items():
            avg_price = pos_data.get('average_price', 0)
            quantity = pos_data.get('quantity', 0)
            current_exposure += avg_price * quantity

        exposure_pct = (current_exposure / settings.account_balance * 100) if settings.account_balance > 0 else 0

        # Get today's P&L
        today_trades = []
        if hasattr(bot.storage, 'get_trades'):
            all_trades = bot.storage.get_trades(limit=1000)  # type: ignore
            today_trades = [t for t in all_trades if hasattr(t, 'timestamp') and t.timestamp.date() == datetime.now().date()]
        daily_pnl = sum(t.pnl or 0 for t in today_trades)

        # Count today's trades
        trades_today = len(today_trades)

        return RiskMetricsResponse(
            account_balance=settings.account_balance,
            current_exposure=current_exposure,
            exposure_percentage=exposure_pct,
            max_position_size=settings.max_position_size,
            max_risk_per_trade=settings.max_risk_per_trade,
            max_daily_loss=settings.max_daily_loss,
            daily_pnl=daily_pnl,
            open_positions=len(bot.order_manager.positions),
            max_open_positions=settings.max_open_positions,
            trades_today=trades_today,
            max_trades_per_day=settings.max_trades_per_day
        )

    @app.get("/signals", response_model=List[Dict[str, Any]])
    async def get_signals(limit: int = 50, api_key: str = Depends(api_key_auth)):
        """
        Get recent signals

        Args:
            limit: Maximum number of signals to return (default: 50)
        """
        signals = []
        if hasattr(bot.storage, 'get_signals'):
            signals = bot.storage.get_signals(limit=limit)  # type: ignore

        # Convert to dict for JSON response
        return [
            {
                "id": s.id,
                "instrument_token": s.instrument_token,
                "signal_type": s.signal_type,
                "price": s.price,
                "stop_loss": s.stop_loss,
                "take_profit_1": s.take_profit_1,
                "confidence": s.confidence,
                "timestamp": s.timestamp.isoformat()
            }
            for s in signals
        ]

    # ========================================================================
    # CONTROL ENDPOINTS
    # ========================================================================

    @app.post("/pause", response_model=MessageResponse)
    async def pause_trading(api_key: str = Depends(api_key_auth)):
        """Pause trading (stop generating new signals)"""
        if not bot.is_running:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bot is not running"
            )

        # Pause the strategy executor
        bot.strategy_executor.paused = True
        logger.info("Trading paused via API")

        return MessageResponse(
            message="Trading paused - no new signals will be generated",
            success=True
        )

    @app.post("/resume", response_model=MessageResponse)
    async def resume_trading(api_key: str = Depends(api_key_auth)):
        """Resume trading"""
        if not bot.is_running:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bot is not running"
            )

        # Resume the strategy executor
        bot.strategy_executor.paused = False
        logger.info("Trading resumed via API")

        return MessageResponse(
            message="Trading resumed - signals will be generated",
            success=True
        )

    @app.post("/stop", response_model=MessageResponse)
    async def stop_bot(api_key: str = Depends(api_key_auth)):
        """Stop the bot gracefully"""
        if not bot.is_running:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bot is already stopped"
            )

        logger.info("Stopping bot via API...")

        # Stop the bot (async)
        await bot.stop()

        return MessageResponse(
            message="Bot stopped successfully",
            success=True
        )

    @app.post("/cancel_order/{order_id}", response_model=MessageResponse)
    async def cancel_order(order_id: str, api_key: str = Depends(api_key_auth)):
        """
        Cancel a specific order

        Args:
            order_id: Order ID to cancel
        """
        try:
            await bot.order_manager.cancel_order(order_id)
            return MessageResponse(
                message=f"Order {order_id} cancelled successfully",
                success=True
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to cancel order: {str(e)}"
            )

    @app.post("/close_position/{symbol}", response_model=MessageResponse)
    async def close_position(symbol: str, api_key: str = Depends(api_key_auth)):
        """
        Close a specific position

        Args:
            symbol: Symbol to close position for
        """
        if symbol not in bot.order_manager.positions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No open position found for {symbol}"
            )

        try:
            # Get position details
            position = bot.order_manager.positions[symbol]
            quantity = position.get('quantity', 0)

            # Determine transaction type (opposite of current position)
            transaction_type = "SELL" if quantity > 0 else "BUY"

            # Place market order to close position
            order_id = await container.broker.place_order(
                tradingsymbol=symbol,
                exchange="NSE",
                transaction_type=transaction_type,
                quantity=abs(quantity),
                order_type="MARKET",
                product="MIS",
                variety="regular"
            )

            logger.info(f"Closing position for {symbol}, order ID: {order_id}")

            return MessageResponse(
                message=f"Position closed for {symbol}, order ID: {order_id}",
                success=True
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to close position: {str(e)}"
            )

    return app
