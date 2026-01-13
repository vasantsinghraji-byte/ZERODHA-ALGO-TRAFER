# -*- coding: utf-8 -*-
"""
Application Container - Dependency Injection Container
Central place to create and access all application services
"""
from typing import Optional
import logging
from kiteconnect import KiteConnect

from zerodha_trader.core.config import Settings
from zerodha_trader.core.services import (
    IMarketDataService, IBrokerService, IDataStorageService, INotificationService,
    ZerodhaMarketDataService, ZerodhaBrokerService,
    SQLiteDataStorageService, ConsoleNotificationService, TelegramNotificationService
)

logger = logging.getLogger(__name__)


class AppContainer:
    """
    Application Dependency Injection Container

    This container holds all the core services your application needs.
    It's created once at startup and provides a single source of truth
    for accessing dependencies.

    Benefits:
    - Centralized service creation and configuration
    - Easy to swap implementations (e.g., mock services for testing)
    - Clear visibility of all application dependencies
    - Loose coupling between components
    """

    def __init__(self, settings: Settings):
        """
        Initialize the application container

        Args:
            settings: Application configuration
        """
        self.settings = settings
        self._kite: Optional[KiteConnect] = None
        self._market_data: Optional[IMarketDataService] = None
        self._broker: Optional[IBrokerService] = None
        self._storage: Optional[IDataStorageService] = None
        self._notifier: Optional[INotificationService] = None

        logger.info(f"AppContainer initialized for environment: {settings.environment.value}")

    # =========================================================================
    # CORE SERVICES (Lazy initialization)
    # =========================================================================

    @property
    def kite(self) -> KiteConnect:
        """
        Get KiteConnect client (singleton)

        Returns:
            Configured KiteConnect instance
        """
        if self._kite is None:
            self._kite = KiteConnect(api_key=self.settings.zerodha_api_key)
            self._kite.set_access_token(self.settings.zerodha_access_token)
            logger.info("KiteConnect client created")

        return self._kite

    @property
    def market_data(self) -> IMarketDataService:
        """
        Get market data service (singleton)

        Returns:
            Market data service instance
        """
        if self._market_data is None:
            self._market_data = ZerodhaMarketDataService(
                api_key=self.settings.zerodha_api_key,
                access_token=self.settings.zerodha_access_token
            )
            logger.info("Market data service created")

        return self._market_data

    @property
    def broker(self) -> IBrokerService:
        """
        Get broker service (singleton)

        Returns:
            Broker service instance
        """
        if self._broker is None:
            self._broker = ZerodhaBrokerService(kite=self.kite)
            logger.info("Broker service created")

        return self._broker

    @property
    def storage(self) -> IDataStorageService:
        """
        Get data storage service (singleton)

        Returns:
            Data storage service instance
        """
        if self._storage is None:
            self._storage = SQLiteDataStorageService(db_path=self.settings.db_path)
            logger.info(f"Data storage service created: {self.settings.db_path}")

        return self._storage

    @property
    def notifier(self) -> INotificationService:
        """
        Get notification service (singleton)

        The notification service chosen depends on configuration:
        - Telegram if credentials provided
        - Console otherwise (for development)

        Returns:
            Notification service instance
        """
        if self._notifier is None:
            # Check if Telegram credentials are available
            telegram_token = getattr(self.settings, 'telegram_bot_token', None)
            telegram_chat_id = getattr(self.settings, 'telegram_chat_id', None)

            if telegram_token and telegram_chat_id:
                self._notifier = TelegramNotificationService(
                    bot_token=telegram_token,
                    chat_id=telegram_chat_id
                )
                logger.info("Telegram notification service created")
            else:
                self._notifier = ConsoleNotificationService()
                logger.info("Console notification service created (Telegram not configured)")

        return self._notifier

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def shutdown(self) -> None:
        """
        Gracefully shutdown all services

        This method should be called when the application is stopping
        to ensure all connections are properly closed.
        """
        logger.info("Shutting down AppContainer...")

        # Disconnect market data
        if self._market_data:
            try:
                self._market_data.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting market data: {e}")

        # Close other resources if needed
        # ...

        logger.info("AppContainer shutdown complete")

    def health_check(self) -> dict:
        """
        Perform health check on all services

        Returns:
            Dictionary with health status of each service
        """
        health = {
            'kite': False,
            'market_data': False,
            'broker': False,
            'storage': False,
            'notifier': False
        }

        # Check Kite connection
        try:
            if self._kite:
                self._kite.profile()
                health['kite'] = True
        except Exception as e:
            logger.warning(f"Kite health check failed: {e}")

        # Check market data
        if self._market_data:
            health['market_data'] = True  # Connected if instantiated

        # Check broker
        try:
            if self._broker:
                self._broker.get_positions()
                health['broker'] = True
        except Exception as e:
            logger.warning(f"Broker health check failed: {e}")

        # Check storage
        try:
            if self._storage:
                self._storage.get_trade_history(limit=1)
                health['storage'] = True
        except Exception as e:
            logger.warning(f"Storage health check failed: {e}")

        # Check notifier
        if self._notifier:
            health['notifier'] = True

        return health

    # =========================================================================
    # FACTORY METHODS (for testing/customization)
    # =========================================================================

    def override_market_data(self, service: IMarketDataService) -> None:
        """Override market data service (useful for testing)"""
        self._market_data = service
        logger.info("Market data service overridden")

    def override_broker(self, service: IBrokerService) -> None:
        """Override broker service (useful for testing)"""
        self._broker = service
        logger.info("Broker service overridden")

    def override_storage(self, service: IDataStorageService) -> None:
        """Override storage service (useful for testing)"""
        self._storage = service
        logger.info("Storage service overridden")

    def override_notifier(self, service: INotificationService) -> None:
        """Override notification service"""
        self._notifier = service
        logger.info("Notification service overridden")

    def __repr__(self) -> str:
        return f"<AppContainer environment={self.settings.environment.value}>"


# =============================================================================
# SINGLETON PATTERN (optional)
# =============================================================================

_container: Optional[AppContainer] = None


def get_container(settings: Optional[Settings] = None, force_reload: bool = False) -> AppContainer:
    """
    Get global AppContainer instance (singleton pattern)

    Args:
        settings: Settings to use (if creating new container)
        force_reload: Force recreation of container

    Returns:
        AppContainer instance
    """
    global _container

    if _container is None or force_reload:
        if settings is None:
            from zerodha_trader.core.config import get_settings
            settings = get_settings()

        _container = AppContainer(settings)
        logger.info("Global AppContainer created")

    return _container


def reset_container() -> None:
    """Reset global container (useful for testing)"""
    global _container
    if _container:
        _container.shutdown()
    _container = None
    logger.info("Global AppContainer reset")
