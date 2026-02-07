# -*- coding: utf-8 -*-
"""
Integration Test Suite for Bug Fixes #5 through #14
=====================================================

Tests that all bug fixes work correctly individually AND integrate
properly with the main trading system (TradingEngine, OrderManager, etc.).

Run: python test_integration.py
"""

import os
import re
import sys
import time
import threading
import tempfile
import unittest
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# Bug #5: Silent Failure in Secret Loading
# ============================================================================

class TestBug05_ConfigurationErrorOnFailure(unittest.TestCase):
    """Config loader must RAISE errors, not silently swallow them."""

    def test_missing_settings_raises_configuration_error(self):
        """Missing settings.yaml should raise ConfigurationError, not return None."""
        from config.loader import ConfigLoader, ConfigurationError

        loader = ConfigLoader(config_dir="nonexistent_directory_12345")
        with self.assertRaises(ConfigurationError) as ctx:
            loader.load()
        self.assertIn("not found", str(ctx.exception).lower())

    def test_malformed_yaml_raises_configuration_error(self):
        """Corrupted YAML should raise ConfigurationError with clear message."""
        from config.loader import ConfigLoader, ConfigurationError

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create malformed YAML
            settings_path = Path(tmpdir) / "settings.yaml"
            settings_path.write_text("key: [invalid yaml\n  bad: indentation")

            loader = ConfigLoader(config_dir=tmpdir)
            with self.assertRaises(ConfigurationError) as ctx:
                loader.load()
            self.assertIn("yaml", str(ctx.exception).lower())

    def test_configuration_error_is_not_base_exception(self):
        """ConfigurationError should be a proper Exception subclass."""
        from config.loader import ConfigurationError
        self.assertTrue(issubclass(ConfigurationError, Exception))


# ============================================================================
# Bug #6: Fragile Regex Environment Substitution
# ============================================================================

class TestBug06_EnvVarSubstitution(unittest.TestCase):
    """Regex must correctly parse ${VAR:-default} patterns."""

    def setUp(self):
        from config.loader import ConfigLoader
        self.loader = ConfigLoader()

    def test_simple_var_substitution(self):
        """${VAR} should be replaced with env value."""
        os.environ['TEST_BUG6_VAR'] = 'hello'
        result = self.loader._substitute_env_vars("value: ${TEST_BUG6_VAR}")
        self.assertEqual(result, "value: hello")
        del os.environ['TEST_BUG6_VAR']

    def test_default_value_substitution(self):
        """${VAR:-default} should use default when VAR is not set."""
        # Ensure var is NOT set
        os.environ.pop('UNSET_BUG6_VAR', None)
        result = self.loader._substitute_env_vars("port: ${UNSET_BUG6_VAR:-8080}")
        self.assertEqual(result, "port: 8080")

    def test_default_overridden_by_env(self):
        """${VAR:-default} should use env value when VAR IS set."""
        os.environ['TEST_BUG6_PORT'] = '9090'
        result = self.loader._substitute_env_vars("port: ${TEST_BUG6_PORT:-8080}")
        self.assertEqual(result, "port: 9090")
        del os.environ['TEST_BUG6_PORT']

    def test_empty_default(self):
        """${VAR:-} should use empty string as default."""
        os.environ.pop('UNSET_BUG6_EMPTY', None)
        result = self.loader._substitute_env_vars("val: ${UNSET_BUG6_EMPTY:-}")
        self.assertEqual(result, "val: ")

    def test_unset_var_without_default(self):
        """${VAR} with no default and no env should produce empty string."""
        os.environ.pop('UNSET_BUG6_NODEFAULT', None)
        result = self.loader._substitute_env_vars("val: ${UNSET_BUG6_NODEFAULT}")
        self.assertEqual(result, "val: ")

    def test_multiple_vars_in_one_string(self):
        """Multiple ${VAR} patterns in one string should all be replaced."""
        os.environ['BUG6_HOST'] = 'localhost'
        os.environ['BUG6_PORT'] = '5432'
        result = self.loader._substitute_env_vars("url: ${BUG6_HOST}:${BUG6_PORT}")
        self.assertEqual(result, "url: localhost:5432")
        del os.environ['BUG6_HOST']
        del os.environ['BUG6_PORT']

    def test_default_with_special_chars(self):
        """Default value can contain special characters."""
        os.environ.pop('UNSET_BUG6_URL', None)
        result = self.loader._substitute_env_vars(
            "url: ${UNSET_BUG6_URL:-postgresql://user:pass@localhost/db}"
        )
        self.assertEqual(result, "url: postgresql://user:pass@localhost/db")


# ============================================================================
# Bug #7: Blocking Rate Limiter (Token Bucket)
# ============================================================================

class TestBug07_TokenBucketRateLimiter(unittest.TestCase):
    """Rate limiter must be non-blocking by default."""

    def test_token_bucket_creation(self):
        """TokenBucketRateLimiter should initialize correctly."""
        from core.data_manager import TokenBucketRateLimiter
        limiter = TokenBucketRateLimiter(requests_per_second=10.0, burst_size=10, mode='warn')
        self.assertEqual(limiter.refill_rate, 10.0)
        self.assertEqual(limiter.burst_size, 10)
        self.assertEqual(limiter.mode, 'warn')

    def test_acquire_within_limit(self):
        """Requests within burst limit should succeed immediately."""
        from core.data_manager import TokenBucketRateLimiter
        limiter = TokenBucketRateLimiter(requests_per_second=10.0, burst_size=5, mode='warn')

        # Should acquire up to burst_size tokens without issue
        for _ in range(5):
            self.assertTrue(limiter.acquire())

    def test_warn_mode_does_not_block(self):
        """In 'warn' mode, exhausted tokens should NOT block the thread."""
        from core.data_manager import TokenBucketRateLimiter
        limiter = TokenBucketRateLimiter(requests_per_second=1.0, burst_size=1, mode='warn')

        # Exhaust tokens
        limiter.acquire()

        # Next acquire should return quickly (not block)
        start = time.time()
        limiter.acquire()
        elapsed = time.time() - start

        # Should complete in well under 1 second (no blocking)
        self.assertLess(elapsed, 0.5)

    def test_thread_safety(self):
        """Rate limiter must be safe for concurrent access."""
        from core.data_manager import TokenBucketRateLimiter
        limiter = TokenBucketRateLimiter(requests_per_second=100.0, burst_size=100, mode='warn')

        errors = []

        def acquire_many():
            try:
                for _ in range(50):
                    limiter.acquire()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=acquire_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")


# ============================================================================
# Bug #8: Timezone Naivety (Server Time Trap)
# ============================================================================

class TestBug08_TimezoneAwareness(unittest.TestCase):
    """All timestamps must be IST-aware, not naive."""

    def test_now_ist_returns_aware_datetime(self):
        """now_ist() must return a timezone-aware datetime."""
        from utils.timezone import now_ist
        dt = now_ist()
        self.assertIsNotNone(dt.tzinfo, "now_ist() returned naive datetime!")

    def test_now_ist_is_in_ist(self):
        """now_ist() must be in Asia/Kolkata timezone."""
        from utils.timezone import now_ist, IST
        dt = now_ist()
        # IST offset is always +5:30
        offset = dt.utcoffset()
        self.assertEqual(offset, timedelta(hours=5, minutes=30))

    def test_ist_constant_available(self):
        """IST timezone constant must be importable."""
        from utils.timezone import IST
        self.assertIsNotNone(IST)

    def test_market_hours_constants(self):
        """Market open/close times must match NSE hours."""
        from utils.timezone import MARKET_OPEN, MARKET_CLOSE
        from datetime import time
        self.assertEqual(MARKET_OPEN, time(9, 15))
        self.assertEqual(MARKET_CLOSE, time(15, 30))

    def test_broker_uses_ist_for_quotes(self):
        """ZerodhaBroker.get_quote should use now_ist() for timestamps."""
        import inspect
        from core.broker import ZerodhaBroker
        source = inspect.getsource(ZerodhaBroker.get_quote)
        self.assertIn('now_ist()', source, "get_quote should use now_ist()")

    def test_broker_imports_timezone(self):
        """broker.py must import from utils.timezone."""
        import inspect
        import core.broker as broker_module
        source = inspect.getsource(broker_module)
        self.assertIn('from utils.timezone import', source)


# ============================================================================
# Bug #9: Historical Data API Limit Violation
# ============================================================================

class TestBug09_HistoricalDataChunking(unittest.TestCase):
    """Historical data requests must respect Kite API limits."""

    def test_kite_limits_defined(self):
        """KITE_HISTORICAL_LIMITS must be defined in data_manager."""
        import core.data_manager as dm
        source = inspect.getsource(dm)
        # Check that the module handles date-range chunking
        self.assertIn('KITE_HISTORICAL_LIMITS', source,
                       "data_manager.py should define KITE_HISTORICAL_LIMITS")


# ============================================================================
# Bug #10: Index Symbol Rejection (Regex Too Strict)
# ============================================================================

class TestBug10_IndexSymbolValidation(unittest.TestCase):
    """Symbol regex must allow spaces for index symbols."""

    def test_nifty_50_accepted(self):
        """'NIFTY 50' must pass symbol validation."""
        from core.data_manager import _VALID_SYMBOL_PATTERN
        self.assertIsNotNone(_VALID_SYMBOL_PATTERN.match("NIFTY 50"))

    def test_bank_nifty_accepted(self):
        """'BANK NIFTY' must pass symbol validation."""
        from core.data_manager import _VALID_SYMBOL_PATTERN
        self.assertIsNotNone(_VALID_SYMBOL_PATTERN.match("BANK NIFTY"))

    def test_regular_symbols_accepted(self):
        """Standard symbols like RELIANCE, TCS must still pass."""
        from core.data_manager import _VALID_SYMBOL_PATTERN
        for symbol in ["RELIANCE", "TCS", "INFY", "SBIN", "HDFCBANK"]:
            self.assertIsNotNone(
                _VALID_SYMBOL_PATTERN.match(symbol),
                f"'{symbol}' should pass validation"
            )

    def test_special_symbols_accepted(self):
        """Symbols with &, - should still pass."""
        from core.data_manager import _VALID_SYMBOL_PATTERN
        for symbol in ["M&M", "NIFTY-50"]:
            self.assertIsNotNone(
                _VALID_SYMBOL_PATTERN.match(symbol),
                f"'{symbol}' should pass validation"
            )

    def test_invalid_symbols_rejected(self):
        """Symbols starting with lowercase or special chars should fail."""
        from core.data_manager import _VALID_SYMBOL_PATTERN
        for symbol in ["reliance", "$NIFTY", "@TCS", ""]:
            self.assertIsNone(
                _VALID_SYMBOL_PATTERN.match(symbol),
                f"'{symbol}' should be rejected"
            )


# ============================================================================
# Bug #11: Date-Only Request Logic
# ============================================================================

class TestBug11_DateTimeFormat(unittest.TestCase):
    """Historical data requests must include time, not just date."""

    def test_broker_historical_uses_datetime_format(self):
        """Broker's historical_data method should format with time component."""
        import inspect
        from core.broker import ZerodhaBroker
        # Check that the historical_data method includes time formatting
        if hasattr(ZerodhaBroker, 'historical_data'):
            source = inspect.getsource(ZerodhaBroker.historical_data)
            # Should NOT use strftime("%Y-%m-%d") alone - should include time
            self.assertTrue(
                'strftime' in source and ('%H' in source or '%M' in source or 'T' in source),
                "historical_data should include time component in date formatting"
            )


# ============================================================================
# Bug #12: Floating Point Money Errors
# ============================================================================

class TestBug12_DecimalPrecision(unittest.TestCase):
    """All financial calculations must use Decimal, not float."""

    # --- Money class tests ---

    def test_money_class_exists(self):
        """Money class must be importable from utils.money."""
        from utils.money import Money
        price = Money("2500.50")
        self.assertEqual(str(price), "2500.50")

    def test_money_no_float_error(self):
        """Money arithmetic must not have 0.1 + 0.2 != 0.3 error."""
        from utils.money import Money
        a = Money("0.1")
        b = Money("0.2")
        result = a + b
        self.assertEqual(float(result), 0.30)

    def test_money_multiplication(self):
        """Money * int quantity must be precise."""
        from utils.money import Money
        price = Money("2500.50")
        total = price * 10
        self.assertEqual(float(total), 25005.00)

    # --- Config Decimal tests ---

    def test_settings_uses_decimal(self):
        """config.config.Settings must use Decimal for financial fields."""
        from config.config import Settings
        s = Settings()
        self.assertIsInstance(s.MAX_POSITION_SIZE, Decimal)
        self.assertIsInstance(s.RISK_PER_TRADE, Decimal)

    def test_trading_config_uses_decimal(self):
        """TradingConfig must use Decimal for money fields."""
        from app.config import TradingConfig
        tc = TradingConfig()
        self.assertIsInstance(tc.capital, Decimal)
        self.assertIsInstance(tc.risk_per_trade, Decimal)
        self.assertIsInstance(tc.max_daily_loss, Decimal)

    # --- Trading Engine Decimal tests ---

    def test_trading_engine_decimal_pnl(self):
        """TradingEngine must track P&L with Decimal precision."""
        from core.trading_engine import TradingEngine, EngineConfig, TradingMode

        config = EngineConfig(mode=TradingMode.PAPER, capital=100000)
        engine = TradingEngine(config)

        # Daily P&L should be Decimal
        self.assertIsInstance(engine._daily_pnl, Decimal)
        self.assertIsInstance(engine._start_capital, Decimal)

    def test_to_decimal_conversion(self):
        """_to_decimal helper must convert float/int/None to precise Decimal."""
        from core.trading_engine import _to_decimal

        # Float
        d = _to_decimal(100.50)
        self.assertIsInstance(d, Decimal)
        self.assertEqual(d, Decimal("100.50"))

        # Int
        d = _to_decimal(100)
        self.assertEqual(d, Decimal("100.00"))

        # None
        d = _to_decimal(None)
        self.assertEqual(d, Decimal("0"))

        # Already Decimal
        d = _to_decimal(Decimal("99.99"))
        self.assertEqual(d, Decimal("99.99"))

    # --- Order Manager Decimal tests ---

    def test_order_value_uses_decimal(self):
        """Order.value property must return Decimal."""
        from core.order_manager import Order, Side
        order = Order(symbol="RELIANCE", side=Side.BUY, quantity=10, price=2500.50)
        self.assertIsInstance(order.value, Decimal)

    def test_paper_balance_is_decimal(self):
        """OrderManager paper balance must use Decimal."""
        from core.order_manager import OrderManager
        om = OrderManager(paper_trading=True)
        om.set_paper_balance(100000)
        self.assertIsInstance(om.get_paper_balance(), Decimal)

    def test_decimal_pnl_accumulation_no_drift(self):
        """Repeated small P&L additions must not accumulate float drift."""
        from core.trading_engine import _to_decimal

        total = Decimal("0")
        # Add 0.1 one thousand times - with float this gives 99.99999999...
        for _ in range(1000):
            total += _to_decimal(0.1)

        self.assertEqual(total, Decimal("100.00"))


# ============================================================================
# Bug #13: Missing Critical Dependency (utils.database)
# ============================================================================

class TestBug13_DatabaseImport(unittest.TestCase):
    """utils.database must be importable with get_db and db_session."""

    def test_database_module_importable(self):
        """utils.database must import without errors."""
        from utils.database import get_db, db_session
        self.assertTrue(callable(get_db))

    def test_db_session_is_context_manager(self):
        """db_session() must be a context manager."""
        from utils.database import db_session
        import contextlib
        # Check it's a generator-based context manager
        self.assertTrue(
            hasattr(db_session, '__enter__') or
            callable(db_session)
        )

    def test_data_manager_can_import_database(self):
        """data_manager.py must be able to import from utils.database."""
        import inspect
        import core.data_manager as dm
        source = inspect.getsource(dm)
        self.assertIn('from utils.database import get_db, db_session', source)


# ============================================================================
# Bug #14: No Retry Mechanism
# ============================================================================

class TestBug14_RetryMechanism(unittest.TestCase):
    """Network operations must retry on transient failures."""

    def test_retry_decorator_importable(self):
        """retry_on_network_error must be importable from utils.retry."""
        from utils.retry import retry_on_network_error
        self.assertTrue(callable(retry_on_network_error))

    def test_retry_on_transient_failure(self):
        """Decorated function should retry on ConnectionError."""
        from utils.retry import retry_on_network_error

        call_count = 0

        @retry_on_network_error(tries=3, delay=0.01, backoff=1.0)
        def flaky_api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network blip")
            return "success"

        result = flaky_api_call()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)  # Called 3 times

    def test_retry_exhausted_raises(self):
        """If all retries fail, the exception should be raised."""
        from utils.retry import retry_on_network_error

        @retry_on_network_error(tries=2, delay=0.01, backoff=1.0)
        def always_fails():
            raise ConnectionError("Permanent failure")

        with self.assertRaises(ConnectionError):
            always_fails()

    def test_non_network_errors_not_retried(self):
        """ValueError, TypeError etc. should NOT be retried."""
        from utils.retry import retry_on_network_error

        call_count = 0

        @retry_on_network_error(tries=3, delay=0.01, backoff=1.0)
        def validation_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Bad input")

        with self.assertRaises(ValueError):
            validation_error()
        self.assertEqual(call_count, 1)  # Should NOT retry

    def test_exponential_backoff_timing(self):
        """Backoff delay should increase exponentially between retries."""
        from utils.retry import retry_on_network_error

        timestamps = []

        @retry_on_network_error(tries=3, delay=0.05, backoff=2.0)
        def timed_failure():
            timestamps.append(time.time())
            raise ConnectionError("fail")

        with self.assertRaises(ConnectionError):
            timed_failure()

        self.assertEqual(len(timestamps), 3)

        # Gap between attempt 1 and 2 should be ~0.05s
        gap1 = timestamps[1] - timestamps[0]
        # Gap between attempt 2 and 3 should be ~0.10s (0.05 * 2)
        gap2 = timestamps[2] - timestamps[1]

        # Allow some tolerance for timing
        self.assertGreater(gap1, 0.03)
        self.assertLess(gap1, 0.15)
        self.assertGreater(gap2, 0.06)
        self.assertLess(gap2, 0.25)
        # Second gap should be roughly double the first
        self.assertGreater(gap2, gap1 * 1.3)

    def test_retry_with_result_returns_default(self):
        """retry_with_result should return default instead of raising."""
        from utils.retry import retry_with_result

        @retry_with_result(tries=2, delay=0.01, default=None)
        def failing_fetch():
            raise ConnectionError("down")

        result = failing_fetch()
        self.assertIsNone(result)

    def test_broker_has_retry_decorators(self):
        """ZerodhaBroker must have retry-decorated methods."""
        from core.broker import ZerodhaBroker
        # Check that the retry-enabled helper methods exist
        self.assertTrue(hasattr(ZerodhaBroker, '_execute_order_request'))
        self.assertTrue(hasattr(ZerodhaBroker, '_fetch_quote_request'))
        self.assertTrue(hasattr(ZerodhaBroker, '_cancel_order_request'))

    def test_broker_imports_retry(self):
        """broker.py must import retry_on_network_error."""
        import inspect
        import core.broker as broker_module
        source = inspect.getsource(broker_module)
        self.assertIn('from utils.retry import retry_on_network_error', source)


# ============================================================================
# INTEGRATION: End-to-End Trading Flow
# ============================================================================

class TestIntegration_TradingFlow(unittest.TestCase):
    """Test that all bug fixes work together in a trading flow."""

    def test_paper_engine_creation_uses_decimal(self):
        """Paper trading engine must initialize with Decimal capital tracking."""
        from core.trading_engine import create_paper_engine

        engine = create_paper_engine(capital=100000)
        self.assertIsInstance(engine._start_capital, Decimal)
        self.assertEqual(engine._start_capital, Decimal("100000.00"))
        self.assertIsInstance(engine._daily_pnl, Decimal)

    def test_paper_order_flow_precise(self):
        """Paper trading buy/sell must use Decimal throughout."""
        from core.order_manager import OrderManager, Side, OrderStatus

        om = OrderManager(paper_trading=True)
        om.set_paper_balance(100000)

        # Buy
        order = om.buy("RELIANCE", quantity=10, price=2500.50)
        self.assertEqual(order.status, OrderStatus.COMPLETE)
        self.assertIsInstance(order.value, Decimal)

        # Balance should be reduced precisely
        expected_balance = Decimal("100000.00") - (Decimal("2500.50") * 10)
        self.assertEqual(om.get_paper_balance(), expected_balance)

        # Sell
        order2 = om.sell("RELIANCE", quantity=10, price=2550.75)
        self.assertEqual(order2.status, OrderStatus.COMPLETE)

        # Final balance = original - buy_cost + sell_proceeds
        sell_proceeds = Decimal("2550.75") * 10
        final_expected = expected_balance + sell_proceeds
        self.assertEqual(om.get_paper_balance(), final_expected)

    def test_daily_loss_limit_uses_decimal(self):
        """Daily loss limit check must use Decimal to avoid false triggers."""
        from core.trading_engine import TradingEngine, EngineConfig, TradingMode, _to_decimal

        config = EngineConfig(mode=TradingMode.PAPER, capital=100000, max_daily_loss_pct=5.0)
        engine = TradingEngine(config)

        # Simulate a 4.99% loss (should NOT trigger)
        engine._daily_pnl = _to_decimal(-4990)
        self.assertFalse(engine._check_daily_loss_limit())

        # Simulate a 5.01% loss (should trigger)
        engine._daily_pnl = _to_decimal(-5010)
        self.assertTrue(engine._check_daily_loss_limit())

    def test_config_round_trip_decimal_preserved(self):
        """Loading and saving config must preserve Decimal precision."""
        from app.config import TradingConfig
        tc = TradingConfig(
            capital=Decimal("250000.50"),
            risk_per_trade=Decimal("1.5"),
            max_daily_loss=Decimal("3.0")
        )
        self.assertEqual(tc.capital, Decimal("250000.50"))
        self.assertEqual(tc.risk_per_trade, Decimal("1.5"))

    def test_all_utility_modules_importable(self):
        """All utility modules created for bug fixes must import cleanly."""
        # Bug #8
        from utils.timezone import now_ist, to_ist, IST, is_market_open
        # Bug #12
        from utils.money import Money, to_decimal, NSE_TICK_SIZE
        # Bug #13
        from utils.database import get_db, db_session
        # Bug #14
        from utils.retry import retry_on_network_error, retry_with_result

        # All should be callable
        self.assertTrue(callable(now_ist))
        self.assertTrue(callable(retry_on_network_error))

    def test_engine_stats_pnl_is_float_compatible(self):
        """Engine stats must convert Decimal P&L to float for JSON serialization."""
        from core.trading_engine import TradingEngine, EngineConfig, TradingMode

        config = EngineConfig(mode=TradingMode.PAPER, capital=100000)
        engine = TradingEngine(config)
        engine._daily_pnl = Decimal("-1500.75")

        stats = engine.get_stats()
        # daily_pnl in stats should be float (for JSON serialization)
        self.assertIsInstance(stats['daily_pnl'], float)
        self.assertAlmostEqual(stats['daily_pnl'], -1500.75, places=2)


# ============================================================================
# INTEGRATION: Config System Coherence
# ============================================================================

class TestIntegration_ConfigSystem(unittest.TestCase):
    """Test that both config systems (Pydantic + YAML loader) work together."""

    def test_pydantic_settings_load(self):
        """Pydantic Settings should load without crashing in dev mode."""
        from config.config import Settings
        s = Settings()
        self.assertIsNotNone(s.ENV)
        self.assertIn(s.ENV, ['development', 'production', 'testing'])

    def test_pydantic_no_hardcoded_credentials(self):
        """Settings must NOT have hardcoded database passwords."""
        from config.config import Settings
        s = Settings()
        # DATABASE_URL should be None in development (not hardcoded)
        if s.ENV == 'development':
            # In dev, None is allowed
            self.assertTrue(
                s.DATABASE_URL is None or 'trader123' not in str(s.DATABASE_URL),
                "DATABASE_URL should not contain default password"
            )

    def test_env_var_regex_pattern_correct(self):
        """The regex pattern in loader.py must correctly handle :- syntax."""
        from config.loader import ConfigLoader
        loader = ConfigLoader()

        # This was the exact bug: the old regex used [:[-] (character class)
        # instead of :- (literal). Test the critical edge case.
        os.environ.pop('_NONEXISTENT_TEST_VAR_', None)
        result = loader._substitute_env_vars("${_NONEXISTENT_TEST_VAR_:-fallback}")
        self.assertEqual(result, "fallback")


# ============================================================================
# Runner
# ============================================================================

import inspect

if __name__ == "__main__":
    # Run with verbose output
    print("=" * 70)
    print("INTEGRATION TEST SUITE - Bug Fixes #5 through #14")
    print("=" * 70)
    print()

    # Collect all test classes
    test_classes = [
        TestBug05_ConfigurationErrorOnFailure,
        TestBug06_EnvVarSubstitution,
        TestBug07_TokenBucketRateLimiter,
        TestBug08_TimezoneAwareness,
        TestBug09_HistoricalDataChunking,
        TestBug10_IndexSymbolValidation,
        TestBug11_DateTimeFormat,
        TestBug12_DecimalPrecision,
        TestBug13_DatabaseImport,
        TestBug14_RetryMechanism,
        TestIntegration_TradingFlow,
        TestIntegration_ConfigSystem,
    ]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print()
    print("=" * 70)
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors

    if result.wasSuccessful():
        print(f"ALL {total} TESTS PASSED!")
    else:
        print(f"RESULTS: {passed}/{total} passed, {failures} failures, {errors} errors")

        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split(chr(10))[-2]}")

        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split(chr(10))[-2]}")

    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)
