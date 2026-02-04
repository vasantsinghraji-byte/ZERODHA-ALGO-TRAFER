# -*- coding: utf-8 -*-
"""
Alerts Module - Never Miss a Trade!
===================================
Send notifications via Telegram, Email, or Desktop.

Get instant alerts for:
- Trade signals
- Order executions
- Stop loss triggers
- Price alerts
- Daily summaries
"""

import logging
import smtplib
import ssl
from collections import deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable, Tuple, Union, Deque
from enum import Enum
import threading
import queue
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal

# Import Money for precise monetary formatting
try:
    from utils.money import Money
    HAS_MONEY = True
except ImportError:
    HAS_MONEY = False

# Try importing requests for Telegram
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


def format_money(value: Union[float, int, Decimal, 'Money', None], include_sign: bool = False) -> str:
    """
    Format monetary value with precise decimal handling.

    Converts floats to Decimal to avoid floating-point display errors
    like showing 2500.5000000001 instead of 2500.50.

    Args:
        value: Monetary value (float, int, Decimal, or Money)
        include_sign: Include +/- sign for positive/negative values

    Returns:
        Formatted string like "2,500.50" or "+2,500.50"
    """
    if value is None:
        return "0.00"

    # Convert to Decimal for precise formatting
    if HAS_MONEY and hasattr(value, '_value'):
        # It's a Money object
        decimal_value = value._value
    elif isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, float):
        # Convert float to Decimal via string to avoid precision issues
        decimal_value = Decimal(str(value))
    else:
        decimal_value = Decimal(value)

    # Round to 2 decimal places
    decimal_value = decimal_value.quantize(Decimal("0.01"))

    # Format with sign if requested
    if include_sign and decimal_value > 0:
        return f"+{decimal_value:,.2f}"
    return f"{decimal_value:,.2f}"


class AlertType(Enum):
    """Types of alerts"""
    TRADE_SIGNAL = "trade_signal"
    ORDER_EXECUTED = "order_executed"
    ORDER_FAILED = "order_failed"
    STOP_LOSS = "stop_loss"
    TARGET_HIT = "target_hit"
    PRICE_ALERT = "price_alert"
    DAILY_SUMMARY = "daily_summary"
    BOT_STATUS = "bot_status"
    ERROR = "error"
    INFO = "info"


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass(order=False)
class Alert:
    """A single alert

    Comparable by priority for use with PriorityQueue.
    Higher priority (URGENT=4) comes FIRST (lower comparison value).
    """
    alert_type: AlertType
    title: str
    message: str
    priority: AlertPriority = AlertPriority.NORMAL
    symbol: Optional[str] = None
    price: Optional[float] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    _sequence: int = field(default=0, compare=False)  # For FIFO within same priority

    def __lt__(self, other: 'Alert') -> bool:
        """Compare for PriorityQueue - higher priority value = lower sort order (comes first)"""
        if self.priority.value != other.priority.value:
            # Higher priority value (URGENT=4) should come FIRST (lower comparison)
            return self.priority.value > other.priority.value
        # Same priority: FIFO by sequence number
        return self._sequence < other._sequence

    def __le__(self, other: 'Alert') -> bool:
        return self == other or self < other

    @property
    def emoji(self) -> str:
        """Get emoji for alert type"""
        emojis = {
            AlertType.TRADE_SIGNAL: "📊",
            AlertType.ORDER_EXECUTED: "✅",
            AlertType.ORDER_FAILED: "❌",
            AlertType.STOP_LOSS: "🛑",
            AlertType.TARGET_HIT: "🎯",
            AlertType.PRICE_ALERT: "🔔",
            AlertType.DAILY_SUMMARY: "📈",
            AlertType.BOT_STATUS: "🤖",
            AlertType.ERROR: "⚠️",
            AlertType.INFO: "ℹ️",
        }
        return emojis.get(self.alert_type, "📌")

    def format_telegram(self) -> str:
        """Format for Telegram (Markdown)"""
        lines = [
            f"{self.emoji} *{self.title}*",
            "",
            self.message,
        ]

        if self.symbol:
            lines.append(f"\n📌 Symbol: `{self.symbol}`")
        if self.price:
            lines.append(f"💰 Price: ₹{format_money(self.price)}")

        lines.append(f"\n🕐 {self.timestamp.strftime('%H:%M:%S')}")

        return "\n".join(lines)

    def format_email(self) -> tuple:
        """Format for Email (HTML)"""
        subject = f"[AlgoTrader] {self.emoji} {self.title}"

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #333;">{self.emoji} {self.title}</h2>
            <p style="font-size: 16px;">{self.message}</p>

            <table style="margin-top: 20px;">
        """

        if self.symbol:
            body += f'<tr><td><strong>Symbol:</strong></td><td>{self.symbol}</td></tr>'
        if self.price:
            body += f'<tr><td><strong>Price:</strong></td><td>₹{format_money(self.price)}</td></tr>'

        for key, value in self.data.items():
            body += f'<tr><td><strong>{key}:</strong></td><td>{value}</td></tr>'

        body += f"""
            </table>

            <p style="color: #666; margin-top: 30px; font-size: 12px;">
                Sent at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                <br>AlgoTrader Pro
            </p>
        </body>
        </html>
        """

        return subject, body

    def format_desktop(self) -> tuple:
        """Format for desktop notification"""
        return self.title, f"{self.emoji} {self.message}"


class TelegramNotifier:
    """
    Send notifications via Telegram.

    Setup:
    1. Create a bot with @BotFather
    2. Get your chat ID from @userinfobot
    3. Configure bot_token and chat_id

    Uses aggressive timeouts to prevent blocking the alert queue.
    """

    # Aggressive timeouts - don't let slow networks block urgent alerts
    CONNECT_TIMEOUT = 2.0   # 2 seconds to establish connection
    READ_TIMEOUT = 3.0      # 3 seconds to receive response
    URGENT_TIMEOUT = 1.0    # Even faster for URGENT alerts

    def __init__(self, bot_token: str = "", chat_id: str = ""):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
        self.api_url = f"https://api.telegram.org/bot{bot_token}"

        if not HAS_REQUESTS:
            logger.warning("requests library not installed. Telegram disabled.")
            self.enabled = False

    def send(self, alert: Alert, urgent: bool = False) -> bool:
        """Send alert via Telegram with aggressive timeouts"""
        if not self.enabled:
            logger.debug("Telegram not configured, skipping")
            return False

        try:
            message = alert.format_telegram()

            # Use faster timeout for urgent alerts
            timeout = (
                (self.URGENT_TIMEOUT, self.URGENT_TIMEOUT) if urgent
                else (self.CONNECT_TIMEOUT, self.READ_TIMEOUT)
            )

            response = requests.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "Markdown"
                },
                timeout=timeout  # (connect_timeout, read_timeout)
            )

            if response.status_code == 200:
                logger.debug(f"Telegram sent: {alert.title}")
                return True
            else:
                logger.error(f"Telegram error: {response.text}")
                return False

        except requests.exceptions.Timeout:
            logger.warning(f"Telegram timeout for: {alert.title}")
            return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def send_photo(self, photo_path: str, caption: str = "") -> bool:
        """Send a photo via Telegram with timeout"""
        if not self.enabled:
            return False

        try:
            with open(photo_path, 'rb') as photo:
                response = requests.post(
                    f"{self.api_url}/sendPhoto",
                    data={"chat_id": self.chat_id, "caption": caption},
                    files={"photo": photo},
                    timeout=(self.CONNECT_TIMEOUT, 15.0)  # Photos need more read time
                )
            return response.status_code == 200

        except requests.exceptions.Timeout:
            logger.warning(f"Telegram photo timeout")
            return False
        except Exception as e:
            logger.error(f"Telegram photo failed: {e}")
            return False

    def test_connection(self) -> bool:
        """Test Telegram connection with timeout"""
        if not self.enabled:
            return False

        try:
            response = requests.get(
                f"{self.api_url}/getMe",
                timeout=(self.CONNECT_TIMEOUT, self.READ_TIMEOUT)
            )
            if response.status_code == 200:
                bot_info = response.json()
                logger.info(f"Telegram connected: @{bot_info['result']['username']}")
                return True
            return False
        except requests.exceptions.Timeout:
            logger.warning("Telegram connection test timed out")
            return False
        except Exception as e:
            logger.error(f"Telegram test failed: {e}")
            return False


class EmailNotifier:
    """
    Send notifications via Email.

    Supports Gmail, Outlook, and custom SMTP.
    Uses aggressive timeouts to prevent blocking the alert queue.
    """

    # Aggressive timeouts for SMTP operations
    SMTP_TIMEOUT = 5.0  # 5 seconds max for entire SMTP transaction

    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: str = "",
        sender_password: str = "",
        recipient_email: str = ""
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email or sender_email
        self.enabled = bool(sender_email and sender_password)

    def send(self, alert: Alert, urgent: bool = False) -> bool:
        """Send alert via Email with timeout protection"""
        if not self.enabled:
            logger.debug("Email not configured, skipping")
            return False

        try:
            subject, html_body = alert.format_email()

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.sender_email
            msg["To"] = self.recipient_email

            # Plain text version
            plain_text = f"{alert.title}\n\n{alert.message}"
            msg.attach(MIMEText(plain_text, "plain"))

            # HTML version
            msg.attach(MIMEText(html_body, "html"))

            # Send with timeout
            timeout = 3.0 if urgent else self.SMTP_TIMEOUT
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=timeout) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(
                    self.sender_email,
                    self.recipient_email,
                    msg.as_string()
                )

            logger.debug(f"Email sent: {alert.title}")
            return True

        except (TimeoutError, smtplib.SMTPException) as e:
            logger.warning(f"Email timeout/SMTP error for: {alert.title} - {e}")
            return False
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False

    def test_connection(self) -> bool:
        """Test email connection with timeout"""
        if not self.enabled:
            return False

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=self.SMTP_TIMEOUT) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
            logger.info("Email connection successful")
            return True
        except Exception as e:
            logger.error(f"Email test failed: {e}")
            return False


class DesktopNotifier:
    """
    Show desktop notifications.

    Works on Windows, Mac, and Linux.
    """

    def __init__(self):
        self.enabled = True

        # Try to import platform-specific notifier
        try:
            from plyer import notification
            self._notify = notification.notify
        except ImportError:
            logger.warning("plyer not installed. Desktop notifications disabled.")
            self._notify = None
            self.enabled = False

    def send(self, alert: Alert) -> bool:
        """Show desktop notification"""
        if not self.enabled or not self._notify:
            return False

        try:
            title, message = alert.format_desktop()

            self._notify(
                title=title,
                message=message[:256],  # Limit message length
                app_name="AlgoTrader Pro",
                timeout=10
            )

            logger.debug(f"Desktop notification: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Desktop notification failed: {e}")
            return False


class SoundNotifier:
    """
    Play sound alerts.

    Platform compatibility:
    - Windows: Uses native winsound.MessageBeep()
    - Linux/Mac with display: Attempts system beep
    - Headless servers: Logs alert (no audio hardware)

    Note: On cloud servers (AWS, DigitalOcean, etc.), sound alerts are
    logged instead of played since there's no audio output device.
    """

    def __init__(self):
        self.enabled = True
        self._winsound = None
        self._platform = self._detect_platform()

    def _detect_platform(self) -> str:
        """
        Detect platform and audio capability.

        Returns:
            'windows' - Windows with winsound available
            'headless' - Server environment without display/audio
            'unix' - Unix-like with potential audio (Mac/Linux desktop)
        """
        import os
        import sys

        # Check for Windows
        if sys.platform == 'win32':
            try:
                import winsound
                self._winsound = winsound
                return "windows"
            except ImportError:
                pass

        # Check for headless server environment
        # No DISPLAY = likely headless Linux server
        if sys.platform.startswith('linux'):
            if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
                logger.info("SoundNotifier: Headless server detected, sound alerts will be logged only")
                return "headless"

        # Check for common CI/cloud environment variables
        ci_indicators = ['CI', 'CONTINUOUS_INTEGRATION', 'AWS_EXECUTION_ENV',
                        'KUBERNETES_SERVICE_HOST', 'DOCKER_CONTAINER']
        if any(os.environ.get(var) for var in ci_indicators):
            logger.info("SoundNotifier: Cloud/CI environment detected, sound alerts will be logged only")
            return "headless"

        # Unix-like with display (Mac, Linux desktop)
        return "unix"

    def send(self, alert: Alert) -> bool:
        """
        Play sound for alert or log it on headless servers.

        Args:
            alert: The alert to sound

        Returns:
            True if sound played or logged successfully
        """
        if not self.enabled:
            return False

        try:
            if self._platform == "windows" and self._winsound:
                # Use different Windows sounds based on alert type
                sounds = {
                    AlertType.ORDER_EXECUTED: self._winsound.MB_OK,
                    AlertType.STOP_LOSS: self._winsound.MB_ICONHAND,
                    AlertType.TARGET_HIT: self._winsound.MB_ICONASTERISK,
                    AlertType.ERROR: self._winsound.MB_ICONEXCLAMATION,
                }
                sound = sounds.get(alert.alert_type, self._winsound.MB_OK)
                self._winsound.MessageBeep(sound)
                return True

            elif self._platform == "headless":
                # On headless servers, log the alert instead of trying to beep
                # This prevents the useless print('\a') that does nothing
                logger.debug(f"Sound alert (headless): [{alert.alert_type.value}] {alert.title}")
                return True

            else:
                # Unix-like with display - attempt system beep
                # Note: May not work on all systems, but won't crash
                print('\a', end='', flush=True)
                return True

        except Exception as e:
            logger.error(f"Sound alert failed: {e}")
            return False

    @property
    def is_headless(self) -> bool:
        """Check if running in headless mode (useful for tests/debugging)."""
        return self._platform == "headless"


class AlertManager:
    """
    Central alert manager that handles all notifications.

    Features:
    - Multiple notification channels
    - PriorityQueue for urgent alerts (URGENT alerts jump to front!)
    - ThreadPoolExecutor for parallel, non-blocking sends
    - Alert history
    - Rate limiting
    - Aggressive timeouts to prevent queue blocking

    CRITICAL FIX: Uses PriorityQueue + ThreadPool to ensure URGENT alerts
    (like STOP LOSS) are never delayed behind slow LOW priority alerts.
    """

    # Thread pool configuration
    MAX_SEND_WORKERS = 4  # Parallel senders per channel type

    def __init__(self):
        self.telegram = TelegramNotifier()
        self.email = EmailNotifier()
        self.desktop = DesktopNotifier()
        self.sound = SoundNotifier()

        # CRITICAL: PriorityQueue ensures URGENT alerts jump to front!
        # Regular Queue would process FIFO, blocking urgent alerts behind slow ones
        self.alert_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._sequence_counter = 0  # For FIFO within same priority
        self._sequence_lock = threading.Lock()

        # Alert history using deque for thread-safe, bounded storage
        # deque with maxlen automatically discards old items when full (O(1))
        # This is more efficient than list slicing which creates new list objects
        self.max_history = 1000
        self._alert_history: Deque[Alert] = deque(maxlen=self.max_history)
        self._history_lock = threading.Lock()  # Still needed for iteration safety

        # Rate limiting
        self.rate_limits = {
            AlertPriority.LOW: 60,      # Max 1 per minute
            AlertPriority.NORMAL: 10,   # Max 1 per 10 seconds
            AlertPriority.HIGH: 1,      # Max 1 per second
            AlertPriority.URGENT: 0,    # No limit
        }
        self.last_alert_time: Dict[str, float] = {}
        self._rate_lock = threading.Lock()

        # Channels enabled
        self.channels_enabled = {
            'telegram': True,
            'email': True,
            'desktop': True,
            'sound': True,
        }

        # ThreadPoolExecutor for parallel, non-blocking sends
        self._executor: Optional[ThreadPoolExecutor] = None

        # Start background worker
        self._running = False
        self._worker_thread = None

    def configure_telegram(self, bot_token: str, chat_id: str):
        """Configure Telegram notifications"""
        self.telegram = TelegramNotifier(bot_token, chat_id)
        logger.info(f"Telegram configured: {self.telegram.enabled}")

    def configure_email(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        recipient_email: str = ""
    ):
        """Configure Email notifications"""
        self.email = EmailNotifier(
            smtp_server, smtp_port,
            sender_email, sender_password,
            recipient_email
        )
        logger.info(f"Email configured: {self.email.enabled}")

    def enable_channel(self, channel: str, enabled: bool = True):
        """Enable or disable a notification channel"""
        if channel in self.channels_enabled:
            self.channels_enabled[channel] = enabled
            logger.info(f"Channel {channel}: {'enabled' if enabled else 'disabled'}")

    def send(self, alert: Alert, immediate: bool = False) -> bool:
        """
        Send an alert.

        Args:
            alert: The alert to send
            immediate: If True, send synchronously (bypasses queue entirely)

        Returns:
            True if queued/sent successfully

        IMPORTANT: URGENT alerts skip rate limiting and use faster timeouts.
        PriorityQueue ensures they jump to front of queue.
        """
        # URGENT alerts bypass rate limiting entirely!
        if alert.priority != AlertPriority.URGENT:
            # Rate limiting for non-urgent
            rate_key = f"{alert.alert_type.value}_{alert.symbol or 'general'}"
            now = time.time()
            min_interval = self.rate_limits.get(alert.priority, 10)

            with self._rate_lock:
                if rate_key in self.last_alert_time:
                    elapsed = now - self.last_alert_time[rate_key]
                    if elapsed < min_interval:
                        logger.debug(f"Rate limited: {alert.title}")
                        return False
                self.last_alert_time[rate_key] = now

        # Assign sequence number for FIFO within same priority
        with self._sequence_lock:
            alert._sequence = self._sequence_counter
            self._sequence_counter += 1

        # Add to history (thread-safe)
        # deque with maxlen automatically evicts oldest when full
        with self._history_lock:
            self._alert_history.append(alert)

        if immediate or alert.priority == AlertPriority.URGENT:
            # URGENT alerts send immediately in parallel threads, don't block
            if self._executor:
                self._executor.submit(self._send_alert, alert, urgent=True)
                return True
            else:
                return self._send_alert(alert, urgent=True)
        else:
            # Queue for background processing (PriorityQueue orders by priority)
            self.alert_queue.put(alert)
            return True

    def _send_alert(self, alert: Alert, urgent: bool = False) -> bool:
        """
        Send the alert through all channels IN PARALLEL.

        Uses ThreadPoolExecutor to send to multiple channels simultaneously,
        preventing one slow channel from blocking others.

        Args:
            alert: The alert to send
            urgent: If True, use faster timeouts

        Returns:
            True if at least one channel succeeded
        """
        futures = []
        results = []

        # Use the class executor if available, otherwise send sequentially
        executor = self._executor

        def send_telegram():
            if self.channels_enabled.get('telegram', True):
                return ('telegram', self.telegram.send(alert, urgent=urgent))
            return ('telegram', False)

        def send_email():
            if self.channels_enabled.get('email', True):
                if alert.priority in [AlertPriority.HIGH, AlertPriority.URGENT]:
                    return ('email', self.email.send(alert, urgent=urgent))
            return ('email', False)

        def send_desktop():
            if self.channels_enabled.get('desktop', True):
                return ('desktop', self.desktop.send(alert))
            return ('desktop', False)

        def send_sound():
            if self.channels_enabled.get('sound', True):
                if alert.priority in [AlertPriority.HIGH, AlertPriority.URGENT]:
                    return ('sound', self.sound.send(alert))
            return ('sound', False)

        if executor:
            # PARALLEL: Submit all sends to thread pool
            futures = [
                executor.submit(send_telegram),
                executor.submit(send_email),
                executor.submit(send_desktop),
                executor.submit(send_sound),
            ]

            # Wait for all with timeout (don't block forever)
            max_wait = 2.0 if urgent else 10.0
            for future in as_completed(futures, timeout=max_wait):
                try:
                    channel, success = future.result(timeout=1.0)
                    results.append(success)
                    if success:
                        logger.debug(f"Alert sent via {channel}: {alert.title}")
                except Exception as e:
                    logger.warning(f"Channel send failed: {e}")
        else:
            # SEQUENTIAL fallback (no executor)
            for send_fn in [send_telegram, send_email, send_desktop, send_sound]:
                try:
                    channel, success = send_fn()
                    results.append(success)
                except Exception as e:
                    logger.warning(f"Send failed: {e}")

        return any(results)

    def start(self):
        """Start background alert processing with ThreadPoolExecutor"""
        if self._running:
            return

        self._running = True

        # Create thread pool for parallel sends
        self._executor = ThreadPoolExecutor(
            max_workers=self.MAX_SEND_WORKERS,
            thread_name_prefix="alert_sender"
        )

        # Start queue worker
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

        logger.info(f"Alert manager started (pool={self.MAX_SEND_WORKERS} workers)")

    def stop(self):
        """Stop background processing and cleanup"""
        self._running = False

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

        # Wait for worker thread
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            self._worker_thread = None

        logger.info("Alert manager stopped")

    def _worker(self):
        """
        Background worker that processes alert queue.

        Uses PriorityQueue so URGENT alerts jump to front automatically!
        Delegates actual sending to ThreadPoolExecutor for parallel processing.
        """
        while self._running:
            try:
                # PriorityQueue.get() returns highest priority first!
                alert = self.alert_queue.get(timeout=1)

                # Send via thread pool (non-blocking)
                if self._executor:
                    self._executor.submit(self._send_alert, alert, urgent=False)
                else:
                    self._send_alert(alert, urgent=False)

                self.alert_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Alert worker error: {e}")

    # === CONVENIENCE METHODS ===

    def trade_signal(
        self,
        symbol: str,
        signal: str,
        price: float,
        reason: str = ""
    ):
        """Send trade signal alert"""
        alert = Alert(
            alert_type=AlertType.TRADE_SIGNAL,
            title=f"{signal} Signal: {symbol}",
            message=f"{'🟢' if signal == 'BUY' else '🔴'} {signal} {symbol} at ₹{format_money(price)}\n{reason}",
            priority=AlertPriority.HIGH,
            symbol=symbol,
            price=price,
            data={"signal": signal, "reason": reason}
        )
        self.send(alert)

    def order_executed(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order_id: str = ""
    ):
        """Send order executed alert"""
        alert = Alert(
            alert_type=AlertType.ORDER_EXECUTED,
            title=f"Order Executed: {symbol}",
            message=f"{side} {quantity} {symbol} @ ₹{format_money(price)}",
            priority=AlertPriority.NORMAL,
            symbol=symbol,
            price=price,
            data={"side": side, "quantity": quantity, "order_id": order_id}
        )
        self.send(alert)

    def stop_loss_hit(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        loss: float
    ):
        """Send stop loss alert"""
        alert = Alert(
            alert_type=AlertType.STOP_LOSS,
            title=f"Stop Loss Hit: {symbol}",
            message=f"🛑 {symbol} stopped out!\nEntry: ₹{format_money(entry_price)} → Exit: ₹{format_money(exit_price)}\nLoss: ₹{format_money(loss)}",
            priority=AlertPriority.URGENT,
            symbol=symbol,
            price=exit_price,
            data={"entry": entry_price, "exit": exit_price, "loss": loss}
        )
        self.send(alert, immediate=True)

    def target_hit(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        profit: float
    ):
        """Send target hit alert"""
        alert = Alert(
            alert_type=AlertType.TARGET_HIT,
            title=f"Target Hit: {symbol}",
            message=f"🎯 {symbol} target reached!\nEntry: ₹{format_money(entry_price)} → Exit: ₹{format_money(exit_price)}\nProfit: ₹{format_money(profit)}",
            priority=AlertPriority.HIGH,
            symbol=symbol,
            price=exit_price,
            data={"entry": entry_price, "exit": exit_price, "profit": profit}
        )
        self.send(alert)

    def price_alert(self, symbol: str, price: float, condition: str):
        """Send price alert"""
        alert = Alert(
            alert_type=AlertType.PRICE_ALERT,
            title=f"Price Alert: {symbol}",
            message=f"🔔 {symbol} {condition} ₹{format_money(price)}",
            priority=AlertPriority.NORMAL,
            symbol=symbol,
            price=price,
            data={"condition": condition}
        )
        self.send(alert)

    def daily_summary(self, summary: Dict[str, Any]):
        """Send daily summary"""
        pnl = summary.get('pnl', 0)
        trades = summary.get('trades', 0)
        win_rate = summary.get('win_rate', 0)
        balance = summary.get('balance', 0)

        emoji = "📈" if pnl >= 0 else "📉"

        message = f"""
{emoji} Daily P&L: ₹{format_money(pnl, include_sign=True)}
📊 Total Trades: {trades}
🎯 Win Rate: {win_rate:.1f}%
💰 Balance: ₹{format_money(balance)}
        """.strip()

        alert = Alert(
            alert_type=AlertType.DAILY_SUMMARY,
            title="Daily Trading Summary",
            message=message,
            priority=AlertPriority.NORMAL,
            data=summary
        )
        self.send(alert)

    def error(self, title: str, message: str):
        """Send error alert"""
        alert = Alert(
            alert_type=AlertType.ERROR,
            title=f"Error: {title}",
            message=message,
            priority=AlertPriority.HIGH
        )
        self.send(alert, immediate=True)

    def info(self, title: str, message: str):
        """Send info alert"""
        alert = Alert(
            alert_type=AlertType.INFO,
            title=title,
            message=message,
            priority=AlertPriority.LOW
        )
        self.send(alert)

    def bot_status(self, status: str, details: str = ""):
        """Send bot status update"""
        emoji = "🟢" if status.lower() in ['started', 'running'] else "🔴"
        alert = Alert(
            alert_type=AlertType.BOT_STATUS,
            title=f"Bot {status}",
            message=f"{emoji} Trading bot {status.lower()}\n{details}",
            priority=AlertPriority.NORMAL,
            data={"status": status}
        )
        self.send(alert)

    def get_history(self, alert_type: AlertType = None, limit: int = 50) -> List[Alert]:
        """
        Get alert history (thread-safe).

        Uses lock to ensure consistent snapshot during iteration.
        Returns a copy to prevent external mutation.
        """
        with self._history_lock:
            # Create list copy under lock to ensure consistent snapshot
            history = list(self._alert_history)

        if alert_type:
            history = [a for a in history if a.alert_type == alert_type]

        return history[-limit:]

    @property
    def alert_history(self) -> List[Alert]:
        """
        Backward-compatible property to access alert history.

        Returns a copy to maintain thread safety.
        """
        with self._history_lock:
            return list(self._alert_history)

    def test_all_channels(self) -> Dict[str, bool]:
        """Test all notification channels"""
        results = {
            'telegram': self.telegram.test_connection(),
            'email': self.email.test_connection(),
            'desktop': self.desktop.enabled,
            'sound': self.sound.enabled,
        }
        return results


# ============== GLOBAL INSTANCE ==============

alert_manager = AlertManager()


# ============== QUICK FUNCTIONS ==============

def send_alert(title: str, message: str, priority: str = "normal"):
    """Quick function to send an alert"""
    priority_map = {
        'low': AlertPriority.LOW,
        'normal': AlertPriority.NORMAL,
        'high': AlertPriority.HIGH,
        'urgent': AlertPriority.URGENT,
    }
    alert = Alert(
        alert_type=AlertType.INFO,
        title=title,
        message=message,
        priority=priority_map.get(priority.lower(), AlertPriority.NORMAL)
    )
    alert_manager.send(alert)


def notify_trade(symbol: str, signal: str, price: float):
    """Quick trade notification"""
    alert_manager.trade_signal(symbol, signal, price)


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("ALERTS MODULE - Test")
    print("=" * 50)

    # Create manager
    manager = AlertManager()

    # Test creating alerts
    print("\n--- Creating Test Alerts ---")

    manager.trade_signal("RELIANCE", "BUY", 2500.50, "RSI oversold + MACD crossover")
    manager.order_executed("RELIANCE", "BUY", 10, 2500.50, "ORD123")
    manager.target_hit("TCS", 3400, 3570, 1700)
    manager.stop_loss_hit("INFY", 1500, 1470, -300)
    manager.price_alert("HDFC", 2800, "crossed above")

    manager.daily_summary({
        'pnl': 5250,
        'trades': 12,
        'win_rate': 66.7,
        'balance': 105250
    })

    manager.bot_status("Started", "Using TURTLE strategy")
    manager.info("Market Open", "Trading session started at 9:15 AM")
    manager.error("API Error", "Failed to fetch quotes. Retrying...")

    # Show history
    print("\n--- Alert History ---")
    for alert in manager.get_history(limit=10):
        print(f"{alert.emoji} {alert.title}")
        print(f"   {alert.message[:50]}...")
        print()

    # Test channel status
    print("\n--- Channel Status ---")
    status = manager.test_all_channels()
    for channel, enabled in status.items():
        emoji = "✅" if enabled else "❌"
        print(f"{emoji} {channel}: {'configured' if enabled else 'not configured'}")

    print("\n" + "=" * 50)
    print("Alerts module ready!")
    print("=" * 50)
    print("\nTo enable Telegram:")
    print("  manager.configure_telegram('YOUR_BOT_TOKEN', 'YOUR_CHAT_ID')")
    print("\nTo enable Email:")
    print("  manager.configure_email('smtp.gmail.com', 587, 'your@email.com', 'app_password')")
