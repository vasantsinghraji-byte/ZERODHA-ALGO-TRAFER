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
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import threading
import queue
import json
import time

# Try importing requests for Telegram
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


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


@dataclass
class Alert:
    """A single alert"""
    alert_type: AlertType
    title: str
    message: str
    priority: AlertPriority = AlertPriority.NORMAL
    symbol: Optional[str] = None
    price: Optional[float] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

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
            lines.append(f"💰 Price: ₹{self.price:,.2f}")

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
            body += f'<tr><td><strong>Price:</strong></td><td>₹{self.price:,.2f}</td></tr>'

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
    """

    def __init__(self, bot_token: str = "", chat_id: str = ""):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
        self.api_url = f"https://api.telegram.org/bot{bot_token}"

        if not HAS_REQUESTS:
            logger.warning("requests library not installed. Telegram disabled.")
            self.enabled = False

    def send(self, alert: Alert) -> bool:
        """Send alert via Telegram"""
        if not self.enabled:
            logger.debug("Telegram not configured, skipping")
            return False

        try:
            message = alert.format_telegram()

            response = requests.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "Markdown"
                },
                timeout=10
            )

            if response.status_code == 200:
                logger.debug(f"Telegram sent: {alert.title}")
                return True
            else:
                logger.error(f"Telegram error: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def send_photo(self, photo_path: str, caption: str = "") -> bool:
        """Send a photo via Telegram"""
        if not self.enabled:
            return False

        try:
            with open(photo_path, 'rb') as photo:
                response = requests.post(
                    f"{self.api_url}/sendPhoto",
                    data={"chat_id": self.chat_id, "caption": caption},
                    files={"photo": photo},
                    timeout=30
                )
            return response.status_code == 200

        except Exception as e:
            logger.error(f"Telegram photo failed: {e}")
            return False

    def test_connection(self) -> bool:
        """Test Telegram connection"""
        if not self.enabled:
            return False

        try:
            response = requests.get(f"{self.api_url}/getMe", timeout=5)
            if response.status_code == 200:
                bot_info = response.json()
                logger.info(f"Telegram connected: @{bot_info['result']['username']}")
                return True
            return False
        except Exception as e:
            logger.error(f"Telegram test failed: {e}")
            return False


class EmailNotifier:
    """
    Send notifications via Email.

    Supports Gmail, Outlook, and custom SMTP.
    """

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

    def send(self, alert: Alert) -> bool:
        """Send alert via Email"""
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

            # Send
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(
                    self.sender_email,
                    self.recipient_email,
                    msg.as_string()
                )

            logger.debug(f"Email sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False

    def test_connection(self) -> bool:
        """Test email connection"""
        if not self.enabled:
            return False

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
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

    Different sounds for different alert types.
    """

    def __init__(self):
        self.enabled = True

        try:
            import winsound
            self._winsound = winsound
            self._platform = "windows"
        except ImportError:
            self._winsound = None
            self._platform = "other"

    def send(self, alert: Alert) -> bool:
        """Play sound for alert"""
        if not self.enabled:
            return False

        try:
            if self._platform == "windows" and self._winsound:
                # Use different Windows sounds
                sounds = {
                    AlertType.ORDER_EXECUTED: self._winsound.MB_OK,
                    AlertType.STOP_LOSS: self._winsound.MB_ICONHAND,
                    AlertType.TARGET_HIT: self._winsound.MB_ICONASTERISK,
                    AlertType.ERROR: self._winsound.MB_ICONEXCLAMATION,
                }
                sound = sounds.get(alert.alert_type, self._winsound.MB_OK)
                self._winsound.MessageBeep(sound)
                return True
            else:
                # For other platforms, just print a beep character
                print('\a', end='', flush=True)
                return True

        except Exception as e:
            logger.error(f"Sound alert failed: {e}")
            return False


class AlertManager:
    """
    Central alert manager that handles all notifications.

    Features:
    - Multiple notification channels
    - Alert queue with async processing
    - Alert history
    - Rate limiting
    - Priority handling
    """

    def __init__(self):
        self.telegram = TelegramNotifier()
        self.email = EmailNotifier()
        self.desktop = DesktopNotifier()
        self.sound = SoundNotifier()

        self.alert_queue = queue.Queue()
        self.alert_history: List[Alert] = []
        self.max_history = 1000

        # Rate limiting
        self.rate_limits = {
            AlertPriority.LOW: 60,      # Max 1 per minute
            AlertPriority.NORMAL: 10,   # Max 1 per 10 seconds
            AlertPriority.HIGH: 1,      # Max 1 per second
            AlertPriority.URGENT: 0,    # No limit
        }
        self.last_alert_time: Dict[str, float] = {}

        # Channels enabled
        self.channels_enabled = {
            'telegram': True,
            'email': True,
            'desktop': True,
            'sound': True,
        }

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
            immediate: If True, send synchronously

        Returns:
            True if queued/sent successfully
        """
        # Rate limiting
        rate_key = f"{alert.alert_type.value}_{alert.symbol or 'general'}"
        now = time.time()
        min_interval = self.rate_limits.get(alert.priority, 10)

        if rate_key in self.last_alert_time:
            elapsed = now - self.last_alert_time[rate_key]
            if elapsed < min_interval:
                logger.debug(f"Rate limited: {alert.title}")
                return False

        self.last_alert_time[rate_key] = now

        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

        if immediate:
            return self._send_alert(alert)
        else:
            self.alert_queue.put(alert)
            return True

    def _send_alert(self, alert: Alert) -> bool:
        """Actually send the alert through all channels"""
        success = False

        # Telegram
        if self.channels_enabled.get('telegram', True):
            if self.telegram.send(alert):
                success = True

        # Email (only for high priority)
        if self.channels_enabled.get('email', True):
            if alert.priority in [AlertPriority.HIGH, AlertPriority.URGENT]:
                if self.email.send(alert):
                    success = True

        # Desktop
        if self.channels_enabled.get('desktop', True):
            if self.desktop.send(alert):
                success = True

        # Sound
        if self.channels_enabled.get('sound', True):
            if alert.priority in [AlertPriority.HIGH, AlertPriority.URGENT]:
                self.sound.send(alert)

        return success

    def start(self):
        """Start background alert processing"""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        logger.info("Alert manager started")

    def stop(self):
        """Stop background processing"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("Alert manager stopped")

    def _worker(self):
        """Background worker that processes alert queue"""
        while self._running:
            try:
                alert = self.alert_queue.get(timeout=1)
                self._send_alert(alert)
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
            message=f"{'🟢' if signal == 'BUY' else '🔴'} {signal} {symbol} at ₹{price:,.2f}\n{reason}",
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
            message=f"{side} {quantity} {symbol} @ ₹{price:,.2f}",
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
            message=f"🛑 {symbol} stopped out!\nEntry: ₹{entry_price:,.2f} → Exit: ₹{exit_price:,.2f}\nLoss: ₹{loss:,.2f}",
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
            message=f"🎯 {symbol} target reached!\nEntry: ₹{entry_price:,.2f} → Exit: ₹{exit_price:,.2f}\nProfit: ₹{profit:,.2f}",
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
            message=f"🔔 {symbol} {condition} ₹{price:,.2f}",
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

        emoji = "📈" if pnl >= 0 else "📉"
        sign = "+" if pnl >= 0 else ""

        message = f"""
{emoji} Daily P&L: {sign}₹{pnl:,.2f}
📊 Total Trades: {trades}
🎯 Win Rate: {win_rate:.1f}%
💰 Balance: ₹{summary.get('balance', 0):,.2f}
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
        """Get alert history"""
        history = self.alert_history

        if alert_type:
            history = [a for a in history if a.alert_type == alert_type]

        return history[-limit:]

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
