# -*- coding: utf-8 -*-
"""
Connection Optimization - WebSocket & REST Connection Pooling
==============================================================
Manages persistent connections for minimal latency.

Features:
- Persistent WebSocket connections with auto-reconnect
- REST API connection pooling with keep-alive
- Co-location awareness for optimal routing
- Connection health monitoring

Example:
    >>> from core.infrastructure import WebSocketManager, ConnectionPool
    >>>
    >>> # WebSocket management
    >>> ws_manager = WebSocketManager()
    >>> ws_manager.connect("wss://ws.kite.trade")
    >>> ws_manager.subscribe(["NSE:RELIANCE", "NSE:TCS"])
    >>>
    >>> # REST connection pooling
    >>> pool = ConnectionPool(max_connections=10)
    >>> response = pool.get("https://api.kite.trade/instruments")
"""

import asyncio
import threading
import time
import socket
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Callable, Any, Set, Tuple
from datetime import datetime, timedelta
from collections import deque
from urllib.parse import urlparse
import logging
import json
import queue

# Try to import websocket libraries
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Try to import requests with urllib3 for connection pooling
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    CLOSED = "closed"


class DataCenter(Enum):
    """Known data center locations for co-location."""
    # Indian exchanges
    NSE_COLO = "nse_colo"           # NSE Colocation (Mumbai)
    BSE_COLO = "bse_colo"           # BSE Colocation (Mumbai)

    # Cloud regions close to exchanges
    AWS_MUMBAI = "ap-south-1"       # AWS Mumbai
    GCP_MUMBAI = "asia-south1"      # GCP Mumbai
    AZURE_INDIA = "centralindia"    # Azure Central India

    # Broker locations
    ZERODHA_DC = "zerodha_dc"       # Zerodha Data Center

    # Generic
    UNKNOWN = "unknown"


@dataclass
class ConnectionStats:
    """Statistics for a connection."""
    total_connects: int = 0
    total_disconnects: int = 0
    total_reconnects: int = 0
    total_errors: int = 0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0

    # Timing
    last_connect_time: Optional[datetime] = None
    last_disconnect_time: Optional[datetime] = None
    last_message_time: Optional[datetime] = None

    # Latency
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self.latency_samples.append(latency_ms)
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.avg_latency_ms = sum(self.latency_samples) / len(self.latency_samples)

    @property
    def uptime_seconds(self) -> float:
        """Get connection uptime in seconds."""
        if self.last_connect_time is None:
            return 0.0
        return (datetime.now() - self.last_connect_time).total_seconds()


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket connections."""
    # Connection settings
    url: str = ""
    ping_interval: int = 30          # Seconds between pings
    ping_timeout: int = 10           # Seconds to wait for pong
    reconnect_delay: float = 1.0     # Initial reconnect delay
    reconnect_delay_max: float = 60.0  # Max reconnect delay
    reconnect_multiplier: float = 2.0  # Exponential backoff multiplier
    max_reconnect_attempts: int = 0  # 0 = infinite

    # SSL settings
    ssl_verify: bool = True
    ssl_cert: Optional[str] = None

    # Buffer settings
    message_queue_size: int = 10000
    send_queue_size: int = 1000

    # Timeouts
    connect_timeout: int = 30
    read_timeout: int = 60

    # Headers
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ConnectionPoolConfig:
    """Configuration for REST connection pool."""
    # Pool settings
    max_connections: int = 10
    max_connections_per_host: int = 5

    # Retry settings
    max_retries: int = 3
    retry_backoff_factor: float = 0.5
    retry_statuses: List[int] = field(default_factory=lambda: [500, 502, 503, 504])

    # Timeout settings
    connect_timeout: float = 5.0
    read_timeout: float = 30.0

    # Keep-alive
    keep_alive: bool = True
    pool_block: bool = False

    # SSL
    ssl_verify: bool = True


@dataclass
class CoLocationConfig:
    """Configuration for co-location optimization."""
    # Current deployment location
    current_datacenter: DataCenter = DataCenter.UNKNOWN

    # Target exchange/broker
    target_endpoints: List[str] = field(default_factory=list)

    # Optimization preferences
    prefer_tcp_nodelay: bool = True
    prefer_keep_alive: bool = True
    socket_buffer_size: int = 65536

    # Latency thresholds
    excellent_latency_ms: float = 1.0   # Co-located
    good_latency_ms: float = 5.0        # Same region
    acceptable_latency_ms: float = 20.0  # Same country


class WebSocketConnection:
    """
    Persistent WebSocket connection with auto-reconnect.

    Handles:
    - Automatic reconnection with exponential backoff
    - Ping/pong keepalive
    - Message queuing during disconnection
    - Thread-safe operations

    Example:
        >>> ws = WebSocketConnection(config)
        >>> ws.on_message = lambda msg: print(f"Received: {msg}")
        >>> ws.connect()
        >>> ws.send({"action": "subscribe", "symbols": ["RELIANCE"]})
    """

    def __init__(self, config: WebSocketConfig):
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self.stats = ConnectionStats()

        # WebSocket instance
        self._ws: Optional[Any] = None
        self._ws_thread: Optional[threading.Thread] = None

        # Message queues
        self._receive_queue: queue.Queue = queue.Queue(maxsize=config.message_queue_size)
        self._send_queue: queue.Queue = queue.Queue(maxsize=config.send_queue_size)

        # Reconnect state
        self._reconnect_attempts = 0
        self._current_reconnect_delay = config.reconnect_delay

        # Thread control
        self._running = False
        self._lock = threading.RLock()

        # Callbacks
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[str], None]] = None
        self.on_message: Optional[Callable[[Any], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self.state == ConnectionState.CONNECTED

    def connect(self) -> bool:
        """
        Connect to WebSocket server.

        Returns True if connection initiated successfully.
        """
        if not WEBSOCKET_AVAILABLE:
            logger.error("websocket-client library not installed")
            return False

        with self._lock:
            if self._running:
                return True

            self._running = True
            self.state = ConnectionState.CONNECTING

        # Start connection thread
        self._ws_thread = threading.Thread(
            target=self._connection_loop,
            daemon=True,
            name=f"WebSocket-{self.config.url[:30]}"
        )
        self._ws_thread.start()

        return True

    def disconnect(self) -> None:
        """Disconnect from server."""
        with self._lock:
            self._running = False
            self.state = ConnectionState.CLOSED

        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5.0)

        self.stats.last_disconnect_time = datetime.now()
        self.stats.total_disconnects += 1

    def send(self, message: Any) -> bool:
        """
        Send message to server.

        Message is queued and sent asynchronously.
        Returns True if queued successfully.
        """
        if not self._running:
            return False

        try:
            # Convert to JSON if dict
            if isinstance(message, dict):
                message = json.dumps(message)

            self._send_queue.put_nowait(message)
            return True
        except queue.Full:
            logger.warning("Send queue full, message dropped")
            return False

    def receive(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Receive message from queue.

        Returns None if no message available within timeout.
        """
        try:
            return self._receive_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _connection_loop(self) -> None:
        """Main connection loop with reconnect logic."""
        while self._running:
            try:
                self._establish_connection()

                if self.is_connected:
                    self._run_connection()

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.stats.total_errors += 1
                if self.on_error:
                    try:
                        self.on_error(e)
                    except Exception:
                        pass

            # Handle reconnect
            if self._running and self.state != ConnectionState.CLOSED:
                self._handle_reconnect()

    def _establish_connection(self) -> None:
        """Establish WebSocket connection."""
        self.state = ConnectionState.CONNECTING

        # Build WebSocket
        self._ws = websocket.WebSocketApp(
            self.config.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            header=self.config.headers or None
        )

        # SSL options
        ssl_opts = {}
        if self.config.url.startswith("wss://"):
            if not self.config.ssl_verify:
                ssl_opts = {"cert_reqs": ssl.CERT_NONE}
            elif self.config.ssl_cert:
                ssl_opts = {"ca_certs": self.config.ssl_cert}

        # Run WebSocket (blocking)
        self._ws.run_forever(
            ping_interval=self.config.ping_interval,
            ping_timeout=self.config.ping_timeout,
            sslopt=ssl_opts if ssl_opts else None
        )

    def _run_connection(self) -> None:
        """Run connection - process send queue."""
        while self._running and self.is_connected:
            try:
                # Process send queue
                try:
                    message = self._send_queue.get(timeout=0.1)
                    if self._ws:
                        self._ws.send(message)
                        self.stats.total_messages_sent += 1
                        self.stats.total_bytes_sent += len(message)
                except queue.Empty:
                    pass
            except Exception as e:
                logger.error(f"Send error: {e}")
                break

    def _handle_reconnect(self) -> None:
        """Handle reconnection with exponential backoff."""
        if self.config.max_reconnect_attempts > 0:
            if self._reconnect_attempts >= self.config.max_reconnect_attempts:
                logger.error("Max reconnect attempts reached")
                self._running = False
                self.state = ConnectionState.ERROR
                return

        self.state = ConnectionState.RECONNECTING
        self._reconnect_attempts += 1
        self.stats.total_reconnects += 1

        # Wait with exponential backoff
        logger.info(f"Reconnecting in {self._current_reconnect_delay:.1f}s (attempt {self._reconnect_attempts})")
        time.sleep(self._current_reconnect_delay)

        # Increase delay for next attempt
        self._current_reconnect_delay = min(
            self._current_reconnect_delay * self.config.reconnect_multiplier,
            self.config.reconnect_delay_max
        )

    def _on_open(self, ws) -> None:
        """Handle connection open."""
        logger.info(f"WebSocket connected: {self.config.url}")
        self.state = ConnectionState.CONNECTED
        self._reconnect_attempts = 0
        self._current_reconnect_delay = self.config.reconnect_delay

        self.stats.total_connects += 1
        self.stats.last_connect_time = datetime.now()

        if self.on_connect:
            try:
                self.on_connect()
            except Exception as e:
                logger.error(f"on_connect callback error: {e}")

    def _on_message(self, ws, message) -> None:
        """Handle incoming message."""
        self.stats.total_messages_received += 1
        self.stats.total_bytes_received += len(message)
        self.stats.last_message_time = datetime.now()

        # Try to parse JSON
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            data = message

        # Queue message
        try:
            self._receive_queue.put_nowait(data)
        except queue.Full:
            logger.warning("Receive queue full, message dropped")

        # Callback
        if self.on_message:
            try:
                self.on_message(data)
            except Exception as e:
                logger.error(f"on_message callback error: {e}")

    def _on_error(self, ws, error) -> None:
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")
        self.stats.total_errors += 1

        if self.on_error:
            try:
                self.on_error(error)
            except Exception:
                pass

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """Handle connection close."""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.state = ConnectionState.DISCONNECTED
        self.stats.last_disconnect_time = datetime.now()

        if self.on_disconnect:
            try:
                self.on_disconnect(close_msg or "")
            except Exception as e:
                logger.error(f"on_disconnect callback error: {e}")


class WebSocketManager:
    """
    Manages multiple WebSocket connections.

    Example:
        >>> manager = WebSocketManager()
        >>> manager.add_connection("market", "wss://ws.kite.trade")
        >>> manager.connect_all()
        >>> manager.send("market", {"action": "subscribe"})
    """

    def __init__(self):
        self._connections: Dict[str, WebSocketConnection] = {}
        self._lock = threading.RLock()

    def add_connection(
        self,
        name: str,
        url: str,
        config: Optional[WebSocketConfig] = None
    ) -> WebSocketConnection:
        """Add a new WebSocket connection."""
        if config is None:
            config = WebSocketConfig(url=url)
        else:
            config.url = url

        conn = WebSocketConnection(config)

        with self._lock:
            self._connections[name] = conn

        return conn

    def get_connection(self, name: str) -> Optional[WebSocketConnection]:
        """Get connection by name."""
        return self._connections.get(name)

    def connect(self, name: str) -> bool:
        """Connect specific connection."""
        conn = self._connections.get(name)
        if conn:
            return conn.connect()
        return False

    def connect_all(self) -> None:
        """Connect all connections."""
        for name, conn in self._connections.items():
            try:
                conn.connect()
            except Exception as e:
                logger.error(f"Failed to connect {name}: {e}")

    def disconnect(self, name: str) -> None:
        """Disconnect specific connection."""
        conn = self._connections.get(name)
        if conn:
            conn.disconnect()

    def disconnect_all(self) -> None:
        """Disconnect all connections."""
        for conn in self._connections.values():
            try:
                conn.disconnect()
            except Exception:
                pass

    def send(self, name: str, message: Any) -> bool:
        """Send message to specific connection."""
        conn = self._connections.get(name)
        if conn:
            return conn.send(message)
        return False

    def get_stats(self) -> Dict[str, ConnectionStats]:
        """Get stats for all connections."""
        return {name: conn.stats for name, conn in self._connections.items()}

    @property
    def all_connected(self) -> bool:
        """Check if all connections are connected."""
        return all(conn.is_connected for conn in self._connections.values())


class ConnectionPool:
    """
    HTTP Connection Pool with keep-alive and retries.

    Uses requests Session with HTTPAdapter for connection pooling.

    Example:
        >>> pool = ConnectionPool()
        >>> response = pool.get("https://api.kite.trade/instruments")
        >>> data = pool.post("https://api.kite.trade/orders", json=order_data)
    """

    def __init__(self, config: Optional[ConnectionPoolConfig] = None):
        self.config = config or ConnectionPoolConfig()
        self._sessions: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self.stats = ConnectionStats()

        # Create default session
        self._default_session = self._create_session()

    def _create_session(self, base_url: Optional[str] = None) -> Any:
        """Create a requests Session with connection pooling."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library not installed")

        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff_factor,
            status_forcelist=self.config.retry_statuses,
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )

        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=self.config.max_connections,
            pool_maxsize=self.config.max_connections_per_host,
            max_retries=retry_strategy,
            pool_block=self.config.pool_block
        )

        # Mount adapter for both http and https
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Configure SSL
        session.verify = self.config.ssl_verify

        return session

    def get_session(self, base_url: Optional[str] = None) -> Any:
        """Get or create session for base URL."""
        if base_url is None:
            return self._default_session

        with self._lock:
            if base_url not in self._sessions:
                self._sessions[base_url] = self._create_session(base_url)
            return self._sessions[base_url]

    def request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> Any:
        """
        Make HTTP request with connection pooling.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional arguments for requests

        Returns:
            requests.Response object
        """
        # Set default timeouts
        if 'timeout' not in kwargs:
            kwargs['timeout'] = (
                self.config.connect_timeout,
                self.config.read_timeout
            )

        # Get appropriate session
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        session = self.get_session(base_url)

        # Make request and track stats
        start_time = time.perf_counter()
        try:
            response = session.request(method, url, **kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.stats.record_latency(latency_ms)
            self.stats.total_messages_sent += 1
            self.stats.total_messages_received += 1
            return response
        except Exception as e:
            self.stats.total_errors += 1
            raise

    def get(self, url: str, **kwargs) -> Any:
        """HTTP GET request."""
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> Any:
        """HTTP POST request."""
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs) -> Any:
        """HTTP PUT request."""
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs) -> Any:
        """HTTP DELETE request."""
        return self.request("DELETE", url, **kwargs)

    def close(self) -> None:
        """Close all sessions."""
        with self._lock:
            for session in self._sessions.values():
                try:
                    session.close()
                except Exception:
                    pass
            self._sessions.clear()

            try:
                self._default_session.close()
            except Exception:
                pass


class CoLocationOptimizer:
    """
    Utilities for co-location and latency optimization.

    Provides tools to:
    - Measure network latency to endpoints
    - Configure optimal socket settings
    - Detect current datacenter location
    - Recommend optimal deployment locations

    Example:
        >>> optimizer = CoLocationOptimizer()
        >>> latency = optimizer.measure_latency("api.kite.trade", port=443)
        >>> print(f"Latency to Zerodha: {latency:.2f}ms")
        >>>
        >>> optimizer.optimize_socket(sock)  # Apply optimal settings
    """

    def __init__(self, config: Optional[CoLocationConfig] = None):
        self.config = config or CoLocationConfig()
        self._latency_cache: Dict[str, Tuple[float, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    def measure_latency(
        self,
        host: str,
        port: int = 443,
        samples: int = 5
    ) -> float:
        """
        Measure TCP connection latency to host.

        Returns average latency in milliseconds.
        """
        # Check cache
        cache_key = f"{host}:{port}"
        if cache_key in self._latency_cache:
            latency, timestamp = self._latency_cache[cache_key]
            if datetime.now() - timestamp < self._cache_ttl:
                return latency

        latencies = []

        for _ in range(samples):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)

                start = time.perf_counter()
                sock.connect((host, port))
                latency = (time.perf_counter() - start) * 1000

                latencies.append(latency)
                sock.close()

            except Exception as e:
                logger.debug(f"Latency measurement failed: {e}")

            time.sleep(0.1)  # Brief pause between samples

        if not latencies:
            return float('inf')

        avg_latency = sum(latencies) / len(latencies)

        # Cache result
        self._latency_cache[cache_key] = (avg_latency, datetime.now())

        return avg_latency

    def optimize_socket(self, sock: socket.socket) -> None:
        """
        Apply optimal socket settings for low latency.

        Settings applied:
        - TCP_NODELAY: Disable Nagle's algorithm
        - SO_KEEPALIVE: Enable keepalive
        - Buffer sizes: Increase for throughput
        """
        try:
            # Disable Nagle's algorithm for lower latency
            if self.config.prefer_tcp_nodelay:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # Enable keepalive
            if self.config.prefer_keep_alive:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

            # Set buffer sizes
            sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF,
                self.config.socket_buffer_size
            )
            sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF,
                self.config.socket_buffer_size
            )

        except Exception as e:
            logger.warning(f"Could not optimize socket: {e}")

    def get_latency_rating(self, latency_ms: float) -> str:
        """
        Rate latency quality.

        Returns: 'excellent', 'good', 'acceptable', or 'poor'
        """
        if latency_ms <= self.config.excellent_latency_ms:
            return "excellent"
        elif latency_ms <= self.config.good_latency_ms:
            return "good"
        elif latency_ms <= self.config.acceptable_latency_ms:
            return "acceptable"
        else:
            return "poor"

    def analyze_endpoints(
        self,
        endpoints: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze latency to multiple endpoints.

        Returns dict with latency analysis per endpoint.
        """
        endpoints = endpoints or self.config.target_endpoints

        if not endpoints:
            # Default Indian trading endpoints
            endpoints = [
                "api.kite.trade",      # Zerodha
                "ant.aliceblueonline.com",  # Alice Blue
                "api.upstox.com",      # Upstox
            ]

        results = {}

        for endpoint in endpoints:
            try:
                # Parse host from URL if needed
                if "://" in endpoint:
                    parsed = urlparse(endpoint)
                    host = parsed.netloc or parsed.path
                    port = parsed.port or (443 if parsed.scheme == "https" else 80)
                else:
                    host = endpoint
                    port = 443

                latency = self.measure_latency(host, port)
                rating = self.get_latency_rating(latency)

                results[endpoint] = {
                    'host': host,
                    'port': port,
                    'latency_ms': round(latency, 2),
                    'rating': rating,
                    'recommendation': self._get_recommendation(rating, latency)
                }

            except Exception as e:
                results[endpoint] = {
                    'error': str(e),
                    'rating': 'unknown'
                }

        return results

    def _get_recommendation(self, rating: str, latency_ms: float) -> str:
        """Get optimization recommendation based on latency."""
        if rating == "excellent":
            return "Optimal - likely co-located or same datacenter"
        elif rating == "good":
            return "Good - same region, consider co-location for HFT"
        elif rating == "acceptable":
            return "Acceptable - consider moving to Mumbai region"
        else:
            return f"Poor ({latency_ms:.0f}ms) - strongly recommend Mumbai datacenter"

    def get_colocation_info(self) -> Dict[str, Any]:
        """
        Get co-location information and recommendations.

        Returns dict with deployment recommendations.
        """
        return {
            'current_datacenter': self.config.current_datacenter.value,
            'recommendations': {
                'nse_bse': {
                    'optimal': 'NSE/BSE Colocation facility in Mumbai',
                    'cloud_alternative': 'AWS ap-south-1 (Mumbai) or GCP asia-south1',
                    'latency_target': '<1ms for co-location, <5ms for cloud'
                },
                'zerodha': {
                    'api_endpoint': 'api.kite.trade',
                    'websocket': 'wss://ws.kite.trade',
                    'recommended_region': 'Mumbai (AWS ap-south-1)'
                }
            },
            'socket_optimizations': {
                'tcp_nodelay': self.config.prefer_tcp_nodelay,
                'keepalive': self.config.prefer_keep_alive,
                'buffer_size': self.config.socket_buffer_size
            },
            'thresholds': {
                'excellent_ms': self.config.excellent_latency_ms,
                'good_ms': self.config.good_latency_ms,
                'acceptable_ms': self.config.acceptable_latency_ms
            }
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

_ws_manager: Optional[WebSocketManager] = None
_connection_pool: Optional[ConnectionPool] = None
_colo_optimizer: Optional[CoLocationOptimizer] = None


def get_websocket_manager() -> WebSocketManager:
    """Get global WebSocket manager."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager


def set_websocket_manager(manager: WebSocketManager) -> None:
    """Set global WebSocket manager."""
    global _ws_manager
    _ws_manager = manager


def get_connection_pool() -> ConnectionPool:
    """Get global connection pool."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = ConnectionPool()
    return _connection_pool


def set_connection_pool(pool: ConnectionPool) -> None:
    """Set global connection pool."""
    global _connection_pool
    _connection_pool = pool


def get_colocation_optimizer() -> CoLocationOptimizer:
    """Get global co-location optimizer."""
    global _colo_optimizer
    if _colo_optimizer is None:
        _colo_optimizer = CoLocationOptimizer()
    return _colo_optimizer


def measure_latency(host: str, port: int = 443) -> float:
    """Measure latency to host using global optimizer."""
    return get_colocation_optimizer().measure_latency(host, port)


def analyze_network() -> Dict[str, Any]:
    """Analyze network latency to common trading endpoints."""
    return get_colocation_optimizer().analyze_endpoints()
