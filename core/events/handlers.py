"""
Event Handler Registry for Event-Driven Trading Engine.

This module provides handler registration and management for events.
Handlers are functions that process specific event types.

Features:
- Type-safe handler registration
- Support for sync and async handlers
- Handler priority ordering
- Error handling and logging
- Handler groups for bulk operations
"""

import asyncio
import functools
import inspect
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
)
from weakref import WeakMethod, ref

from .events import Event, EventType


logger = logging.getLogger(__name__)


# Type alias for event handlers
EventHandler = Callable[[Event], Any]
AsyncEventHandler = Callable[[Event], Any]  # Returns awaitable


@dataclass
class HandlerRegistration:
    """
    Represents a registered event handler.

    Attributes:
        handler: The handler function
        event_types: Set of event types this handler processes
        priority: Handler priority (lower = called first)
        name: Optional handler name for debugging
        group: Optional group name for bulk operations
        is_async: Whether handler is async
        weak: Whether to use weak reference (auto-cleanup)
    """
    handler: EventHandler
    event_types: Set[EventType]
    priority: int = 100
    name: str = ""
    group: str = ""
    is_async: bool = False
    weak: bool = False
    _weak_ref: Optional[ref] = field(default=None, repr=False)

    def __post_init__(self):
        if not self.name:
            self.name = getattr(self.handler, '__name__', str(self.handler))
        self.is_async = asyncio.iscoroutinefunction(self.handler)

        # Create weak reference if requested
        if self.weak:
            if inspect.ismethod(self.handler):
                self._weak_ref = WeakMethod(self.handler)
            else:
                self._weak_ref = ref(self.handler)

    def get_handler(self) -> Optional[EventHandler]:
        """Get handler, resolving weak reference if needed."""
        if self._weak_ref is not None:
            return self._weak_ref()
        return self.handler

    def is_alive(self) -> bool:
        """Check if handler is still valid (for weak refs)."""
        if self._weak_ref is not None:
            return self._weak_ref() is not None
        return True


class HandlerRegistry:
    """
    Registry for event handlers.

    Manages handler registration, lookup, and invocation.

    Usage:
        registry = HandlerRegistry()

        # Register handler for specific event types
        registry.register(
            handler=my_handler,
            event_types=[EventType.TICK, EventType.BAR],
            priority=10,
            name="market_data_handler"
        )

        # Get handlers for an event
        handlers = registry.get_handlers(tick_event)
        for reg in handlers:
            reg.handler(tick_event)

        # Unregister by name
        registry.unregister("market_data_handler")
    """

    def __init__(self):
        self._handlers: List[HandlerRegistration] = []
        self._by_type: Dict[EventType, List[HandlerRegistration]] = {}
        self._by_name: Dict[str, HandlerRegistration] = {}
        self._by_group: Dict[str, List[HandlerRegistration]] = {}
        self._all_handlers: List[HandlerRegistration] = []  # Handlers that receive all events

    def register(
        self,
        handler: EventHandler,
        event_types: Optional[Union[EventType, List[EventType], Set[EventType]]] = None,
        priority: int = 100,
        name: str = "",
        group: str = "",
        weak: bool = False
    ) -> str:
        """
        Register an event handler.

        Args:
            handler: Function to handle events
            event_types: Event type(s) to handle. None = all events.
            priority: Handler priority (lower = called first)
            name: Handler name (auto-generated if not provided)
            group: Group name for bulk operations
            weak: Use weak reference (auto-cleanup when handler is garbage collected)

        Returns:
            Handler name (for later unregistration)
        """
        # Normalize event_types to a set
        if event_types is None:
            types_set = set()  # Empty = all events
        elif isinstance(event_types, EventType):
            types_set = {event_types}
        elif isinstance(event_types, list):
            types_set = set(event_types)
        else:
            types_set = event_types

        # Create registration
        reg = HandlerRegistration(
            handler=handler,
            event_types=types_set,
            priority=priority,
            name=name,
            group=group,
            weak=weak
        )

        # Ensure unique name
        if reg.name in self._by_name:
            # Generate unique name
            base_name = reg.name
            counter = 1
            while f"{base_name}_{counter}" in self._by_name:
                counter += 1
            reg.name = f"{base_name}_{counter}"

        # Store in all data structures
        self._handlers.append(reg)
        self._by_name[reg.name] = reg

        if not types_set:
            # Handler receives all events
            self._all_handlers.append(reg)
        else:
            # Handler receives specific event types
            for event_type in types_set:
                if event_type not in self._by_type:
                    self._by_type[event_type] = []
                self._by_type[event_type].append(reg)

        if group:
            if group not in self._by_group:
                self._by_group[group] = []
            self._by_group[group].append(reg)

        # Sort by priority
        self._sort_handlers()

        logger.debug(f"Registered handler: {reg.name} for types: {types_set or 'ALL'}")
        return reg.name

    def _sort_handlers(self):
        """Sort all handler lists by priority."""
        self._handlers.sort(key=lambda r: r.priority)
        self._all_handlers.sort(key=lambda r: r.priority)
        for handlers in self._by_type.values():
            handlers.sort(key=lambda r: r.priority)
        for handlers in self._by_group.values():
            handlers.sort(key=lambda r: r.priority)

    def unregister(self, name: str) -> bool:
        """
        Unregister a handler by name.

        Args:
            name: Handler name

        Returns:
            True if handler was found and removed
        """
        if name not in self._by_name:
            return False

        reg = self._by_name.pop(name)
        self._handlers.remove(reg)

        if not reg.event_types:
            self._all_handlers.remove(reg)
        else:
            for event_type in reg.event_types:
                if event_type in self._by_type:
                    self._by_type[event_type].remove(reg)

        if reg.group and reg.group in self._by_group:
            self._by_group[reg.group].remove(reg)

        logger.debug(f"Unregistered handler: {name}")
        return True

    def unregister_group(self, group: str) -> int:
        """
        Unregister all handlers in a group.

        Args:
            group: Group name

        Returns:
            Number of handlers removed
        """
        if group not in self._by_group:
            return 0

        handlers = list(self._by_group[group])
        count = 0
        for reg in handlers:
            if self.unregister(reg.name):
                count += 1

        return count

    def get_handlers(self, event: Event) -> List[HandlerRegistration]:
        """
        Get all handlers for an event.

        Args:
            event: The event to find handlers for

        Returns:
            List of handler registrations, sorted by priority
        """
        # Clean up dead weak references
        self._cleanup_dead_handlers()

        handlers = []

        # Add type-specific handlers
        if event.event_type in self._by_type:
            handlers.extend(self._by_type[event.event_type])

        # Add catch-all handlers
        handlers.extend(self._all_handlers)

        # Sort by priority and deduplicate
        seen = set()
        unique = []
        for reg in sorted(handlers, key=lambda r: r.priority):
            if reg.name not in seen and reg.is_alive():
                seen.add(reg.name)
                unique.append(reg)

        return unique

    def _cleanup_dead_handlers(self):
        """Remove handlers with dead weak references."""
        dead = [reg for reg in self._handlers if not reg.is_alive()]
        for reg in dead:
            self.unregister(reg.name)

    def get_handler_count(self, event_type: Optional[EventType] = None) -> int:
        """Get number of registered handlers."""
        if event_type is None:
            return len(self._handlers)
        return len(self._by_type.get(event_type, [])) + len(self._all_handlers)

    def clear(self):
        """Remove all handlers."""
        self._handlers.clear()
        self._by_type.clear()
        self._by_name.clear()
        self._by_group.clear()
        self._all_handlers.clear()

    def list_handlers(self) -> List[Dict[str, Any]]:
        """List all registered handlers."""
        return [
            {
                'name': reg.name,
                'event_types': [t.name for t in reg.event_types] or ['ALL'],
                'priority': reg.priority,
                'group': reg.group,
                'is_async': reg.is_async,
                'is_alive': reg.is_alive()
            }
            for reg in self._handlers
        ]


def on_event(*event_types: EventType, priority: int = 100, group: str = ""):
    """
    Decorator for marking methods as event handlers.

    Usage:
        class MyStrategy:
            @on_event(EventType.TICK, EventType.BAR)
            def handle_market_data(self, event):
                pass

            @on_event(EventType.SIGNAL_GENERATED, priority=10)
            def handle_signal(self, event):
                pass
    """
    def decorator(func):
        # Store metadata on the function
        func._event_handler_info = {
            'event_types': set(event_types) if event_types else set(),
            'priority': priority,
            'group': group
        }
        return func
    return decorator


def collect_handlers(obj: Any) -> List[tuple]:
    """
    Collect all methods decorated with @on_event from an object.

    Args:
        obj: Object to scan for handlers

    Returns:
        List of (method, handler_info) tuples
    """
    handlers = []
    for name in dir(obj):
        if name.startswith('_'):
            continue
        method = getattr(obj, name, None)
        if method and callable(method):
            info = getattr(method, '_event_handler_info', None)
            if info:
                handlers.append((method, info))
    return handlers


def register_object_handlers(bus_or_registry: Any, obj: Any, group: str = "") -> List[str]:
    """
    Register all @on_event decorated methods from an object.

    Args:
        bus_or_registry: EventBus or HandlerRegistry instance
        obj: Object with decorated methods
        group: Optional group name override

    Returns:
        List of registered handler names
    """
    # Support both EventBus and HandlerRegistry
    if hasattr(bus_or_registry, 'subscribe'):
        # It's an EventBus
        bus = bus_or_registry
        names = []
        for method, info in collect_handlers(obj):
            handler_group = group or info.get('group', '')
            event_types = list(info['event_types']) if info['event_types'] else None
            name = bus.subscribe(
                event_types=event_types,
                handler=method,
                priority=info['priority'],
                group=handler_group,
                name=f"{obj.__class__.__name__}.{method.__name__}"
            )
            names.append(name)
        return names
    else:
        # It's a HandlerRegistry
        registry = bus_or_registry
        names = []
        for method, info in collect_handlers(obj):
            handler_group = group or info.get('group', '')
            name = registry.register(
                handler=method,
                event_types=info['event_types'],
                priority=info['priority'],
                group=handler_group,
                name=f"{obj.__class__.__name__}.{method.__name__}"
            )
            names.append(name)
        return names
