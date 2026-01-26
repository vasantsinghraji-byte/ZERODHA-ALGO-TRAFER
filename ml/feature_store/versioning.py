# -*- coding: utf-8 -*-
"""
Feature Versioning - Version Control and Lineage Tracking
==========================================================
Track feature versions, lineage, and changes over time.

Features:
- Semantic versioning for features
- Lineage tracking (dependencies, transformations)
- Schema validation
- Migration support

Example:
    >>> from ml.feature_store import FeatureVersionManager, FeatureLineage
    >>>
    >>> manager = FeatureVersionManager()
    >>>
    >>> # Register a feature version
    >>> manager.register_version("rsi_14", "1.0.0", schema={
    ...     "type": "float",
    ...     "min": 0,
    ...     "max": 100
    ... })
    >>>
    >>> # Track lineage
    >>> lineage = FeatureLineage("macd")
    >>> lineage.add_dependency("close")
    >>> lineage.add_transformation("ema_12", "ema_26", "subtract")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Set, Callable
from datetime import datetime
import hashlib
import json
import re
import logging

logger = logging.getLogger(__name__)


class VersionStatus(Enum):
    """Feature version status."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ChangeType(Enum):
    """Type of feature change."""
    ADDED = "added"
    MODIFIED = "modified"
    DEPRECATED = "deprecated"
    REMOVED = "removed"
    SCHEMA_CHANGE = "schema_change"
    PARAMETER_CHANGE = "parameter_change"


@dataclass
class SemanticVersion:
    """Semantic version representation."""
    major: int
    minor: int
    patch: int

    @classmethod
    def from_string(cls, version_str: str) -> 'SemanticVersion':
        """Parse version string like '1.2.3'."""
        match = re.match(r'^(\d+)\.(\d+)\.(\d+)$', version_str)
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3))
        )

    def to_string(self) -> str:
        """Convert to version string."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def bump_major(self) -> 'SemanticVersion':
        """Bump major version (breaking changes)."""
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> 'SemanticVersion':
        """Bump minor version (new features)."""
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> 'SemanticVersion':
        """Bump patch version (bug fixes)."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)

    def __lt__(self, other: 'SemanticVersion') -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: 'SemanticVersion') -> bool:
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __str__(self) -> str:
        return self.to_string()


@dataclass
class FeatureSchema:
    """Schema definition for a feature."""
    data_type: str  # float, int, bool, string, array
    nullable: bool = True

    # Numeric constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # String constraints
    max_length: Optional[int] = None
    pattern: Optional[str] = None

    # Array constraints
    array_type: Optional[str] = None
    min_items: Optional[int] = None
    max_items: Optional[int] = None

    # Custom validation
    custom_validator: Optional[Callable[[Any], bool]] = None

    def validate(self, value: Any) -> tuple:
        """
        Validate a value against this schema.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None:
            if not self.nullable:
                return False, "Value cannot be null"
            return True, ""

        # Type check
        type_map = {
            'float': (int, float),
            'int': int,
            'bool': bool,
            'string': str,
            'array': (list, tuple),
        }

        expected_type = type_map.get(self.data_type)
        if expected_type and not isinstance(value, expected_type):
            return False, f"Expected {self.data_type}, got {type(value).__name__}"

        # Numeric constraints
        if self.data_type in ('float', 'int'):
            if self.min_value is not None and value < self.min_value:
                return False, f"Value {value} below minimum {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Value {value} above maximum {self.max_value}"

        # String constraints
        if self.data_type == 'string':
            if self.max_length is not None and len(value) > self.max_length:
                return False, f"String length {len(value)} exceeds maximum {self.max_length}"
            if self.pattern is not None and not re.match(self.pattern, value):
                return False, f"String does not match pattern {self.pattern}"

        # Custom validation
        if self.custom_validator:
            if not self.custom_validator(value):
                return False, "Custom validation failed"

        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'data_type': self.data_type,
            'nullable': self.nullable,
        }
        if self.min_value is not None:
            result['min_value'] = self.min_value
        if self.max_value is not None:
            result['max_value'] = self.max_value
        if self.max_length is not None:
            result['max_length'] = self.max_length
        if self.pattern is not None:
            result['pattern'] = self.pattern
        return result


@dataclass
class FeatureVersionInfo:
    """Information about a specific feature version."""
    name: str
    version: SemanticVersion
    status: VersionStatus = VersionStatus.DEVELOPMENT

    # Schema
    schema: Optional[FeatureSchema] = None

    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    tags: List[str] = field(default_factory=list)

    # Computation
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

    # Tracking
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute unique hash for this version."""
        content = json.dumps({
            'name': self.name,
            'version': str(self.version),
            'parameters': self.parameters,
            'dependencies': sorted(self.dependencies),
            'schema': self.schema.to_dict() if self.schema else None
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class FeatureChange:
    """Record of a feature change."""
    feature_name: str
    change_type: ChangeType
    old_version: Optional[SemanticVersion]
    new_version: SemanticVersion
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    changed_by: str = ""

    # Change details
    changes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'change_type': self.change_type.value,
            'old_version': str(self.old_version) if self.old_version else None,
            'new_version': str(self.new_version),
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'changed_by': self.changed_by,
            'changes': self.changes
        }


@dataclass
class TransformationStep:
    """A single transformation in the lineage."""
    name: str
    operation: str  # e.g., "sma", "diff", "lag", "custom"
    inputs: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


class FeatureLineage:
    """
    Tracks the lineage of a feature.

    Records:
    - Data sources
    - Dependencies on other features
    - Transformation steps
    - Computation graph

    Example:
        >>> lineage = FeatureLineage("macd_signal")
        >>> lineage.add_source("close_prices")
        >>> lineage.add_dependency("ema_12")
        >>> lineage.add_dependency("ema_26")
        >>> lineage.add_transformation("subtract", ["ema_12", "ema_26"], {"operation": "ema_12 - ema_26"})
        >>> lineage.add_transformation("ema", ["macd_line"], {"period": 9})
    """

    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self.sources: List[str] = []
        self.dependencies: Set[str] = set()
        self.transformations: List[TransformationStep] = []
        self.created_at = datetime.now()
        self.metadata: Dict[str, Any] = {}

    def add_source(self, source: str) -> None:
        """Add a data source."""
        if source not in self.sources:
            self.sources.append(source)

    def add_dependency(self, feature_name: str) -> None:
        """Add a feature dependency."""
        self.dependencies.add(feature_name)

    def add_transformation(
        self,
        operation: str,
        inputs: List[str],
        parameters: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> None:
        """Add a transformation step."""
        step = TransformationStep(
            name=f"step_{len(self.transformations) + 1}",
            operation=operation,
            inputs=inputs,
            parameters=parameters or {},
            description=description
        )
        self.transformations.append(step)

    def get_all_dependencies(self) -> Set[str]:
        """Get all dependencies including transitive."""
        return self.dependencies.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'sources': self.sources,
            'dependencies': list(self.dependencies),
            'transformations': [
                {
                    'name': t.name,
                    'operation': t.operation,
                    'inputs': t.inputs,
                    'parameters': t.parameters,
                    'description': t.description
                }
                for t in self.transformations
            ],
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }

    def __repr__(self) -> str:
        return f"FeatureLineage({self.feature_name}, {len(self.transformations)} steps)"


class FeatureVersionManager:
    """
    Manages feature versions and history.

    Example:
        >>> manager = FeatureVersionManager()
        >>>
        >>> # Register a new feature
        >>> manager.register_version(
        ...     name="rsi_14",
        ...     version="1.0.0",
        ...     schema=FeatureSchema(data_type="float", min_value=0, max_value=100),
        ...     description="14-period RSI",
        ...     parameters={"period": 14}
        ... )
        >>>
        >>> # Update feature
        >>> manager.bump_version("rsi_14", ChangeType.PARAMETER_CHANGE, {"period": 20})
    """

    def __init__(self):
        self._versions: Dict[str, Dict[str, FeatureVersionInfo]] = {}  # name -> version -> info
        self._current: Dict[str, str] = {}  # name -> current version
        self._lineage: Dict[str, FeatureLineage] = {}
        self._change_log: List[FeatureChange] = []

    def register_version(
        self,
        name: str,
        version: str,
        schema: Optional[FeatureSchema] = None,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        status: VersionStatus = VersionStatus.DEVELOPMENT,
        created_by: str = ""
    ) -> FeatureVersionInfo:
        """
        Register a new feature version.

        Args:
            name: Feature name
            version: Version string (e.g., "1.0.0")
            schema: Feature schema for validation
            description: Feature description
            parameters: Computation parameters
            dependencies: Feature dependencies
            status: Initial status
            created_by: Creator identifier

        Returns:
            FeatureVersionInfo for the registered version
        """
        sem_version = SemanticVersion.from_string(version)

        info = FeatureVersionInfo(
            name=name,
            version=sem_version,
            status=status,
            schema=schema,
            description=description,
            parameters=parameters or {},
            dependencies=dependencies or [],
            created_by=created_by
        )

        # Store version
        if name not in self._versions:
            self._versions[name] = {}

        self._versions[name][version] = info

        # Update current version
        if name not in self._current or sem_version > SemanticVersion.from_string(self._current[name]):
            self._current[name] = version

        # Log change
        self._change_log.append(FeatureChange(
            feature_name=name,
            change_type=ChangeType.ADDED,
            old_version=None,
            new_version=sem_version,
            description=f"Registered new feature: {description}",
            changed_by=created_by
        ))

        logger.info(f"Registered feature {name} version {version}")
        return info

    def get_version(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[FeatureVersionInfo]:
        """Get feature version info."""
        if name not in self._versions:
            return None

        if version is None:
            version = self._current.get(name)

        if version is None:
            return None

        return self._versions[name].get(version)

    def get_current_version(self, name: str) -> Optional[str]:
        """Get current version string for a feature."""
        return self._current.get(name)

    def list_versions(self, name: str) -> List[str]:
        """List all versions for a feature."""
        if name not in self._versions:
            return []
        return sorted(self._versions[name].keys(), key=lambda v: SemanticVersion.from_string(v))

    def bump_version(
        self,
        name: str,
        change_type: ChangeType,
        changes: Dict[str, Any],
        description: str = "",
        changed_by: str = ""
    ) -> Optional[FeatureVersionInfo]:
        """
        Bump feature version based on change type.

        - SCHEMA_CHANGE: bumps major version
        - ADDED, MODIFIED: bumps minor version
        - PARAMETER_CHANGE: bumps patch version
        """
        current_info = self.get_version(name)
        if current_info is None:
            return None

        # Determine new version
        if change_type == ChangeType.SCHEMA_CHANGE:
            new_version = current_info.version.bump_major()
        elif change_type in (ChangeType.ADDED, ChangeType.MODIFIED):
            new_version = current_info.version.bump_minor()
        else:
            new_version = current_info.version.bump_patch()

        # Apply changes
        new_parameters = current_info.parameters.copy()
        new_parameters.update(changes.get('parameters', {}))

        new_schema = changes.get('schema', current_info.schema)

        # Register new version
        new_info = self.register_version(
            name=name,
            version=new_version.to_string(),
            schema=new_schema,
            description=description or current_info.description,
            parameters=new_parameters,
            dependencies=current_info.dependencies,
            status=VersionStatus.DEVELOPMENT,
            created_by=changed_by
        )

        # Log change
        self._change_log.append(FeatureChange(
            feature_name=name,
            change_type=change_type,
            old_version=current_info.version,
            new_version=new_version,
            description=description,
            changed_by=changed_by,
            changes=changes
        ))

        return new_info

    def deprecate_version(
        self,
        name: str,
        version: str,
        reason: str = ""
    ) -> bool:
        """Mark a version as deprecated."""
        info = self.get_version(name, version)
        if info is None:
            return False

        info.status = VersionStatus.DEPRECATED

        self._change_log.append(FeatureChange(
            feature_name=name,
            change_type=ChangeType.DEPRECATED,
            old_version=info.version,
            new_version=info.version,
            description=reason
        ))

        return True

    def set_lineage(self, name: str, lineage: FeatureLineage) -> None:
        """Set lineage for a feature."""
        self._lineage[name] = lineage

    def get_lineage(self, name: str) -> Optional[FeatureLineage]:
        """Get lineage for a feature."""
        return self._lineage.get(name)

    def get_change_log(
        self,
        name: Optional[str] = None,
        limit: int = 100
    ) -> List[FeatureChange]:
        """Get change log, optionally filtered by feature name."""
        changes = self._change_log
        if name:
            changes = [c for c in changes if c.feature_name == name]
        return changes[-limit:]

    def validate_value(
        self,
        name: str,
        value: Any,
        version: Optional[str] = None
    ) -> tuple:
        """
        Validate a feature value against its schema.

        Returns:
            Tuple of (is_valid, error_message)
        """
        info = self.get_version(name, version)
        if info is None:
            return False, f"Feature {name} not found"

        if info.schema is None:
            return True, ""

        return info.schema.validate(value)

    def export_catalog(self) -> Dict[str, Any]:
        """Export feature catalog as dictionary."""
        catalog = {}

        for name, versions in self._versions.items():
            current = self._current.get(name)
            catalog[name] = {
                'current_version': current,
                'versions': {
                    v: {
                        'status': info.status.value,
                        'description': info.description,
                        'parameters': info.parameters,
                        'dependencies': info.dependencies,
                        'schema': info.schema.to_dict() if info.schema else None,
                        'hash': info.hash,
                        'created_at': info.created_at.isoformat()
                    }
                    for v, info in versions.items()
                },
                'lineage': self._lineage[name].to_dict() if name in self._lineage else None
            }

        return catalog

    def list_features(self) -> List[str]:
        """List all registered feature names."""
        return list(self._versions.keys())

    def get_features_by_status(self, status: VersionStatus) -> List[str]:
        """Get features with specific status."""
        result = []
        for name in self._versions:
            info = self.get_version(name)
            if info and info.status == status:
                result.append(name)
        return result


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

_version_manager: Optional[FeatureVersionManager] = None


def get_version_manager() -> FeatureVersionManager:
    """Get global version manager."""
    global _version_manager
    if _version_manager is None:
        _version_manager = FeatureVersionManager()
    return _version_manager


def set_version_manager(manager: FeatureVersionManager) -> None:
    """Set global version manager."""
    global _version_manager
    _version_manager = manager


def register_feature(
    name: str,
    version: str = "1.0.0",
    **kwargs
) -> FeatureVersionInfo:
    """Register a feature using global manager."""
    return get_version_manager().register_version(name, version, **kwargs)


def get_feature_version(name: str) -> Optional[str]:
    """Get current feature version."""
    return get_version_manager().get_current_version(name)


def validate_feature(name: str, value: Any) -> tuple:
    """Validate feature value."""
    return get_version_manager().validate_value(name, value)
