# -*- coding: utf-8 -*-
"""
Model Registry & Deployment - ML Model Lifecycle Management
============================================================
Production-grade model registry with versioning, A/B testing,
automated deployment, and rollback capabilities.

Features:
- Model versioning with semantic versioning
- A/B testing framework for model comparison
- Staged deployment pipeline (dev -> staging -> production)
- Automatic rollback on performance degradation

Example:
    >>> from ml.model_registry import ModelRegistry, ABTestManager
    >>>
    >>> # Register a model
    >>> registry = ModelRegistry()
    >>> version = registry.register_model(
    ...     model=trained_model,
    ...     name="price_predictor",
    ...     metrics={'accuracy': 0.85, 'sharpe': 1.5}
    ... )
    >>>
    >>> # A/B test models
    >>> ab_test = ABTestManager()
    >>> ab_test.create_test("price_predictor", "v1.0.0", "v1.1.0")
    >>> winner = ab_test.get_winner()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Tuple, Union, Generic, TypeVar
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import json
import pickle
import threading
import logging
import random
import os

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Generic model type


class ModelStage(Enum):
    """Model deployment stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ModelStatus(Enum):
    """Model status in registry."""
    REGISTERED = "registered"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DeploymentAction(Enum):
    """Deployment pipeline actions."""
    PROMOTE = "promote"
    DEMOTE = "demote"
    ROLLBACK = "rollback"
    ARCHIVE = "archive"
    DEPRECATE = "deprecate"


class ABTestStatus(Enum):
    """A/B test status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class SemanticVersion:
    """Semantic version (major.minor.patch)."""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    def __str__(self) -> str:
        version = f"v{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: 'SemanticVersion') -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))

    @classmethod
    def parse(cls, version_str: str) -> 'SemanticVersion':
        """Parse version string like 'v1.2.3' or '1.2.3-beta+build123'."""
        v = version_str.lstrip('v')

        # Handle build metadata
        build = None
        if '+' in v:
            v, build = v.split('+', 1)

        # Handle prerelease
        prerelease = None
        if '-' in v:
            v, prerelease = v.split('-', 1)

        parts = v.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")

        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
            prerelease=prerelease,
            build=build
        )

    def bump_major(self) -> 'SemanticVersion':
        """Return new version with bumped major."""
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> 'SemanticVersion':
        """Return new version with bumped minor."""
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> 'SemanticVersion':
        """Return new version with bumped patch."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    name: str
    version: SemanticVersion
    created_at: datetime
    updated_at: datetime
    stage: ModelStage
    status: ModelStatus
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    tags: Dict[str, str]
    description: str
    model_type: str
    framework: str
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    artifact_path: Optional[str] = None
    model_hash: Optional[str] = None
    parent_version: Optional[str] = None
    training_data_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'version': str(self.version),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'stage': self.stage.value,
            'status': self.status.value,
            'metrics': self.metrics,
            'parameters': self.parameters,
            'tags': self.tags,
            'description': self.description,
            'model_type': self.model_type,
            'framework': self.framework,
            'input_schema': self.input_schema,
            'output_schema': self.output_schema,
            'artifact_path': self.artifact_path,
            'model_hash': self.model_hash,
            'parent_version': self.parent_version,
            'training_data_hash': self.training_data_hash
        }


@dataclass
class ModelVersion:
    """A specific version of a model."""
    metadata: ModelMetadata
    model: Any  # The actual model object
    validators: List[Callable[[Any], bool]] = field(default_factory=list)

    def predict(self, X: Any) -> Any:
        """Make prediction using the model."""
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        elif callable(self.model):
            return self.model(X)
        else:
            raise ValueError("Model does not support prediction")

    def validate(self) -> Tuple[bool, List[str]]:
        """Run all validators on the model."""
        errors = []
        for validator in self.validators:
            try:
                if not validator(self.model):
                    errors.append(f"Validator {validator.__name__} failed")
            except Exception as e:
                errors.append(f"Validator {validator.__name__} error: {e}")

        return len(errors) == 0, errors


@dataclass
class DeploymentRecord:
    """Record of a deployment action."""
    record_id: str
    model_name: str
    version: str
    action: DeploymentAction
    from_stage: ModelStage
    to_stage: ModelStage
    timestamp: datetime
    performed_by: str
    reason: str
    success: bool
    error_message: Optional[str] = None
    rollback_version: Optional[str] = None


@dataclass
class ABTest:
    """A/B test configuration and results."""
    test_id: str
    model_name: str
    control_version: str
    treatment_version: str
    traffic_split: float  # Fraction of traffic to treatment (0.0-1.0)
    start_time: datetime
    end_time: Optional[datetime]
    status: ABTestStatus
    min_samples: int
    confidence_level: float
    primary_metric: str
    metrics_control: Dict[str, List[float]] = field(default_factory=dict)
    metrics_treatment: Dict[str, List[float]] = field(default_factory=dict)
    sample_count_control: int = 0
    sample_count_treatment: int = 0
    winner: Optional[str] = None
    p_value: Optional[float] = None

    def get_variant(self) -> str:
        """Get which variant to use for next request."""
        if random.random() < self.traffic_split:
            return self.treatment_version
        return self.control_version


class ModelStorage(ABC):
    """Abstract base for model storage backends."""

    @abstractmethod
    def save_model(self, model: Any, path: str) -> str:
        """Save model to storage."""
        pass

    @abstractmethod
    def load_model(self, path: str) -> Any:
        """Load model from storage."""
        pass

    @abstractmethod
    def delete_model(self, path: str) -> bool:
        """Delete model from storage."""
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if model exists."""
        pass


class FileSystemStorage(ModelStorage):
    """File system based model storage."""

    def __init__(self, base_path: str = "./models"):
        """
        Initialize file storage.

        Args:
            base_path: Base directory for model storage
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save_model(self, model: Any, path: str) -> str:
        """Save model using pickle."""
        full_path = os.path.join(self.base_path, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, 'wb') as f:
            pickle.dump(model, f)

        return full_path

    def load_model(self, path: str) -> Any:
        """Load model from pickle file."""
        full_path = os.path.join(self.base_path, path)

        with open(full_path, 'rb') as f:
            return pickle.load(f)

    def delete_model(self, path: str) -> bool:
        """Delete model file."""
        full_path = os.path.join(self.base_path, path)

        if os.path.exists(full_path):
            os.remove(full_path)
            return True
        return False

    def exists(self, path: str) -> bool:
        """Check if model file exists."""
        full_path = os.path.join(self.base_path, path)
        return os.path.exists(full_path)


class InMemoryStorage(ModelStorage):
    """In-memory model storage for testing."""

    def __init__(self):
        self._storage: Dict[str, Any] = {}

    def save_model(self, model: Any, path: str) -> str:
        """Store model in memory."""
        self._storage[path] = model
        return path

    def load_model(self, path: str) -> Any:
        """Load model from memory."""
        if path not in self._storage:
            raise FileNotFoundError(f"Model not found: {path}")
        return self._storage[path]

    def delete_model(self, path: str) -> bool:
        """Remove model from memory."""
        if path in self._storage:
            del self._storage[path]
            return True
        return False

    def exists(self, path: str) -> bool:
        """Check if model exists in memory."""
        return path in self._storage


class ModelRegistry:
    """
    Central model registry for versioning and management.

    Provides MLflow-like functionality for model lifecycle management.
    """

    def __init__(
        self,
        storage: Optional[ModelStorage] = None,
        metadata_path: str = "./model_registry"
    ):
        """
        Initialize model registry.

        Args:
            storage: Model storage backend
            metadata_path: Path for metadata storage
        """
        self.storage = storage or InMemoryStorage()
        self.metadata_path = metadata_path

        self._models: Dict[str, Dict[str, ModelVersion]] = defaultdict(dict)
        self._deployment_history: List[DeploymentRecord] = []
        self._validators: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

        os.makedirs(metadata_path, exist_ok=True)

    def register_model(
        self,
        model: Any,
        name: str,
        metrics: Dict[str, float],
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
        model_type: str = "unknown",
        framework: str = "custom",
        version: Optional[str] = None,
        parent_version: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model: The model object to register
            name: Model name
            metrics: Performance metrics
            parameters: Model parameters/hyperparameters
            tags: Custom tags for organization
            description: Model description
            model_type: Type of model (classifier, regressor, etc.)
            framework: ML framework used
            version: Explicit version (auto-incremented if None)
            parent_version: Version this was derived from
            input_schema: Input data schema
            output_schema: Output data schema

        Returns:
            The registered ModelVersion
        """
        with self._lock:
            # Determine version
            if version:
                sem_version = SemanticVersion.parse(version)
            else:
                sem_version = self._get_next_version(name)

            version_str = str(sem_version)

            # Check for duplicate
            if version_str in self._models[name]:
                raise ValueError(f"Version {version_str} already exists for {name}")

            # Calculate model hash
            model_hash = self._compute_model_hash(model)

            # Create artifact path
            artifact_path = f"{name}/{version_str}/model.pkl"

            # Save model
            self.storage.save_model(model, artifact_path)

            # Create metadata
            now = datetime.now()
            metadata = ModelMetadata(
                name=name,
                version=sem_version,
                created_at=now,
                updated_at=now,
                stage=ModelStage.DEVELOPMENT,
                status=ModelStatus.REGISTERED,
                metrics=metrics,
                parameters=parameters or {},
                tags=tags or {},
                description=description,
                model_type=model_type,
                framework=framework,
                input_schema=input_schema,
                output_schema=output_schema,
                artifact_path=artifact_path,
                model_hash=model_hash,
                parent_version=parent_version
            )

            # Create model version
            model_version = ModelVersion(
                metadata=metadata,
                model=model,
                validators=list(self._validators.get(name, []))
            )

            # Store
            self._models[name][version_str] = model_version

            # Save metadata
            self._save_metadata(name, version_str, metadata)

            logger.info(f"Registered model {name} version {version_str}")

            return model_version

    def get_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Optional[ModelVersion]:
        """
        Get a model by name and version/stage.

        Args:
            name: Model name
            version: Specific version (e.g., "v1.2.3")
            stage: Get latest model in this stage

        Returns:
            ModelVersion or None if not found
        """
        with self._lock:
            if name not in self._models:
                return None

            if version:
                return self._models[name].get(version)

            if stage:
                # Find latest in stage
                candidates = [
                    mv for mv in self._models[name].values()
                    if mv.metadata.stage == stage
                ]
                if candidates:
                    return max(candidates, key=lambda mv: mv.metadata.version)

            # Return latest version
            if self._models[name]:
                return max(
                    self._models[name].values(),
                    key=lambda mv: mv.metadata.version
                )

            return None

    def get_production_model(self, name: str) -> Optional[ModelVersion]:
        """Get the current production model."""
        return self.get_model(name, stage=ModelStage.PRODUCTION)

    def list_models(self, name: Optional[str] = None) -> Dict[str, List[str]]:
        """List all models and their versions."""
        with self._lock:
            if name:
                if name in self._models:
                    return {name: list(self._models[name].keys())}
                return {}
            return {n: list(versions.keys()) for n, versions in self._models.items()}

    def list_versions(
        self,
        name: str,
        stage: Optional[ModelStage] = None
    ) -> List[ModelVersion]:
        """List all versions of a model."""
        with self._lock:
            if name not in self._models:
                return []

            versions = list(self._models[name].values())

            if stage:
                versions = [v for v in versions if v.metadata.stage == stage]

            return sorted(versions, key=lambda v: v.metadata.version, reverse=True)

    def update_metrics(
        self,
        name: str,
        version: str,
        metrics: Dict[str, float]
    ) -> bool:
        """Update metrics for a model version."""
        with self._lock:
            mv = self._models.get(name, {}).get(version)
            if not mv:
                return False

            mv.metadata.metrics.update(metrics)
            mv.metadata.updated_at = datetime.now()
            self._save_metadata(name, version, mv.metadata)

            return True

    def add_validator(
        self,
        name: str,
        validator: Callable[[Any], bool]
    ) -> None:
        """Add a validator for a model."""
        self._validators[name].append(validator)

    def validate_model(
        self,
        name: str,
        version: str
    ) -> Tuple[bool, List[str]]:
        """Validate a model version."""
        mv = self.get_model(name, version)
        if not mv:
            return False, [f"Model {name} version {version} not found"]

        mv.metadata.status = ModelStatus.VALIDATING
        is_valid, errors = mv.validate()

        if is_valid:
            mv.metadata.status = ModelStatus.VALIDATED
        else:
            mv.metadata.status = ModelStatus.FAILED

        return is_valid, errors

    def delete_model(self, name: str, version: str) -> bool:
        """Delete a model version."""
        with self._lock:
            if name not in self._models or version not in self._models[name]:
                return False

            mv = self._models[name][version]

            # Don't allow deleting production models
            if mv.metadata.stage == ModelStage.PRODUCTION:
                raise ValueError("Cannot delete production model. Demote first.")

            # Delete artifact
            if mv.metadata.artifact_path:
                self.storage.delete_model(mv.metadata.artifact_path)

            # Remove from registry
            del self._models[name][version]

            logger.info(f"Deleted model {name} version {version}")
            return True

    def _get_next_version(self, name: str) -> SemanticVersion:
        """Get next version number for a model."""
        if name not in self._models or not self._models[name]:
            return SemanticVersion(1, 0, 0)

        latest = max(
            self._models[name].values(),
            key=lambda mv: mv.metadata.version
        )
        return latest.metadata.version.bump_minor()

    def _compute_model_hash(self, model: Any) -> str:
        """Compute hash of model for integrity checking."""
        try:
            serialized = pickle.dumps(model)
            return hashlib.sha256(serialized).hexdigest()[:16]
        except Exception:
            return "unhashable"

    def _save_metadata(
        self,
        name: str,
        version: str,
        metadata: ModelMetadata
    ) -> None:
        """Save metadata to disk."""
        path = os.path.join(self.metadata_path, name, f"{version}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)


class DeploymentPipeline:
    """
    Automated deployment pipeline with staged promotion.

    Manages model promotion through stages with validation gates.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        validation_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Initialize deployment pipeline.

        Args:
            registry: Model registry instance
            validation_metrics: Minimum metrics required for promotion
        """
        self.registry = registry
        self.validation_metrics = validation_metrics or {
            'accuracy': 0.6,
            'sharpe_ratio': 0.5
        }

        self._deployment_history: List[DeploymentRecord] = []
        self._promotion_callbacks: Dict[ModelStage, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self._record_counter = 0

    def promote(
        self,
        name: str,
        version: str,
        to_stage: ModelStage,
        performed_by: str = "system",
        reason: str = "",
        skip_validation: bool = False
    ) -> DeploymentRecord:
        """
        Promote a model to a new stage.

        Args:
            name: Model name
            version: Model version
            to_stage: Target stage
            performed_by: Who initiated the promotion
            reason: Reason for promotion
            skip_validation: Skip metric validation

        Returns:
            Deployment record
        """
        mv = self.registry.get_model(name, version)
        if not mv:
            raise ValueError(f"Model {name} version {version} not found")

        from_stage = mv.metadata.stage

        # Validate promotion path
        valid_promotions = {
            ModelStage.DEVELOPMENT: [ModelStage.STAGING, ModelStage.ARCHIVED],
            ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.DEVELOPMENT, ModelStage.ARCHIVED],
            ModelStage.PRODUCTION: [ModelStage.ARCHIVED, ModelStage.DEPRECATED],
            ModelStage.ARCHIVED: [ModelStage.DEVELOPMENT],
            ModelStage.DEPRECATED: []
        }

        if to_stage not in valid_promotions.get(from_stage, []):
            raise ValueError(
                f"Invalid promotion: {from_stage.value} -> {to_stage.value}"
            )

        # Validate metrics for production promotion
        success = True
        error_message = None

        if to_stage == ModelStage.PRODUCTION and not skip_validation:
            is_valid, errors = self._validate_for_production(mv)
            if not is_valid:
                success = False
                error_message = "; ".join(errors)

        # Demote current production model if promoting to production
        if success and to_stage == ModelStage.PRODUCTION:
            current_prod = self.registry.get_production_model(name)
            if current_prod and current_prod.metadata.version != mv.metadata.version:
                current_prod.metadata.stage = ModelStage.STAGING
                current_prod.metadata.updated_at = datetime.now()
                logger.info(
                    f"Demoted {name} {current_prod.metadata.version} "
                    f"from production to staging"
                )

        # Update stage
        if success:
            mv.metadata.stage = to_stage
            mv.metadata.status = ModelStatus.DEPLOYED
            mv.metadata.updated_at = datetime.now()

            # Run callbacks
            for callback in self._promotion_callbacks.get(to_stage, []):
                try:
                    callback(mv)
                except Exception as e:
                    logger.error(f"Promotion callback failed: {e}")

        # Record deployment
        record = self._create_record(
            name=name,
            version=version,
            action=DeploymentAction.PROMOTE,
            from_stage=from_stage,
            to_stage=to_stage,
            performed_by=performed_by,
            reason=reason,
            success=success,
            error_message=error_message
        )

        if success:
            logger.info(
                f"Promoted {name} {version} from {from_stage.value} "
                f"to {to_stage.value}"
            )
        else:
            logger.warning(
                f"Failed to promote {name} {version}: {error_message}"
            )

        return record

    def rollback(
        self,
        name: str,
        to_version: Optional[str] = None,
        performed_by: str = "system",
        reason: str = "Performance degradation"
    ) -> DeploymentRecord:
        """
        Rollback production model to a previous version.

        Args:
            name: Model name
            to_version: Version to rollback to (previous production if None)
            performed_by: Who initiated rollback
            reason: Reason for rollback

        Returns:
            Deployment record
        """
        current_prod = self.registry.get_production_model(name)
        if not current_prod:
            raise ValueError(f"No production model found for {name}")

        # Find rollback target
        if to_version:
            target = self.registry.get_model(name, to_version)
            if not target:
                raise ValueError(f"Rollback target {to_version} not found")
        else:
            # Find previous production version from history
            target = self._find_previous_production(name)
            if not target:
                raise ValueError("No previous production version found")

        # Demote current production
        current_prod.metadata.stage = ModelStage.ARCHIVED
        current_prod.metadata.status = ModelStatus.ROLLED_BACK
        current_prod.metadata.updated_at = datetime.now()

        # Promote target to production
        target.metadata.stage = ModelStage.PRODUCTION
        target.metadata.status = ModelStatus.DEPLOYED
        target.metadata.updated_at = datetime.now()

        # Record rollback
        record = self._create_record(
            name=name,
            version=str(current_prod.metadata.version),
            action=DeploymentAction.ROLLBACK,
            from_stage=ModelStage.PRODUCTION,
            to_stage=ModelStage.ARCHIVED,
            performed_by=performed_by,
            reason=reason,
            success=True,
            rollback_version=str(target.metadata.version)
        )

        logger.warning(
            f"Rolled back {name} from {current_prod.metadata.version} "
            f"to {target.metadata.version}: {reason}"
        )

        return record

    def register_promotion_callback(
        self,
        stage: ModelStage,
        callback: Callable[[ModelVersion], None]
    ) -> None:
        """Register callback for stage promotions."""
        self._promotion_callbacks[stage].append(callback)

    def get_deployment_history(
        self,
        name: Optional[str] = None,
        limit: int = 100
    ) -> List[DeploymentRecord]:
        """Get deployment history."""
        with self._lock:
            history = list(self._deployment_history)

        if name:
            history = [r for r in history if r.model_name == name]

        return sorted(history, key=lambda r: r.timestamp, reverse=True)[:limit]

    def _validate_for_production(
        self,
        mv: ModelVersion
    ) -> Tuple[bool, List[str]]:
        """Validate model meets production requirements."""
        errors = []

        for metric, threshold in self.validation_metrics.items():
            if metric not in mv.metadata.metrics:
                errors.append(f"Missing required metric: {metric}")
            elif mv.metadata.metrics[metric] < threshold:
                errors.append(
                    f"{metric} ({mv.metadata.metrics[metric]:.3f}) "
                    f"below threshold ({threshold})"
                )

        return len(errors) == 0, errors

    def _find_previous_production(self, name: str) -> Optional[ModelVersion]:
        """Find the previous production model from history."""
        # Look for last successful production deployment
        for record in reversed(self._deployment_history):
            if (record.model_name == name and
                record.to_stage == ModelStage.PRODUCTION and
                record.success and
                record.version != self.registry.get_production_model(name).metadata.version):
                return self.registry.get_model(name, record.version)

        # Fall back to staging
        return self.registry.get_model(name, stage=ModelStage.STAGING)

    def _create_record(
        self,
        name: str,
        version: str,
        action: DeploymentAction,
        from_stage: ModelStage,
        to_stage: ModelStage,
        performed_by: str,
        reason: str,
        success: bool,
        error_message: Optional[str] = None,
        rollback_version: Optional[str] = None
    ) -> DeploymentRecord:
        """Create and store deployment record."""
        with self._lock:
            self._record_counter += 1
            record = DeploymentRecord(
                record_id=f"deploy_{self._record_counter}",
                model_name=name,
                version=version,
                action=action,
                from_stage=from_stage,
                to_stage=to_stage,
                timestamp=datetime.now(),
                performed_by=performed_by,
                reason=reason,
                success=success,
                error_message=error_message,
                rollback_version=rollback_version
            )
            self._deployment_history.append(record)

        return record


class ABTestManager:
    """
    A/B testing framework for model comparison.

    Supports statistical significance testing to determine
    which model variant performs better.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        min_samples: int = 1000,
        confidence_level: float = 0.95
    ):
        """
        Initialize A/B test manager.

        Args:
            registry: Model registry
            min_samples: Minimum samples per variant
            confidence_level: Required confidence for winner declaration
        """
        self.registry = registry
        self.min_samples = min_samples
        self.confidence_level = confidence_level

        self._tests: Dict[str, ABTest] = {}
        self._active_tests: Dict[str, str] = {}  # model_name -> test_id
        self._lock = threading.Lock()
        self._test_counter = 0

    def create_test(
        self,
        model_name: str,
        control_version: str,
        treatment_version: str,
        traffic_split: float = 0.5,
        primary_metric: str = "accuracy",
        duration_hours: Optional[int] = None
    ) -> ABTest:
        """
        Create a new A/B test.

        Args:
            model_name: Model being tested
            control_version: Current/baseline version
            treatment_version: New version to test
            traffic_split: Fraction of traffic to treatment
            primary_metric: Main metric for comparison
            duration_hours: Optional test duration limit

        Returns:
            Created A/B test
        """
        # Validate models exist
        control = self.registry.get_model(model_name, control_version)
        treatment = self.registry.get_model(model_name, treatment_version)

        if not control:
            raise ValueError(f"Control model {control_version} not found")
        if not treatment:
            raise ValueError(f"Treatment model {treatment_version} not found")

        # Check for existing test
        if model_name in self._active_tests:
            raise ValueError(
                f"Active test already exists for {model_name}: "
                f"{self._active_tests[model_name]}"
            )

        with self._lock:
            self._test_counter += 1
            test_id = f"ab_test_{self._test_counter}"

            end_time = None
            if duration_hours:
                end_time = datetime.now() + timedelta(hours=duration_hours)

            test = ABTest(
                test_id=test_id,
                model_name=model_name,
                control_version=control_version,
                treatment_version=treatment_version,
                traffic_split=traffic_split,
                start_time=datetime.now(),
                end_time=end_time,
                status=ABTestStatus.RUNNING,
                min_samples=self.min_samples,
                confidence_level=self.confidence_level,
                primary_metric=primary_metric
            )

            self._tests[test_id] = test
            self._active_tests[model_name] = test_id

        logger.info(
            f"Started A/B test {test_id}: {control_version} vs {treatment_version} "
            f"({traffic_split:.0%} treatment traffic)"
        )

        return test

    def get_variant(self, model_name: str) -> Tuple[str, ModelVersion]:
        """
        Get which model variant to use for a request.

        Args:
            model_name: Model name

        Returns:
            Tuple of (version string, ModelVersion)
        """
        test_id = self._active_tests.get(model_name)

        if test_id:
            test = self._tests[test_id]
            if test.status == ABTestStatus.RUNNING:
                version = test.get_variant()
                mv = self.registry.get_model(model_name, version)
                return version, mv

        # No active test, return production
        mv = self.registry.get_production_model(model_name)
        if mv:
            return str(mv.metadata.version), mv

        raise ValueError(f"No model available for {model_name}")

    def record_result(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Record result for a model variant.

        Args:
            model_name: Model name
            version: Version that made prediction
            metrics: Observed metrics
        """
        test_id = self._active_tests.get(model_name)
        if not test_id:
            return

        test = self._tests[test_id]
        if test.status != ABTestStatus.RUNNING:
            return

        with self._lock:
            if version == test.control_version:
                test.sample_count_control += 1
                for metric, value in metrics.items():
                    if metric not in test.metrics_control:
                        test.metrics_control[metric] = []
                    test.metrics_control[metric].append(value)
            elif version == test.treatment_version:
                test.sample_count_treatment += 1
                for metric, value in metrics.items():
                    if metric not in test.metrics_treatment:
                        test.metrics_treatment[metric] = []
                    test.metrics_treatment[metric].append(value)

        # Check for completion
        self._check_test_completion(test)

    def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an A/B test."""
        test = self._tests.get(test_id)
        if not test:
            return None

        # Calculate current statistics
        control_mean = None
        treatment_mean = None
        p_value = None

        if test.primary_metric in test.metrics_control:
            control_values = test.metrics_control[test.primary_metric]
            treatment_values = test.metrics_treatment.get(test.primary_metric, [])

            if control_values:
                control_mean = np.mean(control_values) if NUMPY_AVAILABLE else sum(control_values) / len(control_values)
            if treatment_values:
                treatment_mean = np.mean(treatment_values) if NUMPY_AVAILABLE else sum(treatment_values) / len(treatment_values)

            # Statistical test
            if SCIPY_AVAILABLE and len(control_values) >= 30 and len(treatment_values) >= 30:
                _, p_value = stats.ttest_ind(control_values, treatment_values)

        return {
            'test_id': test.test_id,
            'model_name': test.model_name,
            'status': test.status.value,
            'control_version': test.control_version,
            'treatment_version': test.treatment_version,
            'traffic_split': test.traffic_split,
            'sample_count_control': test.sample_count_control,
            'sample_count_treatment': test.sample_count_treatment,
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'p_value': p_value,
            'winner': test.winner,
            'start_time': test.start_time.isoformat(),
            'end_time': test.end_time.isoformat() if test.end_time else None
        }

    def get_winner(self, test_id: str) -> Optional[str]:
        """Get the winning variant if test is complete."""
        test = self._tests.get(test_id)
        if test and test.status == ABTestStatus.COMPLETED:
            return test.winner
        return None

    def stop_test(
        self,
        test_id: str,
        winner: Optional[str] = None
    ) -> ABTest:
        """
        Stop an A/B test.

        Args:
            test_id: Test to stop
            winner: Manually declare winner (auto-determined if None)

        Returns:
            Completed test
        """
        test = self._tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")

        with self._lock:
            if winner:
                test.winner = winner
            else:
                test.winner = self._determine_winner(test)

            test.status = ABTestStatus.COMPLETED
            test.end_time = datetime.now()

            # Remove from active tests
            if test.model_name in self._active_tests:
                del self._active_tests[test.model_name]

        logger.info(
            f"Completed A/B test {test_id}. Winner: {test.winner}"
        )

        return test

    def cancel_test(self, test_id: str) -> None:
        """Cancel an A/B test without declaring winner."""
        test = self._tests.get(test_id)
        if not test:
            return

        with self._lock:
            test.status = ABTestStatus.CANCELLED
            test.end_time = datetime.now()

            if test.model_name in self._active_tests:
                del self._active_tests[test.model_name]

        logger.info(f"Cancelled A/B test {test_id}")

    def _check_test_completion(self, test: ABTest) -> None:
        """Check if test has reached completion criteria."""
        # Check minimum samples
        if (test.sample_count_control < test.min_samples or
            test.sample_count_treatment < test.min_samples):
            return

        # Check time limit
        if test.end_time and datetime.now() >= test.end_time:
            self.stop_test(test.test_id)
            return

        # Check for statistical significance
        if SCIPY_AVAILABLE and test.primary_metric in test.metrics_control:
            control = test.metrics_control[test.primary_metric]
            treatment = test.metrics_treatment.get(test.primary_metric, [])

            if len(control) >= 30 and len(treatment) >= 30:
                _, p_value = stats.ttest_ind(control, treatment)
                test.p_value = p_value

                if p_value < (1 - test.confidence_level):
                    self.stop_test(test.test_id)

    def _determine_winner(self, test: ABTest) -> str:
        """Determine winner based on metrics."""
        if test.primary_metric not in test.metrics_control:
            return test.control_version  # Default to control

        control_values = test.metrics_control[test.primary_metric]
        treatment_values = test.metrics_treatment.get(test.primary_metric, [])

        if not control_values or not treatment_values:
            return test.control_version

        if NUMPY_AVAILABLE:
            control_mean = np.mean(control_values)
            treatment_mean = np.mean(treatment_values)
        else:
            control_mean = sum(control_values) / len(control_values)
            treatment_mean = sum(treatment_values) / len(treatment_values)

        # Higher is better assumption
        if treatment_mean > control_mean:
            return test.treatment_version
        return test.control_version


# Convenience functions
_default_registry: Optional[ModelRegistry] = None
_default_pipeline: Optional[DeploymentPipeline] = None
_default_ab_manager: Optional[ABTestManager] = None


def get_registry() -> ModelRegistry:
    """Get default model registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry()
    return _default_registry


def set_registry(registry: ModelRegistry) -> None:
    """Set default model registry."""
    global _default_registry
    _default_registry = registry


def get_deployment_pipeline() -> DeploymentPipeline:
    """Get default deployment pipeline."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = DeploymentPipeline(get_registry())
    return _default_pipeline


def get_ab_manager() -> ABTestManager:
    """Get default A/B test manager."""
    global _default_ab_manager
    if _default_ab_manager is None:
        _default_ab_manager = ABTestManager(get_registry())
    return _default_ab_manager


def register_model(
    model: Any,
    name: str,
    metrics: Dict[str, float],
    **kwargs
) -> ModelVersion:
    """Register a model using the default registry."""
    return get_registry().register_model(model, name, metrics, **kwargs)


def deploy_to_production(
    name: str,
    version: str,
    reason: str = "Manual promotion"
) -> DeploymentRecord:
    """Deploy a model to production."""
    pipeline = get_deployment_pipeline()

    # First promote to staging if needed
    mv = get_registry().get_model(name, version)
    if mv and mv.metadata.stage == ModelStage.DEVELOPMENT:
        pipeline.promote(name, version, ModelStage.STAGING, reason="Staging validation")

    return pipeline.promote(name, version, ModelStage.PRODUCTION, reason=reason)


def rollback_production(name: str, reason: str = "Performance issue") -> DeploymentRecord:
    """Rollback production model."""
    return get_deployment_pipeline().rollback(name, reason=reason)
