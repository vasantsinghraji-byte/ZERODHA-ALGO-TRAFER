# -*- coding: utf-8 -*-
"""
Path Resolution Utilities
=========================
Robust project path resolution that works regardless of:
- Module location (can be moved to different subdirectory)
- Execution context (script, package, frozen executable)
- Installation method (development, pip install, PyInstaller)

BRITTLE PATH FIX: Replaces hardcoded Path(__file__).parent.parent patterns
that break when files are moved or project structure changes.
"""

import os
import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def find_project_root(start_path: Path = None) -> Path:
    """
    Find the project root directory robustly.

    Search order:
    1. PROJECT_ROOT environment variable (explicit override)
    2. Search upward for marker files (.git, pyproject.toml, requirements.txt)
    3. Fall back to educated guess based on common patterns

    Args:
        start_path: Starting path for search (defaults to this file's directory)

    Returns:
        Path to the project root directory

    Example:
        >>> from utils.paths import find_project_root, get_config_dir
        >>> root = find_project_root()
        >>> config = get_config_dir() / "settings.yaml"
    """
    # 1. Check environment variable first (explicit override)
    env_root = os.environ.get('PROJECT_ROOT')
    if env_root:
        root = Path(env_root)
        if root.exists():
            logger.debug(f"Using PROJECT_ROOT from environment: {root}")
            return root

    # 2. Search upward for project marker files
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    marker_files = ['.git', 'pyproject.toml', 'requirements.txt', 'setup.py', 'main.py']
    current = start_path

    # Search up to 10 levels (prevent infinite loop)
    for _ in range(10):
        for marker in marker_files:
            if (current / marker).exists():
                logger.debug(f"Found project root via '{marker}': {current}")
                return current
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    # 3. Fall back: assume utils is one level deep from root
    fallback = Path(__file__).resolve().parent.parent
    logger.debug(f"Project root markers not found, using fallback: {fallback}")
    return fallback


def get_config_dir() -> Path:
    """Get the config directory path."""
    return find_project_root() / "config"


def get_data_dir() -> Path:
    """Get the data directory path."""
    return find_project_root() / "data"


def get_logs_dir() -> Path:
    """Get the logs directory path."""
    logs = find_project_root() / "logs"
    logs.mkdir(exist_ok=True)
    return logs


# Convenience: pre-computed project root
PROJECT_ROOT = find_project_root()
