"""
Centralized cache path configuration.

Cache paths are resolved in priority order:
1. CLI arguments (--cache-dir, --dataset-cache-dir) — handled by each script
2. Environment variables (EMBEDDING_CACHE_DIR, DATASET_CACHE_DIR)
3. .env file in the project root
4. Relative-path defaults

For shared cache usage across collaborators on the same server,
create a .env file from the template:

    cp .env.example .env
    # Edit paths if needed

The .env file is gitignored, so each collaborator can have their own.
"""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Relative-path fallback defaults ---
_DEFAULT_EMBEDDING_CACHE_DIR = "cache/embeddings/mair"
_DEFAULT_DATASET_CACHE_DIR = "cache/datasets/mair"


def _load_env_file() -> dict[str, str]:
    """Load key=value pairs from .env in the project root."""
    env_file = _PROJECT_ROOT / ".env"
    if not env_file.exists():
        return {}
    env_vars: dict[str, str] = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            env_vars[key] = value
    return env_vars


_env_cache: dict[str, str] | None = None


def _env() -> dict[str, str]:
    """Lazy-load .env file (cached after first read)."""
    global _env_cache
    if _env_cache is None:
        _env_cache = _load_env_file()
    return _env_cache


def _resolve(key: str, default: str) -> str:
    """Resolve a config value: env var > .env file > default.

    Relative paths are resolved relative to the project root so that
    users who clone the repository get a working setup without editing
    any configuration files.
    """
    raw = os.environ.get(key) or _env().get(key) or default
    path = Path(raw)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path)


def get_embedding_cache_dir() -> str:
    """Return the embedding cache directory path."""
    return _resolve("EMBEDDING_CACHE_DIR", _DEFAULT_EMBEDDING_CACHE_DIR)


def get_dataset_cache_dir() -> str:
    """Return the dataset cache directory path."""
    return _resolve("DATASET_CACHE_DIR", _DEFAULT_DATASET_CACHE_DIR)


# Module-level constants — evaluated once at import time.
# Import these directly when a simple string default is needed.
EMBEDDING_CACHE_DIR: str = get_embedding_cache_dir()
DATASET_CACHE_DIR: str = get_dataset_cache_dir()
