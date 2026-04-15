"""
Alembic migration environment — async (asyncpg) configuration.

This file is loaded by Alembic for both ``alembic upgrade`` (online) and
``alembic revision --autogenerate`` (offline) commands.

Online mode uses ``run_async_migrations()`` so that the same asyncpg driver
is used for both migration and application code — no second psycopg2 install
needed.

Usage
-----
::

    # Apply all pending migrations
    alembic upgrade head

    # Generate a new migration from ORM diff
    alembic revision --autogenerate -m "describe change"

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

# ---------------------------------------------------------------------------
# Alembic Config object (gives access to alembic.ini values)
# ---------------------------------------------------------------------------
config = context.config

# ---------------------------------------------------------------------------
# Logging — honour the [loggers] section of alembic.ini if present.
# ---------------------------------------------------------------------------
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Import ORM metadata so autogenerate can diff the schema.
# ---------------------------------------------------------------------------
# Import Base AFTER the logging setup so that any module-level log calls
# inside orm.py are captured.
from patient_advocate.db.orm import Base  # noqa: E402

target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Helper — resolve the database URL
# ---------------------------------------------------------------------------

def _get_url() -> str:
    """
    Return the database URL for migrations.

    Priority:
      1. ``sqlalchemy.url`` in alembic.ini (useful for local dev override).
      2. ``DATABASE_URL`` environment variable (CI / production).

    The URL must use an async driver (``postgresql+asyncpg://...``).
    """
    import os

    ini_url: str | None = config.get_main_option("sqlalchemy.url")
    if ini_url:
        return ini_url

    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url

    raise RuntimeError(
        "No database URL configured.  Set 'sqlalchemy.url' in alembic.ini "
        "or the DATABASE_URL environment variable."
    )


# ---------------------------------------------------------------------------
# Offline migrations (no live DB connection — emits SQL to stdout/file)
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """
    Run migrations without a live database connection.

    Alembic emits the SQL statements to the configured output (stdout or a
    file) rather than executing them.  Useful for review or DBAs who prefer
    to apply SQL manually.
    """
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migrations (async, using asyncpg)
# ---------------------------------------------------------------------------

def do_run_migrations(connection: Connection) -> None:
    """Execute migrations on a synchronous connection (called inside asyncio loop)."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations via a synchronous shim."""
    connectable = create_async_engine(
        _get_url(),
        poolclass=pool.NullPool,  # Single-use engine for migration runs.
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point called by Alembic for online migration mode."""
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
