"""
Patient Advocate — GlassBoxLogger
===================================

Append-only audit logger for the Glass Box Imperative.

The absence of ``update()``, ``delete()``, and ``upsert()`` methods is not
an oversight — it is the enforcement mechanism.  Any attempt to add those
methods must be treated as a security/compliance regression.

Every audit event is written to ``transparency_report`` via a fresh
SQLAlchemy async session so that:

  1. Each ``write()`` call is an atomic, durable INSERT.
  2. ``write_many()`` wraps all INSERTs in a single ``session.begin()``
     transaction — either every record lands or none do.
  3. All failures surface as ``AuditWriteError`` (never swallowed silently).

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

import logging
from typing import Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_advocate.core.exceptions import AuditWriteError
from patient_advocate.core.models import TransparencyRecord
from patient_advocate.db.orm import TransparencyReportRow

logger = logging.getLogger(__name__)


class GlassBoxLogger:
    """
    Append-only audit logger.

    **No** ``update()``, ``delete()``, or ``upsert()`` methods exist.
    Their absence IS the enforcement — any code path that bypasses this
    class to mutate audit rows is out-of-compliance with the Glass Box
    Imperative.

    Parameters
    ----------
    session_factory:
        An ``async_sessionmaker`` bound to the application's async engine.
        Obtain one via :func:`~patient_advocate.db.engine.get_async_session_factory`.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    # ------------------------------------------------------------------
    # Public API — INSERT only
    # ------------------------------------------------------------------

    async def write(self, record: TransparencyRecord) -> None:
        """
        Persist a single ``TransparencyRecord`` as an INSERT.

        Parameters
        ----------
        record:
            A frozen ``TransparencyRecord`` produced by any agent.

        Raises
        ------
        AuditWriteError
            If the database INSERT fails for any reason.  The original
            exception is chained via ``__cause__`` for debugging.
            ``agent_name`` is embedded in the error for audit attribution.
        """
        row = TransparencyReportRow.from_domain(record)
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    session.add(row)
        except Exception as exc:
            msg = (
                f"GlassBoxLogger failed to write audit record for agent "
                f"{record.agent_name!r} (event_id={record.event_id}): {exc}"
            )
            logger.error(msg)
            raise AuditWriteError(msg, agent_name=record.agent_name) from exc

    async def write_many(self, records: Sequence[TransparencyRecord]) -> None:
        """
        Persist multiple ``TransparencyRecord`` objects in one transaction.

        All INSERTs share a single ``session.begin()`` context — either
        every record is committed or the entire batch is rolled back.

        Parameters
        ----------
        records:
            An ordered sequence of frozen ``TransparencyRecord`` objects.
            Empty sequences are accepted (no-op).

        Raises
        ------
        AuditWriteError
            If any INSERT fails.  The transaction is rolled back before the
            exception is raised, guaranteeing atomicity.
        """
        if not records:
            return

        rows = [TransparencyReportRow.from_domain(r) for r in records]
        agent_names = ", ".join(r.agent_name for r in records)

        try:
            async with self._session_factory() as session:
                async with session.begin():
                    session.add_all(rows)
        except Exception as exc:
            # Use the first record's agent_name for error attribution.
            first_agent = records[0].agent_name
            msg = (
                f"GlassBoxLogger failed to write batch of {len(rows)} audit "
                f"records (agents: {agent_names}): {exc}"
            )
            logger.error(msg)
            raise AuditWriteError(msg, agent_name=first_agent) from exc

    # ------------------------------------------------------------------
    # Intentionally absent
    # ------------------------------------------------------------------
    # update()  — not implemented; Glass Box Imperative
    # delete()  — not implemented; Glass Box Imperative
    # upsert()  — not implemented; Glass Box Imperative
    # ------------------------------------------------------------------
