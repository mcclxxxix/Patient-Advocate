"""
Patient Advocate — SQLAlchemy 2.0 ORM Model
=============================================

Maps the frozen Pydantic ``TransparencyRecord`` domain object to the
``transparency_report`` relational table.  The mapping is intentionally
explicit (no **kwargs unpacking) so that schema drift is caught at import
time rather than silently at write time.

Design notes
------------
* ``created_at`` uses a server-side default so the DB clock is authoritative.
* Naive ``datetime`` objects arriving from ``TransparencyRecord.timestamp`` are
  converted to UTC-aware before persistence to avoid ambiguous timezone data.
* Enum values are stored as their string ``.value`` (e.g. ``"nadir_window"``)
  so the table stays readable without ORM tooling.

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import DateTime, Float, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from patient_advocate.core.models import TransparencyRecord


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Project-wide SQLAlchemy declarative base."""
    pass


# ---------------------------------------------------------------------------
# ORM Row
# ---------------------------------------------------------------------------

class TransparencyReportRow(Base):
    """
    SQLAlchemy ORM representation of the ``transparency_report`` table.

    Each row corresponds to one ``TransparencyRecord`` emitted by an agent.
    The table is append-only by convention; no UPDATE or DELETE operations
    are issued by the application layer (enforced via ``GlassBoxLogger``).
    """

    __tablename__ = "transparency_report"

    # ---- Primary key -------------------------------------------------------
    event_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        nullable=False,
    )

    # ---- Foreign-key-adjacent reference (nullable) -------------------------
    suggestion_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
        index=False,  # covered by dedicated index below
    )

    # ---- Agent identity ----------------------------------------------------
    agent_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    # ---- Clinical conflict --------------------------------------------------
    conflict_type: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    # ---- Confidence bound --------------------------------------------------
    confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    # ---- Scholarly provenance ----------------------------------------------
    scholarly_basis: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ---- HITL outcome -------------------------------------------------------
    patient_decision: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
    )

    # ---- Full reasoning chain (the Glass Box audit payload) ----------------
    reasoning_chain: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # ---- Wall-clock timestamp (server default; application also supplies it) -
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # ---- Indexes -----------------------------------------------------------
    __table_args__ = (
        Index("ix_transparency_report_agent_name", "agent_name"),
        Index("ix_transparency_report_suggestion_id", "suggestion_id"),
        Index("ix_transparency_report_created_at", "created_at"),
        Index(
            "ix_transparency_report_agent_created",
            "agent_name",
            "created_at",
        ),
    )

    # -----------------------------------------------------------------------
    # Factory method
    # -----------------------------------------------------------------------

    @classmethod
    def from_domain(cls, record: TransparencyRecord) -> "TransparencyReportRow":
        """
        Convert a frozen ``TransparencyRecord`` to an ORM row.

        All field mapping is explicit — no ``**kwargs`` — so that future
        schema changes produce an ``AttributeError`` rather than silent
        data loss.

        Naive ``datetime`` objects are treated as UTC and converted to an
        aware ``datetime`` before storage.
        """
        # Make timestamp timezone-aware if it is naive (has no tzinfo).
        ts = record.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        return cls(
            event_id=record.event_id,
            suggestion_id=record.suggestion_id,
            agent_name=record.agent_name,
            # Enum: store the string .value; None stays None.
            conflict_type=(
                record.conflict_type.value
                if record.conflict_type is not None
                else None
            ),
            confidence=record.confidence,
            scholarly_basis=record.scholarly_basis,
            patient_decision=(
                record.patient_decision.value
                if record.patient_decision is not None
                else None
            ),
            reasoning_chain=record.reasoning_chain,
            # ``created_at`` is driven by the server default; supplying the
            # application timestamp here overrides it so tests can inspect it.
            created_at=ts,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<TransparencyReportRow event_id={self.event_id!s} "
            f"agent={self.agent_name!r} created_at={self.created_at!s}>"
        )
