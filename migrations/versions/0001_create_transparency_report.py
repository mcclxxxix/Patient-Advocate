"""create_transparency_report

Revision ID: 0001
Revises:
Create Date: 2026-04-14 00:00:00.000000

Creates the ``transparency_report`` table — the append-only audit ledger
mandated by the Glass Box Imperative (claude.md §4.2).

The table is intentionally devoid of UPDATE and DELETE triggers; data
retention and purging must go through a separate, audited archival process
that itself writes TransparencyRecords before moving data.

Schema
------
+-------------------+------------------------+------------------------------------------+
| Column            | Type                   | Notes                                    |
+===================+========================+==========================================+
| event_id          | UUID PRIMARY KEY       | Immutable, supplied by application       |
| suggestion_id     | UUID (nullable)        | FK-adjacent; no FK constraint (sharded)  |
| agent_name        | VARCHAR(255)           | Which agent produced this record         |
| conflict_type     | VARCHAR(64) nullable   | ConflictType enum .value                 |
| confidence        | FLOAT nullable         | [0.0, 1.0]                               |
| scholarly_basis   | TEXT nullable          | Peer-reviewed citation                   |
| patient_decision  | VARCHAR(32) nullable   | HITLDecision enum .value                 |
| reasoning_chain   | TEXT                   | Full reasoning chain (Glass Box payload) |
| created_at        | TIMESTAMPTZ            | Server-side default; never NULL          |
+-------------------+------------------------+------------------------------------------+

Indexes
-------
* ``ix_transparency_report_agent_name``     — point query by agent
* ``ix_transparency_report_suggestion_id``  — join/lookup by suggestion
* ``ix_transparency_report_created_at``     — time-range scans
* ``ix_transparency_report_agent_created``  — composite (agent + time)

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------------------
# Migration identifiers
# ---------------------------------------------------------------------------
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


# ---------------------------------------------------------------------------
# Upgrade — create table and indexes
# ---------------------------------------------------------------------------

def upgrade() -> None:
    op.create_table(
        "transparency_report",
        # ---- Primary key ---------------------------------------------------
        sa.Column(
            "event_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        # ---- Suggestion reference (nullable; no FK — table may be sharded) --
        sa.Column(
            "suggestion_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        # ---- Agent identity -------------------------------------------------
        sa.Column("agent_name", sa.String(255), nullable=False),
        # ---- Clinical conflict ----------------------------------------------
        sa.Column("conflict_type", sa.String(64), nullable=True),
        # ---- Confidence bound -----------------------------------------------
        sa.Column("confidence", sa.Float(), nullable=True),
        # ---- Scholarly provenance -------------------------------------------
        sa.Column("scholarly_basis", sa.Text(), nullable=True),
        # ---- HITL outcome ---------------------------------------------------
        sa.Column("patient_decision", sa.String(32), nullable=True),
        # ---- Reasoning chain (Glass Box payload) ----------------------------
        sa.Column("reasoning_chain", sa.Text(), nullable=False),
        # ---- Timestamp (server default) -------------------------------------
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )

    # ---- Indexes ------------------------------------------------------------
    op.create_index(
        "ix_transparency_report_agent_name",
        "transparency_report",
        ["agent_name"],
    )
    op.create_index(
        "ix_transparency_report_suggestion_id",
        "transparency_report",
        ["suggestion_id"],
    )
    op.create_index(
        "ix_transparency_report_created_at",
        "transparency_report",
        ["created_at"],
    )
    op.create_index(
        "ix_transparency_report_agent_created",
        "transparency_report",
        ["agent_name", "created_at"],
    )


# ---------------------------------------------------------------------------
# Downgrade — drop indexes then table
# ---------------------------------------------------------------------------

def downgrade() -> None:
    op.drop_index("ix_transparency_report_agent_created", table_name="transparency_report")
    op.drop_index("ix_transparency_report_created_at", table_name="transparency_report")
    op.drop_index("ix_transparency_report_suggestion_id", table_name="transparency_report")
    op.drop_index("ix_transparency_report_agent_name", table_name="transparency_report")
    op.drop_table("transparency_report")
