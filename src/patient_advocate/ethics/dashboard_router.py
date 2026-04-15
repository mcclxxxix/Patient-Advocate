"""
Patient Advocate — Case Manager Ethics Dashboard API
=====================================================

FastAPI router that exposes the in-memory EthicsFlagStore to the case
manager dashboard.  All flag operations are append-only: flags are NEVER
deleted, and the "acknowledge" operation only sets a metadata field so that
the full audit trail is preserved.

Design Principles
-----------------
* **Async-safe** — EthicsFlagStore uses asyncio.Lock for concurrent safety.
* **Append-only** — once ingested, a flag record is immutable.  Acknowledge
  merely marks a flag without removing it.
* **Filterable** — get_all() supports filtering by severity, flag_type, and
  unacknowledged_only to support dashboard views.
* **No external dependencies beyond FastAPI** — no database; state lives in
  process memory.  Production deployments should back this with a PostgreSQL
  INSERT-only table.

References
----------
- Carey (2024), Advances in Consumer Research — audit logs as the legal
  record; append-only design supports ex post explanation and litigation.
- Hantel et al. (2024), JAMA Network Open — transparency requirements for
  clinical AI; every decision must be inspectable by human reviewers.

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from patient_advocate.core.models import ComplaintType, EthicsFlag


# ---------------------------------------------------------------------------
# In-memory flag store
# ---------------------------------------------------------------------------

class EthicsFlagStore:
    """
    Async-safe in-memory store for EthicsFlag objects.

    All mutations are serialised through an asyncio.Lock.  Flags are
    NEVER deleted; acknowledge() only flips a boolean metadata field.

    Attributes
    ----------
    _flags : dict[UUID, EthicsFlag]
        Primary index keyed by flag_id.
    _acknowledged : set[UUID]
        Set of flag_ids that have been acknowledged by a case manager.
    _patient_index : dict[UUID, list[UUID]]
        Secondary index mapping patient_id → list of flag_ids.
    _lock : asyncio.Lock
        Coroutine-safe mutex.
    """

    def __init__(self) -> None:
        self._flags: dict[UUID, EthicsFlag] = {}
        self._acknowledged: set[UUID] = set()
        self._patient_index: dict[UUID, list[UUID]] = defaultdict(list)
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def ingest(self, flag: EthicsFlag) -> None:
        """
        Store a new EthicsFlag.

        If a flag with the same flag_id already exists it is silently
        ignored (idempotent ingest for message-bus retry scenarios).

        Parameters
        ----------
        flag:
            The EthicsFlag to store.  Must be a fully validated model.
        """
        async with self._lock:
            if flag.flag_id not in self._flags:
                self._flags[flag.flag_id] = flag
                self._patient_index[flag.patient_id].append(flag.flag_id)

    async def acknowledge(self, flag_id: UUID) -> bool:
        """
        Mark a flag as acknowledged by a case manager.

        Flags are **never** deleted; this is a soft-mark operation only.

        Parameters
        ----------
        flag_id:
            UUID of the flag to acknowledge.

        Returns
        -------
        bool
            True if the flag was found and acknowledged; False if not found.
        """
        async with self._lock:
            if flag_id not in self._flags:
                return False
            self._acknowledged.add(flag_id)
            return True

    # ------------------------------------------------------------------
    # Read operations (no lock needed for dict/set reads in CPython,
    # but we acquire the lock for consistency in non-GIL environments)
    # ------------------------------------------------------------------

    async def get_all(
        self,
        severity: Optional[str] = None,
        flag_type: Optional[ComplaintType] = None,
        unacknowledged_only: bool = False,
    ) -> list[EthicsFlag]:
        """
        Return all stored flags, with optional filters.

        Parameters
        ----------
        severity:
            If provided, only return flags with this severity level.
        flag_type:
            If provided, only return flags of this ComplaintType.
        unacknowledged_only:
            If True, exclude flags that have been acknowledged.

        Returns
        -------
        list[EthicsFlag]
            Matching flags ordered by timestamp ascending.
        """
        async with self._lock:
            flags = list(self._flags.values())
            acked = set(self._acknowledged)

        result = []
        for flag in flags:
            if severity is not None and flag.severity != severity:
                continue
            if flag_type is not None and flag.flag_type != flag_type:
                continue
            if unacknowledged_only and flag.flag_id in acked:
                continue
            result.append(flag)

        result.sort(key=lambda f: f.timestamp)
        return result

    async def get_for_patient(self, patient_id: UUID) -> list[EthicsFlag]:
        """
        Return all flags for a specific patient, ordered by timestamp.

        Parameters
        ----------
        patient_id:
            UUID of the patient.

        Returns
        -------
        list[EthicsFlag]
            All flags for the patient, or an empty list if none exist.
        """
        async with self._lock:
            flag_ids = list(self._patient_index.get(patient_id, []))
            flags = [self._flags[fid] for fid in flag_ids if fid in self._flags]

        flags.sort(key=lambda f: f.timestamp)
        return flags

    async def get_by_id(self, flag_id: UUID) -> Optional[EthicsFlag]:
        """Return a single flag by its flag_id, or None."""
        async with self._lock:
            return self._flags.get(flag_id)

    async def is_acknowledged(self, flag_id: UUID) -> bool:
        """Return True if the flag has been acknowledged."""
        async with self._lock:
            return flag_id in self._acknowledged

    async def summary_counts(self) -> dict:
        """
        Return aggregate counts broken down by flag type and severity.

        Returns
        -------
        dict with keys:
            ``total``        — total number of flags
            ``acknowledged`` — number of acknowledged flags
            ``by_type``      — {ComplaintType.value: int}
            ``by_severity``  — {severity_string: int}
            ``escalation_required`` — count of high/critical flags
        """
        async with self._lock:
            flags = list(self._flags.values())
            acked_count = len(self._acknowledged)

        by_type: dict[str, int] = defaultdict(int)
        by_severity: dict[str, int] = defaultdict(int)
        escalation_count = 0

        for flag in flags:
            by_type[flag.flag_type.value] += 1
            by_severity[flag.severity] += 1
            if flag.severity in ("high", "critical"):
                escalation_count += 1

        return {
            "total": len(flags),
            "acknowledged": acked_count,
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "escalation_required": escalation_count,
        }

    async def get_escalation_queue(self) -> list[EthicsFlag]:
        """
        Return all unacknowledged high/critical flags.

        Used by the /ethics/legal-queue endpoint to surface items requiring
        attorney review.
        """
        return await self.get_all(unacknowledged_only=True)
        # Further filter for high/critical
        all_unacked = await self.get_all(unacknowledged_only=True)
        return [f for f in all_unacked if f.severity in ("high", "critical")]


# ---------------------------------------------------------------------------
# Singleton store instance (process-lifetime)
# ---------------------------------------------------------------------------

_flag_store = EthicsFlagStore()


def get_flag_store() -> EthicsFlagStore:
    """Dependency-injection accessor for the singleton store."""
    return _flag_store


# ---------------------------------------------------------------------------
# Response schemas (lightweight, avoids circular imports from api.schemas)
# ---------------------------------------------------------------------------

def _flag_to_dict(flag: EthicsFlag, acknowledged: bool = False) -> dict:
    """Serialize an EthicsFlag to a JSON-friendly dict."""
    return {
        "flag_id": str(flag.flag_id),
        "patient_id": str(flag.patient_id),
        "flag_type": flag.flag_type.value,
        "description": flag.description,
        "severity": flag.severity,
        "source_agent": flag.source_agent,
        "timestamp": flag.timestamp.isoformat(),
        "acknowledged": acknowledged,
    }


# ---------------------------------------------------------------------------
# FastAPI router — 6 endpoints
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/ethics", tags=["ethics"])


@router.get(
    "/flags",
    summary="List all ethics flags",
    description=(
        "Returns all stored ethics flags.  Optionally filter by severity, "
        "flag_type, or limit to unacknowledged flags only."
    ),
)
async def list_flags(
    severity: Optional[str] = Query(
        None,
        description="Filter by severity: low | medium | high | critical",
    ),
    flag_type: Optional[ComplaintType] = Query(
        None,
        description="Filter by complaint type",
    ),
    unacknowledged_only: bool = Query(
        False,
        description="If true, only return unacknowledged flags",
    ),
) -> list[dict]:
    store = get_flag_store()
    flags = await store.get_all(
        severity=severity,
        flag_type=flag_type,
        unacknowledged_only=unacknowledged_only,
    )
    acked = store._acknowledged  # read-safe: we already hold no lock here
    return [_flag_to_dict(f, acknowledged=(f.flag_id in acked)) for f in flags]


@router.get(
    "/flags/{patient_id}",
    summary="List flags for a specific patient",
    description="Returns all ethics flags associated with the given patient UUID.",
)
async def list_flags_for_patient(patient_id: UUID) -> list[dict]:
    store = get_flag_store()
    flags = await store.get_for_patient(patient_id)
    acked = store._acknowledged
    return [_flag_to_dict(f, acknowledged=(f.flag_id in acked)) for f in flags]


@router.get(
    "/flags/detail/{flag_id}",
    summary="Get a single ethics flag by ID",
    description="Returns the full detail for a single flag.",
)
async def get_flag_detail(flag_id: UUID) -> dict:
    store = get_flag_store()
    flag = await store.get_by_id(flag_id)
    if flag is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Flag {flag_id} not found.",
        )
    acked = await store.is_acknowledged(flag_id)
    return _flag_to_dict(flag, acknowledged=acked)


@router.post(
    "/flags/{flag_id}/acknowledge",
    summary="Acknowledge an ethics flag",
    description=(
        "Marks a flag as reviewed by the case manager.  "
        "The flag is NEVER deleted — this is a soft-mark only."
    ),
    status_code=status.HTTP_200_OK,
)
async def acknowledge_flag(flag_id: UUID) -> dict:
    store = get_flag_store()
    success = await store.acknowledge(flag_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Flag {flag_id} not found.",
        )
    return {"flag_id": str(flag_id), "acknowledged": True}


@router.get(
    "/summary",
    summary="Summary counts of ethics flags",
    description=(
        "Returns aggregate counts broken down by flag type, severity, "
        "and escalation status."
    ),
)
async def get_summary() -> dict:
    store = get_flag_store()
    return await store.summary_counts()


@router.get(
    "/legal-queue",
    summary="Flags requiring legal escalation",
    description=(
        "Returns unacknowledged high/critical flags that require "
        "attorney review before any external filing."
    ),
)
async def get_legal_queue() -> list[dict]:
    store = get_flag_store()
    all_unacked = await store.get_all(unacknowledged_only=True)
    escalation_flags = [
        f for f in all_unacked if f.severity in ("high", "critical")
    ]
    acked = store._acknowledged
    return [
        _flag_to_dict(f, acknowledged=(f.flag_id in acked))
        for f in escalation_flags
    ]
