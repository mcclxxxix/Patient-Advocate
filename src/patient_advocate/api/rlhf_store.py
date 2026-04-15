"""
Patient Advocate — RLHF Feedback Store (Append-Only JSONL)
==========================================================

Captures rejected scheduling suggestions so that future model versions can
be retrained on patient-specific preferences.

Design decisions:
  - **Append-only JSONL**: one JSON record per line, which is trivially
    readable by pandas, HuggingFace datasets, and most ML tooling.
  - **Only REJECTED decisions are persisted.**  Approvals carry no signal
    beyond "don't change this behaviour", which is implicit from baseline
    performance.  Storing them would double I/O for zero retraining benefit.
  - **``asyncio.to_thread``** offloads the blocking ``open()`` / ``write()``
    call to a thread-pool executor so the event loop is never stalled.
  - File rotation is intentionally *out of scope* here; production deployments
    should mount the JSONL path on a log-aggregation volume (e.g., FluentBit).

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from patient_advocate.core.models import HITLDecision, RLHFFeedbackRecord

logger = logging.getLogger(__name__)

# Sentinel value used when no path is configured (data goes to /dev/null).
_DEFAULT_PATH = Path("rlhf_feedback.jsonl")


def _append_sync(path: Path, record: RLHFFeedbackRecord) -> None:
    """
    Synchronous write helper, executed in a thread via ``asyncio.to_thread``.

    Using ``mode="a"`` with ``newline=""`` on text mode guarantees that each
    ``json.dumps()`` call produces exactly one line in the JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = record.model_dump_json() + "\n"
    with path.open(mode="a", encoding="utf-8") as fh:
        fh.write(line)


class RLHFStore:
    """
    Append-only JSONL store for RLHF retraining signals.

    Only ``REJECTED`` decisions are written.  Every call to ``append``
    that receives an ``APPROVED`` or ``DEFERRED`` decision is silently
    ignored (not an error) because those decisions carry no retraining
    signal for the current scheduling heuristics.

    Usage::

        store = RLHFStore(path=Path("/data/rlhf/feedback.jsonl"))
        await store.append(record)
    """

    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Absolute path to the JSONL file (read-only after construction)."""
        return self._path

    async def append(self, record: RLHFFeedbackRecord) -> None:
        """
        Persist a single feedback record if the decision is ``REJECTED``.

        Non-rejected records are a silent no-op; this keeps the call site
        clean — the service layer does not need to guard with an ``if``
        statement before calling.

        Raises:
            OSError: propagated from the underlying file write (disk full,
                     permission denied, etc.) so the caller can log and
                     raise ``AuditWriteError``.
        """
        if record.decision != HITLDecision.REJECTED:
            logger.debug(
                "RLHFStore.append: skipping non-rejected decision '%s' for suggestion %s",
                record.decision.value,
                record.suggestion_id,
            )
            return

        logger.info(
            "RLHFStore: persisting REJECTED feedback for suggestion %s (patient %s)",
            record.suggestion_id,
            record.patient_id,
        )
        await asyncio.to_thread(_append_sync, self._path, record)

    # ------------------------------------------------------------------
    # Test / introspection helpers
    # ------------------------------------------------------------------

    async def read_all(self) -> list[dict]:
        """
        Read all persisted records.

        Provided primarily for integration tests.  In production, consume
        the JSONL directly via the log pipeline rather than through this method.
        """

        def _read_sync() -> list[dict]:
            if not self._path.exists():
                return []
            records: list[dict] = []
            with self._path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning("Malformed JSONL line in %s — skipping", self._path)
            return records

        return await asyncio.to_thread(_read_sync)
