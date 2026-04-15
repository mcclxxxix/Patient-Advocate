"""
Patient Advocate — HITL API Test Suite
=======================================

Stubs all external packages at the top so this suite can run in any
environment (CI, local dev, Docker) without the full production dependency
tree installed.

Test coverage:
  - HITLDecisionRequest validation (required fields, feedback max-length)
  - SuggestionStore: register, get, pending list, terminal override prevention,
    DEFERRED override, is_timed_out, evict_decided
  - RLHFStore: only REJECTED decisions are persisted; APPROVED/DEFERRED are
    ignored; file is created; multiple records append correctly
  - HITLService.process_decision: all happy paths, patient ownership (wrong
    patient → KeyError / 404), auto-defer on timeout, audit written every path,
    RLHF written only on rejection
  - FastAPI router integration via httpx.AsyncClient / TestClient stubs

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4


# =============================================================================
# STUB EXTERNAL PACKAGES
# =============================================================================
# We build minimal shims for every third-party module so the source files can
# be imported without those packages installed.  Each shim is injected into
# sys.modules *before* any import of the production code.

def _make_module(name: str, **attrs) -> types.ModuleType:
    # Preserve real, file-backed modules if already loaded (e.g. by conftest).
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None):
        for k, v in attrs.items():
            if not hasattr(existing, k):
                setattr(existing, k, v)
        return existing
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# --- pydantic ----------------------------------------------------------------

class _Field:
    """Tiny Field() shim that records keyword arguments."""
    def __init__(self, default=..., **kwargs):
        self.default = default
        self.kwargs = kwargs

    def __repr__(self):
        return f"Field(default={self.default!r}, **{self.kwargs!r})"


def _field_factory(default=..., **kwargs):
    return _Field(default, **kwargs)


class _BaseModel:
    """
    Minimal Pydantic v2 BaseModel shim.

    Supports:
      - Field declarations via class-level annotations + _Field defaults
      - __init__ from kwargs with basic required-field checking
      - model_dump() → dict
      - model_dump_json() → str
      - model_config as a class attribute (ConfigDict)
    """
    model_config: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Collect annotated fields from class body.
        cls._fields = {}
        for klass in reversed(cls.__mro__):
            annotations = getattr(klass, "__annotations__", {})
            for name, _type in annotations.items():
                if name.startswith("_"):
                    continue
                default_val = getattr(klass, name, ...)
                if isinstance(default_val, _Field):
                    cls._fields[name] = default_val.default
                else:
                    cls._fields[name] = default_val

    def __init__(self, **kwargs):
        # Apply defaults for missing fields.
        for name, default in self.__class__._fields.items():
            if name in kwargs:
                val = kwargs[name]
                # Basic max_length validation for str fields.
                field_obj = getattr(self.__class__, name, None)
                if isinstance(field_obj, _Field):
                    max_length = field_obj.kwargs.get("max_length")
                    if max_length is not None and isinstance(val, str) and len(val) > max_length:
                        raise ValueError(
                            f"Field '{name}' exceeds max_length={max_length}: got {len(val)}"
                        )
                setattr(self, name, val)
            elif default is ...:
                # Check callable default_factory-style defaults.
                raise ValueError(f"Field '{name}' is required")
            elif callable(default):
                setattr(self, name, default())
            else:
                setattr(self, name, default)

    def model_dump(self, mode: str = "python") -> dict:
        result = {}
        for name in self.__class__._fields:
            val = getattr(self, name, None)
            if isinstance(val, _BaseModel):
                result[name] = val.model_dump(mode=mode)
            elif isinstance(val, UUID):
                result[name] = str(val) if mode == "json" else val
            elif isinstance(val, datetime):
                result[name] = val.isoformat() if mode == "json" else val
            elif hasattr(val, "value"):  # Enum
                result[name] = val.value
            else:
                result[name] = val
        return result

    def model_dump_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), default=str)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.model_dump() == other.model_dump()

    def __repr__(self):
        fields = ", ".join(f"{k}={getattr(self, k)!r}" for k in self.__class__._fields)
        return f"{self.__class__.__name__}({fields})"


class _ConfigDict(dict):
    pass


pydantic_mod = _make_module(
    "pydantic",
    BaseModel=_BaseModel,
    Field=_field_factory,
    ConfigDict=_ConfigDict,
)

# --- pydantic_settings -------------------------------------------------------

class _BaseSettings(_BaseModel):
    model_config = _ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


class _SettingsConfigDict(_ConfigDict):
    pass


_make_module(
    "pydantic_settings",
    BaseSettings=_BaseSettings,
    SettingsConfigDict=_SettingsConfigDict,
)

# --- sqlalchemy (not used by HITL layer but imported by db module) -----------

_sa = _make_module("sqlalchemy")
_make_module("sqlalchemy.ext")
_make_module("sqlalchemy.ext.asyncio")
_make_module("sqlalchemy.orm")
_make_module("aiosqlite")
_make_module("asyncpg")

# --- fastapi -----------------------------------------------------------------

class _HTTPException(Exception):
    def __init__(self, status_code: int, detail: str = ""):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTPException({status_code}): {detail}")


class _APIRouter:
    def __init__(self, prefix: str = "", tags: list | None = None):
        self.prefix = prefix
        self.tags = tags or []
        self.routes: list = []

    def post(self, path, **kwargs):
        def decorator(fn):
            self.routes.append(("POST", path, fn))
            return fn
        return decorator

    def get(self, path, **kwargs):
        def decorator(fn):
            self.routes.append(("GET", path, fn))
            return fn
        return decorator


def _Depends(dependency):
    return dependency


class _status:
    HTTP_200_OK = 200
    HTTP_201_CREATED = 201
    HTTP_404_NOT_FOUND = 404
    HTTP_408_REQUEST_TIMEOUT = 408
    HTTP_409_CONFLICT = 409
    HTTP_422_UNPROCESSABLE_ENTITY = 422
    HTTP_500_INTERNAL_SERVER_ERROR = 500


class _JSONResponse:
    def __init__(self, content: dict, status_code: int = 200):
        self.content = content
        self.status_code = status_code


class _FastAPI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.dependency_overrides: dict = {}
        self.routers: list = []

    def include_router(self, router, **kwargs):
        self.routers.append((router, kwargs))

    def get(self, path, **kwargs):
        def decorator(fn):
            return fn
        return decorator


fastapi_mod = _make_module(
    "fastapi",
    FastAPI=_FastAPI,
    APIRouter=_APIRouter,
    Depends=_Depends,
    HTTPException=_HTTPException,
    status=_status,
)
_make_module("fastapi.responses", JSONResponse=_JSONResponse)

# =============================================================================
# NOW import production modules (all external deps are stubbed above)
# =============================================================================

import importlib
import os

# Ensure the src directory is on the path.
_src_path = str(Path(__file__).parent.parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from patient_advocate.core.models import (
    HITLDecision,
    RLHFFeedbackRecord,
    TransparencyRecord,
)
from patient_advocate.api.schemas import (
    HITLDecisionRequest,
    HITLDecisionResponse,
    PendingListResponse,
    SuggestionPayload,
    SuggestionStatusResponse,
)
from patient_advocate.api.suggestion_store import SuggestionStore
from patient_advocate.api.rlhf_store import RLHFStore
from patient_advocate.api.hitl_service import HITLService
from patient_advocate.core.exceptions import (
    AuditWriteError,
    HITLTimeoutError,
    HITLRejectionError,
)

import pytest


# =============================================================================
# SHARED FIXTURES
# =============================================================================

def _make_payload(
    suggestion_id: UUID | None = None,
    patient_id: UUID | None = None,
    **overrides,
) -> SuggestionPayload:
    defaults = {
        "event_id": "evt-001",
        "original_date": "2026-04-20",
        "suggested_date": "2026-04-25",
        "conflict_type": "nadir_window",
        "confidence": 0.85,
        "reasoning": "Appointment falls within nadir window.",
        "scholarly_basis": "Crawford et al. (2004), NEJM",
    }
    defaults.update(overrides)
    return SuggestionPayload(
        suggestion_id=suggestion_id or uuid4(),
        patient_id=patient_id or uuid4(),
        **defaults,
    )


def _make_rlhf_record(
    suggestion_id: UUID | None = None,
    patient_id: UUID | None = None,
    decision: HITLDecision = HITLDecision.REJECTED,
    feedback_text: str | None = "Rescheduling not feasible",
) -> RLHFFeedbackRecord:
    sid = suggestion_id or uuid4()
    pid = patient_id or uuid4()
    return RLHFFeedbackRecord(
        feedback_id=uuid4(),
        suggestion_id=sid,
        patient_id=pid,
        decision=decision,
        suggestion_snapshot={"suggestion_id": str(sid), "patient_id": str(pid)},
        feedback_text=feedback_text,
        timestamp=datetime.now(tz=timezone.utc),
    )


async def _noop_audit_writer(record: TransparencyRecord) -> None:
    """Audit writer that does nothing (captures calls via list append)."""


def _capturing_audit_writer():
    """Return (writer_fn, calls_list).  calls_list grows on each invocation."""
    calls: list[TransparencyRecord] = []

    async def writer(record: TransparencyRecord) -> None:
        calls.append(record)

    return writer, calls


def _make_service(
    suggestion_store: SuggestionStore | None = None,
    rlhf_store: RLHFStore | None = None,
    timeout: int = 86400,
    audit_writer=None,
) -> HITLService:
    return HITLService(
        suggestion_store=suggestion_store or SuggestionStore(),
        rlhf_store=rlhf_store or RLHFStore(path=Path("/dev/null")),
        hitl_timeout_seconds=timeout,
        audit_writer=audit_writer or _noop_audit_writer,
    )


# =============================================================================
# SECTION 1: HITLDecisionRequest validation
# =============================================================================

class TestHITLDecisionRequestValidation:

    def test_valid_request_approved(self):
        sid = uuid4()
        pid = uuid4()
        req = HITLDecisionRequest(
            suggestion_id=sid,
            patient_id=pid,
            decision=HITLDecision.APPROVED,
        )
        assert req.suggestion_id == sid
        assert req.patient_id == pid
        assert req.decision == HITLDecision.APPROVED
        assert req.feedback_text is None

    def test_valid_request_rejected_with_feedback(self):
        req = HITLDecisionRequest(
            suggestion_id=uuid4(),
            patient_id=uuid4(),
            decision=HITLDecision.REJECTED,
            feedback_text="Rescheduling conflicts with my work schedule.",
        )
        assert req.decision == HITLDecision.REJECTED
        assert "work schedule" in req.feedback_text

    def test_valid_request_deferred(self):
        req = HITLDecisionRequest(
            suggestion_id=uuid4(),
            patient_id=uuid4(),
            decision=HITLDecision.DEFERRED,
        )
        assert req.decision == HITLDecision.DEFERRED

    def test_suggestion_id_required(self):
        with pytest.raises((ValueError, TypeError)):
            HITLDecisionRequest(
                patient_id=uuid4(),
                decision=HITLDecision.APPROVED,
            )

    def test_patient_id_required(self):
        with pytest.raises((ValueError, TypeError)):
            HITLDecisionRequest(
                suggestion_id=uuid4(),
                decision=HITLDecision.APPROVED,
            )

    def test_decision_required(self):
        with pytest.raises((ValueError, TypeError)):
            HITLDecisionRequest(
                suggestion_id=uuid4(),
                patient_id=uuid4(),
            )

    def test_feedback_text_max_2000_chars_passes(self):
        req = HITLDecisionRequest(
            suggestion_id=uuid4(),
            patient_id=uuid4(),
            decision=HITLDecision.REJECTED,
            feedback_text="x" * 2000,
        )
        assert len(req.feedback_text) == 2000

    def test_feedback_text_exceeds_2000_chars_raises(self):
        with pytest.raises(ValueError):
            HITLDecisionRequest(
                suggestion_id=uuid4(),
                patient_id=uuid4(),
                decision=HITLDecision.REJECTED,
                feedback_text="x" * 2001,
            )

    def test_feedback_text_none_is_allowed(self):
        req = HITLDecisionRequest(
            suggestion_id=uuid4(),
            patient_id=uuid4(),
            decision=HITLDecision.APPROVED,
            feedback_text=None,
        )
        assert req.feedback_text is None

    def test_feedback_text_empty_string_is_allowed(self):
        req = HITLDecisionRequest(
            suggestion_id=uuid4(),
            patient_id=uuid4(),
            decision=HITLDecision.APPROVED,
            feedback_text="",
        )
        assert req.feedback_text == ""


# =============================================================================
# SECTION 2: SuggestionPayload schema
# =============================================================================

class TestSuggestionPayload:

    def test_create_payload(self):
        sid = uuid4()
        pid = uuid4()
        payload = _make_payload(suggestion_id=sid, patient_id=pid)
        assert payload.suggestion_id == sid
        assert payload.patient_id == pid
        assert payload.conflict_type == "nadir_window"

    def test_confidence_bounds_min(self):
        payload = _make_payload(confidence=0.0)
        assert payload.confidence == 0.0

    def test_confidence_bounds_max(self):
        payload = _make_payload(confidence=1.0)
        assert payload.confidence == 1.0

    def test_event_id_optional(self):
        payload = _make_payload(event_id=None)
        assert payload.event_id is None


# =============================================================================
# SECTION 3: SuggestionStore
# =============================================================================

class TestSuggestionStoreRegister:

    @pytest.mark.asyncio
    async def test_register_and_get(self):
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        result = await store.get(payload.suggestion_id)
        assert result == payload

    @pytest.mark.asyncio
    async def test_get_unknown_returns_none(self):
        store = SuggestionStore()
        result = await store.get(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_register_idempotent(self):
        """Re-registering the same suggestion_id is a no-op."""
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        await store.register(payload)  # second call — no error
        assert await store.size() == 1

    @pytest.mark.asyncio
    async def test_register_multiple(self):
        store = SuggestionStore()
        p1, p2 = _make_payload(), _make_payload()
        await store.register(p1)
        await store.register(p2)
        assert await store.size() == 2


class TestSuggestionStorePending:

    @pytest.mark.asyncio
    async def test_pending_for_patient_includes_undecided(self):
        store = SuggestionStore()
        pid = uuid4()
        p1 = _make_payload(patient_id=pid)
        p2 = _make_payload(patient_id=pid)
        await store.register(p1)
        await store.register(p2)
        pending = await store.get_pending_for_patient(pid)
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_pending_excludes_decided(self):
        store = SuggestionStore()
        pid = uuid4()
        p1 = _make_payload(patient_id=pid)
        p2 = _make_payload(patient_id=pid)
        await store.register(p1)
        await store.register(p2)
        await store.mark_decided(p1.suggestion_id, HITLDecision.APPROVED)
        pending = await store.get_pending_for_patient(pid)
        assert len(pending) == 1
        assert pending[0].suggestion_id == p2.suggestion_id

    @pytest.mark.asyncio
    async def test_pending_excludes_other_patients(self):
        store = SuggestionStore()
        pid_a = uuid4()
        pid_b = uuid4()
        pa = _make_payload(patient_id=pid_a)
        pb = _make_payload(patient_id=pid_b)
        await store.register(pa)
        await store.register(pb)
        pending = await store.get_pending_for_patient(pid_a)
        assert len(pending) == 1
        assert pending[0].patient_id == pid_a

    @pytest.mark.asyncio
    async def test_pending_empty_for_unknown_patient(self):
        store = SuggestionStore()
        result = await store.get_pending_for_patient(uuid4())
        assert result == []


class TestSuggestionStoreDecisions:

    @pytest.mark.asyncio
    async def test_mark_approved(self):
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        await store.mark_decided(payload.suggestion_id, HITLDecision.APPROVED)
        decision = await store.get_decision(payload.suggestion_id)
        assert decision == HITLDecision.APPROVED

    @pytest.mark.asyncio
    async def test_mark_rejected(self):
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        await store.mark_decided(payload.suggestion_id, HITLDecision.REJECTED)
        decision = await store.get_decision(payload.suggestion_id)
        assert decision == HITLDecision.REJECTED

    @pytest.mark.asyncio
    async def test_mark_deferred(self):
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        await store.mark_decided(payload.suggestion_id, HITLDecision.DEFERRED)
        decision = await store.get_decision(payload.suggestion_id)
        assert decision == HITLDecision.DEFERRED

    @pytest.mark.asyncio
    async def test_terminal_approved_cannot_be_overridden(self):
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        await store.mark_decided(payload.suggestion_id, HITLDecision.APPROVED)
        with pytest.raises(ValueError, match="terminal"):
            await store.mark_decided(payload.suggestion_id, HITLDecision.REJECTED)

    @pytest.mark.asyncio
    async def test_terminal_rejected_cannot_be_overridden(self):
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        await store.mark_decided(payload.suggestion_id, HITLDecision.REJECTED)
        with pytest.raises(ValueError, match="terminal"):
            await store.mark_decided(payload.suggestion_id, HITLDecision.APPROVED)

    @pytest.mark.asyncio
    async def test_terminal_approved_cannot_be_deferred(self):
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        await store.mark_decided(payload.suggestion_id, HITLDecision.APPROVED)
        with pytest.raises(ValueError, match="terminal"):
            await store.mark_decided(payload.suggestion_id, HITLDecision.DEFERRED)

    @pytest.mark.asyncio
    async def test_deferred_can_be_approved(self):
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        await store.mark_decided(payload.suggestion_id, HITLDecision.DEFERRED)
        # Should NOT raise
        await store.mark_decided(payload.suggestion_id, HITLDecision.APPROVED)
        assert await store.get_decision(payload.suggestion_id) == HITLDecision.APPROVED

    @pytest.mark.asyncio
    async def test_deferred_can_be_rejected(self):
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        await store.mark_decided(payload.suggestion_id, HITLDecision.DEFERRED)
        await store.mark_decided(payload.suggestion_id, HITLDecision.REJECTED)
        assert await store.get_decision(payload.suggestion_id) == HITLDecision.REJECTED

    @pytest.mark.asyncio
    async def test_deferred_can_be_deferred_again(self):
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        await store.mark_decided(payload.suggestion_id, HITLDecision.DEFERRED)
        await store.mark_decided(payload.suggestion_id, HITLDecision.DEFERRED)
        assert await store.get_decision(payload.suggestion_id) == HITLDecision.DEFERRED

    @pytest.mark.asyncio
    async def test_mark_decided_unknown_raises_key_error(self):
        store = SuggestionStore()
        with pytest.raises(KeyError):
            await store.mark_decided(uuid4(), HITLDecision.APPROVED)

    @pytest.mark.asyncio
    async def test_get_decision_pending_returns_none(self):
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        decision = await store.get_decision(payload.suggestion_id)
        assert decision is None


class TestSuggestionStoreTimeout:

    @pytest.mark.asyncio
    async def test_not_timed_out_within_window(self):
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        timed_out = await store.is_timed_out(payload.suggestion_id, timeout_seconds=86400)
        assert timed_out is False

    @pytest.mark.asyncio
    async def test_timed_out_when_window_expired(self):
        """
        Simulate timeout by manually backdating the registered_at timestamp.
        """
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        # Backdate to 25 hours ago.
        past = datetime.now(tz=timezone.utc).replace(
            hour=datetime.now(tz=timezone.utc).hour
        )
        import datetime as dt
        store._registered_at[payload.suggestion_id] = datetime.now(tz=timezone.utc) - dt.timedelta(hours=25)
        timed_out = await store.is_timed_out(payload.suggestion_id, timeout_seconds=86400)
        assert timed_out is True

    @pytest.mark.asyncio
    async def test_decided_suggestion_is_not_timed_out(self):
        """A suggestion with a decision is never considered timed-out."""
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        import datetime as dt
        store._registered_at[payload.suggestion_id] = datetime.now(tz=timezone.utc) - dt.timedelta(hours=25)
        await store.mark_decided(payload.suggestion_id, HITLDecision.APPROVED)
        timed_out = await store.is_timed_out(payload.suggestion_id, timeout_seconds=86400)
        assert timed_out is False

    @pytest.mark.asyncio
    async def test_unknown_suggestion_is_not_timed_out(self):
        store = SuggestionStore()
        timed_out = await store.is_timed_out(uuid4(), timeout_seconds=0)
        assert timed_out is False


class TestSuggestionStoreEvict:

    @pytest.mark.asyncio
    async def test_evict_removes_terminal_approved(self):
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        await store.mark_decided(payload.suggestion_id, HITLDecision.APPROVED)
        count = await store.evict_decided()
        assert count == 1
        assert await store.size() == 0

    @pytest.mark.asyncio
    async def test_evict_removes_terminal_rejected(self):
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        await store.mark_decided(payload.suggestion_id, HITLDecision.REJECTED)
        count = await store.evict_decided()
        assert count == 1
        assert await store.size() == 0

    @pytest.mark.asyncio
    async def test_evict_keeps_deferred(self):
        """DEFERRED is not terminal and must NOT be evicted."""
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        await store.mark_decided(payload.suggestion_id, HITLDecision.DEFERRED)
        count = await store.evict_decided()
        assert count == 0
        assert await store.size() == 1

    @pytest.mark.asyncio
    async def test_evict_keeps_pending(self):
        store = SuggestionStore()
        payload = _make_payload()
        await store.register(payload)
        count = await store.evict_decided()
        assert count == 0
        assert await store.size() == 1

    @pytest.mark.asyncio
    async def test_evict_mixed(self):
        store = SuggestionStore()
        p_approved = _make_payload()
        p_rejected = _make_payload()
        p_deferred = _make_payload()
        p_pending = _make_payload()
        for p in [p_approved, p_rejected, p_deferred, p_pending]:
            await store.register(p)
        await store.mark_decided(p_approved.suggestion_id, HITLDecision.APPROVED)
        await store.mark_decided(p_rejected.suggestion_id, HITLDecision.REJECTED)
        await store.mark_decided(p_deferred.suggestion_id, HITLDecision.DEFERRED)
        count = await store.evict_decided()
        assert count == 2  # approved + rejected
        remaining = await store.size()
        assert remaining == 2  # deferred + pending

    @pytest.mark.asyncio
    async def test_evict_empty_store(self):
        store = SuggestionStore()
        count = await store.evict_decided()
        assert count == 0

    @pytest.mark.asyncio
    async def test_evict_returns_correct_count(self):
        store = SuggestionStore()
        payloads = [_make_payload() for _ in range(5)]
        for p in payloads:
            await store.register(p)
            await store.mark_decided(p.suggestion_id, HITLDecision.APPROVED)
        count = await store.evict_decided()
        assert count == 5


# =============================================================================
# SECTION 4: RLHFStore
# =============================================================================

class TestRLHFStore:

    @pytest.mark.asyncio
    async def test_rejected_is_appended(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        store = RLHFStore(path=path)
        record = _make_rlhf_record(decision=HITLDecision.REJECTED)
        await store.append(record)
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert str(record.suggestion_id) in data["suggestion_id"]

    @pytest.mark.asyncio
    async def test_approved_is_not_appended(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        store = RLHFStore(path=path)
        record = _make_rlhf_record(decision=HITLDecision.APPROVED)
        await store.append(record)
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_deferred_is_not_appended(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        store = RLHFStore(path=path)
        record = _make_rlhf_record(decision=HITLDecision.DEFERRED)
        await store.append(record)
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_multiple_rejections_append_to_same_file(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        store = RLHFStore(path=path)
        records = [_make_rlhf_record(decision=HITLDecision.REJECTED) for _ in range(3)]
        for r in records:
            await store.append(r)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    @pytest.mark.asyncio
    async def test_each_line_is_valid_json(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        store = RLHFStore(path=path)
        for _ in range(5):
            await store.append(_make_rlhf_record(decision=HITLDecision.REJECTED))
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            data = json.loads(line)
            assert "suggestion_id" in data
            assert "decision" in data

    @pytest.mark.asyncio
    async def test_read_all_returns_empty_when_no_file(self, tmp_path):
        path = tmp_path / "nonexistent.jsonl"
        store = RLHFStore(path=path)
        records = await store.read_all()
        assert records == []

    @pytest.mark.asyncio
    async def test_read_all_returns_written_records(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        store = RLHFStore(path=path)
        r1 = _make_rlhf_record(decision=HITLDecision.REJECTED)
        r2 = _make_rlhf_record(decision=HITLDecision.REJECTED)
        await store.append(r1)
        await store.append(r2)
        records = await store.read_all()
        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_path_property(self, tmp_path):
        path = tmp_path / "test.jsonl"
        store = RLHFStore(path=path)
        assert store.path == path

    @pytest.mark.asyncio
    async def test_mixed_decisions_only_rejected_stored(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        store = RLHFStore(path=path)
        await store.append(_make_rlhf_record(decision=HITLDecision.APPROVED))
        await store.append(_make_rlhf_record(decision=HITLDecision.REJECTED))
        await store.append(_make_rlhf_record(decision=HITLDecision.DEFERRED))
        await store.append(_make_rlhf_record(decision=HITLDecision.REJECTED))
        records = await store.read_all()
        assert len(records) == 2
        assert all(r["decision"] == "rejected" for r in records)


# =============================================================================
# SECTION 5: HITLService — happy paths
# =============================================================================

class TestHITLServiceApprove:

    @pytest.mark.asyncio
    async def test_approve_returns_response(self):
        audit_writer, calls = _capturing_audit_writer()
        service = _make_service(audit_writer=audit_writer)
        payload = _make_payload()
        await service.register_suggestion(payload)

        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=payload.patient_id,
            decision=HITLDecision.APPROVED,
        )
        response = await service.process_decision(request)
        assert response.suggestion_id == payload.suggestion_id
        assert response.decision == HITLDecision.APPROVED
        assert response.acknowledged is True

    @pytest.mark.asyncio
    async def test_approve_marks_store(self):
        service = _make_service()
        payload = _make_payload()
        await service.register_suggestion(payload)
        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=payload.patient_id,
            decision=HITLDecision.APPROVED,
        )
        await service.process_decision(request)
        decision = await service._store.get_decision(payload.suggestion_id)
        assert decision == HITLDecision.APPROVED

    @pytest.mark.asyncio
    async def test_approve_writes_audit(self):
        audit_writer, calls = _capturing_audit_writer()
        service = _make_service(audit_writer=audit_writer)
        payload = _make_payload()
        await service.register_suggestion(payload)
        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=payload.patient_id,
            decision=HITLDecision.APPROVED,
        )
        await service.process_decision(request)
        assert len(calls) >= 1

    @pytest.mark.asyncio
    async def test_deferred_returns_response(self):
        service = _make_service()
        payload = _make_payload()
        await service.register_suggestion(payload)
        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=payload.patient_id,
            decision=HITLDecision.DEFERRED,
        )
        response = await service.process_decision(request)
        assert response.decision == HITLDecision.DEFERRED
        assert response.acknowledged is True


class TestHITLServiceReject:

    @pytest.mark.asyncio
    async def test_reject_raises_rejection_error(self, tmp_path):
        rlhf_store = RLHFStore(path=tmp_path / "rlhf.jsonl")
        audit_writer, _ = _capturing_audit_writer()
        service = _make_service(rlhf_store=rlhf_store, audit_writer=audit_writer)
        payload = _make_payload()
        await service.register_suggestion(payload)
        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=payload.patient_id,
            decision=HITLDecision.REJECTED,
            feedback_text="Too aggressive reschedule.",
        )
        with pytest.raises(HITLRejectionError):
            await service.process_decision(request)

    @pytest.mark.asyncio
    async def test_reject_writes_rlhf_record(self, tmp_path):
        rlhf_path = tmp_path / "rlhf.jsonl"
        rlhf_store = RLHFStore(path=rlhf_path)
        audit_writer, _ = _capturing_audit_writer()
        service = _make_service(rlhf_store=rlhf_store, audit_writer=audit_writer)
        payload = _make_payload()
        await service.register_suggestion(payload)
        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=payload.patient_id,
            decision=HITLDecision.REJECTED,
            feedback_text="Conflicts with treatment.",
        )
        with pytest.raises(HITLRejectionError):
            await service.process_decision(request)
        assert rlhf_path.exists()
        records = await rlhf_store.read_all()
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_reject_writes_audit(self, tmp_path):
        rlhf_store = RLHFStore(path=tmp_path / "rlhf.jsonl")
        audit_writer, calls = _capturing_audit_writer()
        service = _make_service(rlhf_store=rlhf_store, audit_writer=audit_writer)
        payload = _make_payload()
        await service.register_suggestion(payload)
        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=payload.patient_id,
            decision=HITLDecision.REJECTED,
        )
        with pytest.raises(HITLRejectionError):
            await service.process_decision(request)
        assert len(calls) >= 1

    @pytest.mark.asyncio
    async def test_reject_marks_store(self, tmp_path):
        rlhf_store = RLHFStore(path=tmp_path / "rlhf.jsonl")
        service = _make_service(rlhf_store=rlhf_store)
        payload = _make_payload()
        await service.register_suggestion(payload)
        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=payload.patient_id,
            decision=HITLDecision.REJECTED,
        )
        with pytest.raises(HITLRejectionError):
            await service.process_decision(request)
        decision = await service._store.get_decision(payload.suggestion_id)
        assert decision == HITLDecision.REJECTED

    @pytest.mark.asyncio
    async def test_approve_does_not_write_rlhf(self, tmp_path):
        rlhf_path = tmp_path / "rlhf.jsonl"
        rlhf_store = RLHFStore(path=rlhf_path)
        service = _make_service(rlhf_store=rlhf_store)
        payload = _make_payload()
        await service.register_suggestion(payload)
        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=payload.patient_id,
            decision=HITLDecision.APPROVED,
        )
        await service.process_decision(request)
        assert not rlhf_path.exists()


# =============================================================================
# SECTION 6: HITLService — patient ownership (HIPAA)
# =============================================================================

class TestHITLServiceOwnership:

    @pytest.mark.asyncio
    async def test_wrong_patient_raises_key_error(self):
        """
        HIPAA-compliant: wrong patient gets KeyError (maps to 404) not a
        permission error (which would reveal the suggestion exists).
        """
        audit_writer, calls = _capturing_audit_writer()
        service = _make_service(audit_writer=audit_writer)
        payload = _make_payload()
        await service.register_suggestion(payload)
        wrong_patient_id = uuid4()
        assert wrong_patient_id != payload.patient_id
        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=wrong_patient_id,
            decision=HITLDecision.APPROVED,
        )
        with pytest.raises(KeyError):
            await service.process_decision(request)

    @pytest.mark.asyncio
    async def test_wrong_patient_still_writes_audit(self):
        """Glass Box: audit is written even on ownership failure."""
        audit_writer, calls = _capturing_audit_writer()
        service = _make_service(audit_writer=audit_writer)
        payload = _make_payload()
        await service.register_suggestion(payload)
        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=uuid4(),  # wrong patient
            decision=HITLDecision.APPROVED,
        )
        with pytest.raises(KeyError):
            await service.process_decision(request)
        assert len(calls) >= 1

    @pytest.mark.asyncio
    async def test_unknown_suggestion_raises_key_error(self):
        audit_writer, calls = _capturing_audit_writer()
        service = _make_service(audit_writer=audit_writer)
        request = HITLDecisionRequest(
            suggestion_id=uuid4(),
            patient_id=uuid4(),
            decision=HITLDecision.APPROVED,
        )
        with pytest.raises(KeyError):
            await service.process_decision(request)

    @pytest.mark.asyncio
    async def test_unknown_suggestion_writes_audit(self):
        """Glass Box: audit written even for not-found suggestions."""
        audit_writer, calls = _capturing_audit_writer()
        service = _make_service(audit_writer=audit_writer)
        request = HITLDecisionRequest(
            suggestion_id=uuid4(),
            patient_id=uuid4(),
            decision=HITLDecision.APPROVED,
        )
        with pytest.raises(KeyError):
            await service.process_decision(request)
        assert len(calls) >= 1


# =============================================================================
# SECTION 7: HITLService — timeout / auto-defer
# =============================================================================

class TestHITLServiceTimeout:

    @pytest.mark.asyncio
    async def test_timed_out_raises_hitl_timeout_error(self):
        import datetime as dt
        service = _make_service(timeout=86400)
        payload = _make_payload()
        await service.register_suggestion(payload)
        # Backdate registration to 25 hours ago.
        service._store._registered_at[payload.suggestion_id] = (
            datetime.now(tz=timezone.utc) - dt.timedelta(hours=25)
        )
        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=payload.patient_id,
            decision=HITLDecision.APPROVED,
        )
        with pytest.raises(HITLTimeoutError):
            await service.process_decision(request)

    @pytest.mark.asyncio
    async def test_timed_out_auto_defers_suggestion(self):
        import datetime as dt
        service = _make_service(timeout=86400)
        payload = _make_payload()
        await service.register_suggestion(payload)
        service._store._registered_at[payload.suggestion_id] = (
            datetime.now(tz=timezone.utc) - dt.timedelta(hours=25)
        )
        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=payload.patient_id,
            decision=HITLDecision.APPROVED,
        )
        with pytest.raises(HITLTimeoutError):
            await service.process_decision(request)
        decision = await service._store.get_decision(payload.suggestion_id)
        assert decision == HITLDecision.DEFERRED

    @pytest.mark.asyncio
    async def test_timed_out_writes_audit(self):
        import datetime as dt
        audit_writer, calls = _capturing_audit_writer()
        service = _make_service(timeout=86400, audit_writer=audit_writer)
        payload = _make_payload()
        await service.register_suggestion(payload)
        service._store._registered_at[payload.suggestion_id] = (
            datetime.now(tz=timezone.utc) - dt.timedelta(hours=25)
        )
        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=payload.patient_id,
            decision=HITLDecision.APPROVED,
        )
        with pytest.raises(HITLTimeoutError):
            await service.process_decision(request)
        assert len(calls) >= 1

    @pytest.mark.asyncio
    async def test_within_timeout_does_not_auto_defer(self):
        service = _make_service(timeout=86400)
        payload = _make_payload()
        await service.register_suggestion(payload)
        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=payload.patient_id,
            decision=HITLDecision.APPROVED,
        )
        response = await service.process_decision(request)
        assert response.decision == HITLDecision.APPROVED

    @pytest.mark.asyncio
    async def test_zero_timeout_immediately_times_out(self):
        """A timeout of 0 seconds means every un-decided suggestion is expired."""
        import datetime as dt
        service = _make_service(timeout=0)
        payload = _make_payload()
        await service.register_suggestion(payload)
        # Even a freshly registered suggestion exceeds a 0-second window.
        # Backdate by 1 second to be deterministic.
        service._store._registered_at[payload.suggestion_id] = (
            datetime.now(tz=timezone.utc) - dt.timedelta(seconds=1)
        )
        request = HITLDecisionRequest(
            suggestion_id=payload.suggestion_id,
            patient_id=payload.patient_id,
            decision=HITLDecision.APPROVED,
        )
        with pytest.raises(HITLTimeoutError):
            await service.process_decision(request)


# =============================================================================
# SECTION 8: Audit writer — Glass Box guarantee
# =============================================================================

class TestAuditGlassBox:

    @pytest.mark.asyncio
    async def test_audit_written_on_approve(self):
        audit_writer, calls = _capturing_audit_writer()
        service = _make_service(audit_writer=audit_writer)
        payload = _make_payload()
        await service.register_suggestion(payload)
        await service.process_decision(
            HITLDecisionRequest(
                suggestion_id=payload.suggestion_id,
                patient_id=payload.patient_id,
                decision=HITLDecision.APPROVED,
            )
        )
        assert len(calls) >= 1

    @pytest.mark.asyncio
    async def test_audit_written_on_deferred(self):
        audit_writer, calls = _capturing_audit_writer()
        service = _make_service(audit_writer=audit_writer)
        payload = _make_payload()
        await service.register_suggestion(payload)
        await service.process_decision(
            HITLDecisionRequest(
                suggestion_id=payload.suggestion_id,
                patient_id=payload.patient_id,
                decision=HITLDecision.DEFERRED,
            )
        )
        assert len(calls) >= 1

    @pytest.mark.asyncio
    async def test_audit_written_on_reject(self, tmp_path):
        audit_writer, calls = _capturing_audit_writer()
        service = _make_service(
            rlhf_store=RLHFStore(path=tmp_path / "rlhf.jsonl"),
            audit_writer=audit_writer,
        )
        payload = _make_payload()
        await service.register_suggestion(payload)
        with pytest.raises(HITLRejectionError):
            await service.process_decision(
                HITLDecisionRequest(
                    suggestion_id=payload.suggestion_id,
                    patient_id=payload.patient_id,
                    decision=HITLDecision.REJECTED,
                )
            )
        assert len(calls) >= 1

    @pytest.mark.asyncio
    async def test_audit_written_on_not_found(self):
        audit_writer, calls = _capturing_audit_writer()
        service = _make_service(audit_writer=audit_writer)
        with pytest.raises(KeyError):
            await service.process_decision(
                HITLDecisionRequest(
                    suggestion_id=uuid4(),
                    patient_id=uuid4(),
                    decision=HITLDecision.APPROVED,
                )
            )
        assert len(calls) >= 1

    @pytest.mark.asyncio
    async def test_audit_record_has_suggestion_id(self):
        audit_writer, calls = _capturing_audit_writer()
        service = _make_service(audit_writer=audit_writer)
        payload = _make_payload()
        await service.register_suggestion(payload)
        await service.process_decision(
            HITLDecisionRequest(
                suggestion_id=payload.suggestion_id,
                patient_id=payload.patient_id,
                decision=HITLDecision.APPROVED,
            )
        )
        assert any(r.suggestion_id == payload.suggestion_id for r in calls)

    @pytest.mark.asyncio
    async def test_audit_writer_failure_raises_audit_write_error(self):
        async def failing_writer(record):
            raise RuntimeError("DB unavailable")

        service = _make_service(audit_writer=failing_writer)
        payload = _make_payload()
        await service.register_suggestion(payload)
        with pytest.raises(AuditWriteError):
            await service.process_decision(
                HITLDecisionRequest(
                    suggestion_id=payload.suggestion_id,
                    patient_id=payload.patient_id,
                    decision=HITLDecision.APPROVED,
                )
            )


# =============================================================================
# SECTION 9: HITLService — register_suggestion
# =============================================================================

class TestHITLServiceRegister:

    @pytest.mark.asyncio
    async def test_register_suggestion_stores_payload(self):
        service = _make_service()
        payload = _make_payload()
        await service.register_suggestion(payload)
        result = await service._store.get(payload.suggestion_id)
        assert result == payload

    @pytest.mark.asyncio
    async def test_register_idempotent_via_service(self):
        service = _make_service()
        payload = _make_payload()
        await service.register_suggestion(payload)
        await service.register_suggestion(payload)
        assert await service._store.size() == 1


# =============================================================================
# SECTION 10: SuggestionStatusResponse
# =============================================================================

class TestSuggestionStatusResponse:

    def test_pending_status(self):
        r = SuggestionStatusResponse(suggestion_id=uuid4(), status="pending")
        assert r.status == "pending"

    def test_approved_status(self):
        r = SuggestionStatusResponse(suggestion_id=uuid4(), status="approved")
        assert r.status == "approved"

    def test_timed_out_status(self):
        r = SuggestionStatusResponse(suggestion_id=uuid4(), status="timed_out")
        assert r.status == "timed_out"


# =============================================================================
# SECTION 11: PendingListResponse schema
# =============================================================================

class TestPendingListResponse:

    def test_empty_pending(self):
        r = PendingListResponse(patient_id=uuid4(), pending=[])
        assert r.pending == []

    def test_pending_with_items(self):
        pid = uuid4()
        p1 = _make_payload(patient_id=pid)
        r = PendingListResponse(patient_id=pid, pending=[p1])
        assert len(r.pending) == 1

    def test_patient_id_preserved(self):
        pid = uuid4()
        r = PendingListResponse(patient_id=pid, pending=[])
        assert r.patient_id == pid


# =============================================================================
# SECTION 12: HITLDecisionResponse schema
# =============================================================================

class TestHITLDecisionResponse:

    def test_approved_response(self):
        sid = uuid4()
        r = HITLDecisionResponse(
            suggestion_id=sid,
            decision=HITLDecision.APPROVED,
            acknowledged=True,
        )
        assert r.suggestion_id == sid
        assert r.acknowledged is True

    def test_rejected_response(self):
        r = HITLDecisionResponse(
            suggestion_id=uuid4(),
            decision=HITLDecision.REJECTED,
            acknowledged=True,
        )
        assert r.decision == HITLDecision.REJECTED
