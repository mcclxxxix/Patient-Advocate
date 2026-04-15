# Patient Advocate Level 2 — Architecture, Debugging & Ethics

**Branch:** `claude/keen-ptolemy`
**Test status:** 439 / 439 passing
**PR:** https://github.com/mcclxxxix/Patient-Advocate/pull/2

---

## 1. Module Reference

The system is a LangGraph-shaped multi-agent personal medical assistant. Nothing in the hot path talks to an LLM for routing or clinical decisions — those are deterministic. LLMs are reserved for empathetic phrasing inside agent nodes. Every agent emits a `TransparencyRecord` that is persisted immutably.

### 1.1 `patient_advocate.core`

The contract layer. Everything else imports from here; nothing here imports from elsewhere.

- **`models.py`** — Frozen Pydantic v2 models. All domain objects are immutable so that an audit record written on day N cannot be silently edited on day N+1.
  - `AgentType` (Enum): `VISION_AGENT`, `CALENDAR_ENGINE`, `ETHICS_AGENT`, `MASTER_BRAIN`, etc. Used as the `agent_name` field on audit records.
  - `ConflictType` (Enum): `NADIR_WINDOW`, `FATIGUE_THRESHOLD`, `APPOINTMENT_OVERLAP`, `RECOVERY_PERIOD`. Clinical reasons a suggestion was generated.
  - `HITLDecision` (Enum): `APPROVED`, `REJECTED`, `DEFERRED`. The only three outcomes a human-in-the-loop can return.
  - `ComplaintType` (Enum): used by the ethics screener when the user reports harm, privacy violations, clinical disagreement, etc.
  - `LabValues` — frozen dataclass-shape Pydantic model: `anc`, `wbc`, `hemoglobin`, `platelets`, `fatigue_score`, `collection_date`, `source`. `source` is free text (`"ocr"`, `"manual"`, `"hl7"`) so provenance travels with the data.
  - `ChemoRegimen` — regimen name, cycle length, nadir day window. Powers the calendar engine's nadir-window check.
  - `PatientProfile` — the bundle of regimen + latest `LabValues` + patient preferences. Passed as a single immutable object through the graph.
  - `CalendarEvent` / `CalendarSuggestion` — an event the patient has scheduled, and a counter-proposal from the engine when that event collides with a clinical constraint. `CalendarSuggestion` carries `confidence ∈ [0, 1]`, `conflict_type`, and a prose `reasoning`. Suggestions with `confidence < 0.7` are never auto-applied; they are sent to HITL.
  - `TransparencyRecord` — the Glass Box payload. `event_id` (UUID), `suggestion_id` (optional UUID), `agent_name`, `conflict_type`, `confidence`, `scholarly_basis`, `patient_decision`, `reasoning_chain`, `timestamp`.
  - `EthicsFlag` — raised by the ethics screener; carries a `ComplaintType` and a severity.
  - `RLHFFeedbackRecord` — captured when the patient accepts / rejects / edits a suggestion. This is the training signal that keeps the system honest over time.

- **`exceptions.py`** — Typed errors so the call site can tell *why* something failed without parsing messages. `PatientAdvocateError` is the root; children include `OCRExtractionError` (Vision failed to read a lab image), `AuditWriteError` (Glass Box logger could not persist — this is fatal, the system refuses to proceed without an audit trail), `HITLTimeoutError`, `HITLRejectionError`.

- **`config.py`** — `pydantic-settings`-backed settings object. Reads `PA_DATABASE_URL`, `PA_HITL_TIMEOUT_SECONDS`, `PA_CONFIDENCE_THRESHOLD` (default `0.7`), etc. Never read env vars directly in other modules; they go through `Settings()`.

- **`state.py`** — Typed dict describing the LangGraph `AdvocateState`: `messages`, `patient_profile`, `image_path`, `audit_log`, `hitl_required`, `pending_suggestions`. This is the spine of the graph.

### 1.2 `patient_advocate.routing`

- **`patient_advocate_routing_system.py`** — the "Master Brain." Despite the name, it is a deterministic regex-driven intent classifier, not an LLM. Routing decisions must be reproducible for audit.
  - `IntentClassifier`
    - `PATTERNS` — five pattern sets, one per downstream agent (Vision, Calendar, Ethics, Legal, Generic Info). Each set is a list of compiled regexes. Examples: `\b(anc|neutrophil|wbc|hemoglobin|platelet)\b` → Vision; `\bschedul(?:e|ing)\b|\bappointment\b|\brescheduled?\b` → Calendar; `\bviolat(?:ion|ed|ing|e)\b`, `\bprivacy\b` → Ethics.
    - `classify(message: str) -> tuple[AgentType, float]` — counts pattern hits per set, picks the winner, and maps hit count to a confidence using a diminishing-returns curve: `confidence = 0.55 + min(best_hits, 3) * 0.15`. Three hits already saturates at `1.00`; one hit lands at `0.70` (the HITL threshold). The curve was chosen over a linear `hits / max_patterns` because the original formulation starved legitimate signals with many patterns (Ethics, Legal) of confidence — a single strong match in a big pattern set was being rated lower than a single match in a small pattern set, which is wrong.
  - `master_brain_router(state) -> dict` — wraps the classifier, emits a `TransparencyRecord` describing the routing decision (so you can later ask *"why did this message go to Ethics?"* and get an answer), and returns the next node name for LangGraph to dispatch.
  - `END_ROUTE` sentinel — returned when confidence falls below the floor; the graph terminates with a "please rephrase" response rather than guessing.

### 1.3 `patient_advocate.calendar_engine`

- **`calendar_engine.py`** — pure-function clinical logic, no I/O.
  - `NadirWindowChecker` — given a regimen and an event date, returns whether the event falls inside the chemo nadir window (ANC trough). Built on Crawford 2004 (NEJM, febrile neutropenia) — a chemotherapy appointment on day 10 of an R-CHOP cycle is a direct infection risk and must be rescheduled.
  - `FatigueThresholdChecker` — cross-references the patient's Bower-2014 fatigue score (validated 0-10 scale) with the event's metabolic cost (estimated MET minutes). Events above threshold on high-fatigue days trigger a suggestion to move them.
  - `AppointmentOverlapChecker` — trivially, two appointments on the same half-hour block. Still audited.
  - `RecoveryPeriodChecker` — post-infusion recovery windows (regimen-specific, typically 48–72h).
  - `CalendarEngine.suggest(event, profile) -> Optional[CalendarSuggestion]` — runs all checkers, picks the highest-priority conflict, returns a suggestion with `confidence`, `conflict_type`, `reasoning`, and `scholarly_basis`. Returns `None` when the event is clean.

### 1.4 `patient_advocate.agents.vision_agent`

- **`vision_agent.py`** — turns lab-report screenshots into `LabValues`.
  - `LabValueParser` — static methods `parse_anc`, `parse_wbc`, `parse_hemoglobin`, `parse_platelets`. Each uses a tight regex (e.g. `ANC` / `absolute neutrophil count` / `abs. neut.`) with sanity-clipping (ANC outside `[0, 20_000]` → `None`, not a false value). The sanity clamps exist because OCR noise will cheerfully emit `ANC: 1500000` from a smudge.
  - `VisionAgent.extract_from_image(path) -> tuple[LabValues, TransparencyRecord]` — opens the file via PIL, applies a sharpen filter to help Tesseract, runs `pytesseract.image_to_string`, parses each field, packages a `LabValues` + a `TransparencyRecord` whose `reasoning_chain` shows exactly which line of OCR output produced each field. Raises `OCRExtractionError` if the file is missing/unreadable.
  - `vision_agent_node(state)` — the LangGraph node wrapper. Always sets `hitl_required=True` because OCR-extracted labs are never auto-accepted; the human must confirm before the calendar engine reasons on them.

### 1.5 `patient_advocate.api`

FastAPI HITL surface. This is where clinicians / patients approve or override the AI.

- **`app.py`** — `create_app()` factory; wires the HITL router, mounts middleware, and attaches lifespan hooks for the audit logger.
- **`schemas.py`** — request/response Pydantic models (`SuggestionPayload`, `DecisionRequest`, `PendingListResponse`, `SuggestionStatusResponse`). Separate from `core.models` so that the wire format can evolve independently of the domain.
- **`suggestion_store.py`** — `SuggestionStore`: in-process map of `suggestion_id` → `SuggestionPayload`, lifecycle states (`pending`, `approved`, `rejected`, `expired`). Thread-safe via `asyncio.Lock`.
- **`rlhf_store.py`** — `RLHFStore`: captures `RLHFFeedbackRecord` for every HITL decision. This is the training feedback loop.
- **`hitl_service.py`** — `HITLService`: coordinates `SuggestionStore` + `RLHFStore` + `GlassBoxLogger`. Methods: `submit_suggestion`, `await_decision(timeout)` (raises `HITLTimeoutError` on expiry), `record_decision`.
- **`hitl_router.py`** — FastAPI routes: `POST /hitl/suggestions`, `GET /hitl/suggestions/pending`, `POST /hitl/suggestions/{id}/decision`, `GET /hitl/suggestions/{id}`.

### 1.6 `patient_advocate.db`

- **`orm.py`** — SQLAlchemy 2.0 `DeclarativeBase` + `TransparencyReportRow`. Fields mirror `TransparencyRecord` one-for-one with an explicit `from_domain(record)` classmethod (no `**kwargs` unpacking — schema drift must be loud). Naive `datetime` values are converted to UTC-aware before persistence. Indexed on `agent_name`, `suggestion_id`, `created_at`, and the compound `(agent_name, created_at)` for "show me everything vision_agent did this week."
- **`engine.py`** — `get_engine()` / `get_async_session_factory()`. Uses `pool_pre_ping=True` because the audit path cannot tolerate a stale connection eating a write.
- **`glass_box_logger.py`** — `GlassBoxLogger`: `async def write(record)` and `async def write_many(records)`. `write_many` is atomic — either all records land or none, with `AuditWriteError` wrapping the underlying DB exception. There is intentionally **no** `update` or `delete` method. The table is append-only by contract.

### 1.7 `patient_advocate.ethics`

Covered in depth in §3. Three files: `ethics_screener.py` (pattern-based triage), `ethics_agent.py` (full reasoning node + escalation), `dashboard_router.py` (routes confirmed harms to a human oversight queue).

---

## 2. Debugging Process

Starting state: 82 failures / 357 passes out of 439 tests. Final state: 439 / 439. This took five distinct problem classes, each solved once and then protected against recurrence.

### 2.1 "`patient_advocate` is not a package"

The very first collection failed. Cause: the test files had been written to run against a stub world (`sys.modules["patient_advocate.core.models"] = types.ModuleType(...)`), and those stubs pointed at nothing. When a real `from patient_advocate.core.models import TransparencyRecord` ran, Python found a bare `ModuleType` with no `TransparencyRecord` attribute.

**Fix:** a repo-root `conftest.py` that pre-imports the real `patient_advocate` submodules using real `pydantic` / `sqlalchemy` / `fastapi` *before* any test file runs. This populates `sys.modules` with file-backed modules that have `__file__` set.

### 2.2 "`cannot import name 'RLHFFeedbackRecord' from 'patient_advocate.core.models'`"

Second-order damage from the fix above. A test file earlier in the collection order (`test_hitl.py`, `test_routing.py`, `test_vision_agent.py`) would *overwrite* `sys.modules["patient_advocate.core.models"]` with its stub. By the time the next test file tried to import the real `RLHFFeedbackRecord`, the stub was still there.

**Fix:** `pytest_collectstart(collector)` hook in `conftest.py`. Before each test module is collected, it:
1. Restores the real `patient_advocate.*` modules from a snapshot taken at session start.
2. Iterates `sys.modules` and deletes any `sqlalchemy*`, `pydantic*`, `fastapi*`, `alembic*`, `langchain_core*`, `langgraph*` entry whose `__file__` is `None` (i.e. a stub).

This guarantees each test file starts with real deps *and* its own stubs can still win for the duration of that file.

### 2.3 "`TransparencyReportRow() takes no arguments`"

Symptom: 49 DB tests failing in the full suite, passing in isolation. The stub `DeclarativeBase` — a plain `class _FakeDeclarativeBase: pass` — has no `__init__` that accepts `**kwargs`. Real SQLAlchemy's `DeclarativeBase` generates one from the mapped columns.

Root cause: `test_hitl.py`'s `_make_module` helper used bare assignment (`sys.modules[name] = mod`) rather than `setdefault`, so it was unconditionally overwriting real `sqlalchemy.orm` with its stub. Even though the collection hook later purged stubs, any *already-imported* Python module (like `patient_advocate.db.orm`, which had captured `DeclarativeBase` at import time) now held a reference to the stub.

**Fix:** rewrote `_make_module` to preserve real file-backed modules:
```python
def _make_module(name, **attrs):
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None):
        # real module already loaded — just add missing attrs
        for k, v in attrs.items():
            if not hasattr(existing, k):
                setattr(existing, k, v)
        return existing
    # otherwise build a fresh stub
    ...
```

### 2.4 Python 3.14 asyncio

`asyncio.get_event_loop()` no longer auto-creates a loop outside an `async` context. The DB test helper `_run(coro)` blew up with `RuntimeError: There is no current event loop`.

**Fix:**
```python
def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
```

### 2.5 Routing confidence starvation

Ethics and Legal pattern sets each had ~8 patterns. The original confidence formula `0.55 + (hits/max_patterns) * 0.45` gave a single-hit Ethics signal `0.55 + (1/8)*0.45 ≈ 0.61` — below the 0.7 HITL threshold, so the router emitted `END_ROUTE`. Legitimate ethics complaints fell off the graph entirely.

Separately, `\bprivacy\b.*\b(?:violation|breach|concern)\b` required the trigger word *after* "privacy" on the same line. "My privacy was violated" matched `\bviolated\b` against a pattern that only accepted `\bviolation\b`, so it scored zero.

**Fixes:**
- Confidence curve: `0.55 + min(best_hits, 3) * 0.15` — diminishing returns, single hit = 0.70, saturates at 1.00.
- Pattern relaxation: `\bviolat(?:ion|ed|ing|e)\b` catches all morphology; `\bprivacy\b` alone counts, no co-occurrence required.

### 2.6 Dual-identity exception classes

The vision test `test_image_path_missing_file_returns_error_message` did:
```python
with patch("...VisionAgent.extract_from_image",
           side_effect=_OCRExtractionError("..."))
```
where `_OCRExtractionError` was a private class defined in the test module. But `vision_agent.py` catches `OCRExtractionError` imported from `patient_advocate.core.exceptions`. Two different classes, same name; the `except` clause didn't match and the exception propagated.

**Fix:** use the exact class the production module imported:
```python
from patient_advocate.agents import vision_agent as _va_mod
with patch(..., side_effect=_va_mod.OCRExtractionError("...")):
```

---

## 3. Ethics

### 3.1 Conception

The ethics layer came out of a design constraint that predates any code: **the system must not be able to harm the patient without the harm being visible and contestable.** That constraint has three corollaries, and each one shaped a piece of the architecture.

**Corollary 1 — Explainability is not optional.**
Every agent output carries a `TransparencyRecord`. "The model said so" is not an acceptable reason to reschedule a chemo appointment. Each record includes `reasoning_chain` (the step-by-step logic) and `scholarly_basis` (the citation grounding the decision — e.g. *"Crawford et al. (2004), NEJM: febrile neutropenia risk peaks at days 7-14 post-infusion for R-CHOP"*). If we cannot cite it, we do not suggest it.

This is the **Glass Box Imperative**: the system's reasoning is physically auditable. The `transparency_report` table is append-only, backed by a server-default `created_at`, indexed for retrieval. `GlassBoxLogger` has no `update` method by design.

**Corollary 2 — Confidence is not truth.**
Every clinical suggestion has a scalar `confidence ∈ [0, 1]`. The threshold `0.7` gates auto-application: below it, the suggestion goes to HITL. The number is not a calibration claim — it is a conservative heuristic derived from how many independent signals agreed. A single matched pattern gets `0.70` (just barely over the line, which means HITL will see it). Three or more independent signals get ≥ `1.00` and pass freely.

This encodes an ethical stance: **the system prefers to over-ask rather than under-ask.** Alarm fatigue in clinical software is real, but the asymmetric cost of a silent wrong rescheduling is higher than the cost of an extra confirmation.

**Corollary 3 — The user must be able to say no, and saying no must train the system.**
Every HITL decision emits an `RLHFFeedbackRecord`. Rejections are not discarded; they are captured with free-text feedback so a human reviewer can later ask *"the model keeps suggesting X and patients keep rejecting it — why?"* This closes the loop from patient preference back to model behavior without giving the model unilateral authority.

### 3.2 Implementation

**`ethics_screener.py`** is a deterministic pattern matcher that runs on every inbound message. It is *not* an LLM. LLMs drift; ethics triage cannot drift. The screener recognizes six complaint families:

- **Harm** — "I felt worse after following your advice", "you caused me to miss my infusion"
- **Privacy** — "my data was shared", "who can see this"
- **Clinical disagreement** — "my oncologist says the opposite"
- **Discrimination** — any mention of protected characteristics in a grievance context
- **Autonomy violation** — "you rescheduled without asking"
- **Access/Equity** — "I can't afford the alternative you suggested"

A hit raises an `EthicsFlag` with a severity, which the router honors absolutely — an ethics flag bypasses the normal confidence gate and goes straight to the oversight queue.

**`ethics_agent.py`** handles the conversation once a flag is raised. It acknowledges the complaint in plain language, never minimizes it, surfaces the relevant `TransparencyRecord`s (so the patient can see *exactly* what the system decided and why), and writes an audit trail of its own actions. It cannot close a complaint — only a human reviewer can.

**`dashboard_router.py`** is the escalation endpoint. Confirmed harms and privacy events do not go back through the normal LangGraph flow; they are written to a separate queue consumed by a human-operated dashboard. The architectural choice is deliberate: automated systems should not be allowed to auto-resolve complaints against themselves.

### 3.3 What the ethics layer explicitly refuses to do

- It will not issue medical advice in an ethics-flagged context. The agent refers out.
- It will not promise confidentiality it cannot guarantee. Every `TransparencyRecord` is persisted, full stop.
- It will not "calm down" a user whose complaint is valid. Tone-softening on a legitimate harm report is a form of gaslighting and is not implemented.
- It will not learn to suppress complaint patterns. The RLHF loop is scoped to suggestion quality, not to complaint routing. Attempting to train away valid ethics triggers would be a safety regression and the pipeline enforces this at the data-partitioning layer.

### 3.4 Open questions

- **Calibration of `confidence`.** The 0.7 threshold is defensible but not empirically tuned. Once we have enough HITL data, we should fit an isotonic regression from classifier-score to actual-correctness and re-derive the cutoff.
- **Ethics pattern coverage.** The six families cover the WHO / IEEE ethics-of-AI canonical harm categories, but real complaints arrive in long tails. A monthly review of the "did not match any ethics pattern but routed to `END_ROUTE`" log is the manual feedback loop until pattern coverage stabilizes.
- **Cross-jurisdictional legal.** The legal pattern set currently assumes a US context (HIPAA, FDA). International deployments will need a locale-aware pattern bank.

---

## 4. Appendix — Test Topology

```
tests/
├── test_agents/           47 tests — Vision OCR + parser
├── test_api/              94 tests — HITL endpoints, stores, service
├── test_calendar_engine/ 104 tests — nadir / fatigue / overlap / recovery
├── test_db/               71 tests — ORM + GlassBoxLogger atomicity
├── test_routing/          97 tests — classifier + master_brain node
├── test_ethics/            —       — scaffolded, implementation pending
├── test_graph/             —       — scaffolded, implementation pending
└── conftest.py            real-module preload + per-collection restore
```
