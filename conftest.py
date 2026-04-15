"""
Ensure src/ is on sys.path and pre-load the real patient_advocate submodules
with real pydantic. Then use a pytest collection hook to re-inject the real
core modules before each test file is imported, because some test files
override sys.modules["patient_advocate.core.*"] with stubs and that pollutes
later test collections that need the real classes (RLHFFeedbackRecord,
AuditWriteError, etc.).
"""
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Pre-import real modules with real pydantic.
import patient_advocate  # noqa: F401
import patient_advocate.core  # noqa: F401
import patient_advocate.core.models  # noqa: F401
import patient_advocate.core.exceptions  # noqa: F401
import patient_advocate.core.config  # noqa: F401
import patient_advocate.routing  # noqa: F401
import patient_advocate.calendar_engine  # noqa: F401
import patient_advocate.db  # noqa: F401
import patient_advocate.db.orm  # noqa: F401 — needs real sqlalchemy
import patient_advocate.db.engine  # noqa: F401
import patient_advocate.db.glass_box_logger  # noqa: F401
import patient_advocate.api  # noqa: F401
import patient_advocate.agents  # noqa: F401
import patient_advocate.ethics  # noqa: F401
import patient_advocate.graph  # noqa: F401

# Snapshot real modules so we can restore them between test-file imports.
_REAL_MODULES = {
    name: sys.modules[name]
    for name in [
        "patient_advocate",
        "patient_advocate.core",
        "patient_advocate.core.models",
        "patient_advocate.core.exceptions",
        "patient_advocate.core.config",
        "patient_advocate.db",
        "patient_advocate.db.orm",
        "patient_advocate.db.engine",
        "patient_advocate.db.glass_box_logger",
    ]
}


def pytest_collectstart(collector):
    """Before each test module is collected, restore real core modules and
    clear any stubbed sqlalchemy submodules so real sqlalchemy re-imports fresh."""
    for name, mod in _REAL_MODULES.items():
        sys.modules[name] = mod
    # Purge stubbed sqlalchemy submodules (identified by missing __file__).
    for name in list(sys.modules):
        if name.startswith("sqlalchemy") or name.startswith("fastapi") \
                or name.startswith("pydantic") or name.startswith("alembic") \
                or name.startswith("langchain_core") or name.startswith("langgraph"):
            mod = sys.modules[name]
            if getattr(mod, "__file__", None) is None:
                del sys.modules[name]
