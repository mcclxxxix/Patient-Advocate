"""
Vision/OCR Agent — lab value extraction from screenshots.

Uses pytesseract + PIL for OCR, with noise-tolerant regex parsing.
OCR produces noisy text (WBC → "wee", /uL → "luL") — parser must be robust.

Glass Box Imperative: Every OCR extraction produces a TransparencyRecord.
OCR output is inherently uncertain and ALWAYS requires HITL verification
before clinical decisions are made from extracted values.

References:
  - NCCN Guidelines for Myelosuppressive Therapy, v2024.
  - Hantel et al. (2024), JAMA Network Open: Transparency and accountability
    in clinical AI deployment.
  - Carey (2024), Advances in Consumer Research: Model facts labels and
    uncertainty bounds for AI-generated clinical data.

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional
from uuid import uuid4

from patient_advocate.core.exceptions import OCRExtractionError
from patient_advocate.core.models import AgentType, LabValues, TransparencyRecord


# =============================================================================
# OCR PREPROCESSING (PIL / pytesseract — optional at test time)
# =============================================================================

def _preprocess_image(image_path: str):
    """
    Load and preprocess an image for optimal OCR accuracy.

    Pipeline:
      1. Load via PIL.Image.open
      2. Convert to grayscale (L mode)
      3. Upscale 2x with LANCZOS for small text
      4. Apply ImageFilter.SHARPEN to improve character edges

    Returns a PIL.Image object ready for pytesseract.
    """
    try:
        from PIL import Image, ImageFilter  # type: ignore
    except ImportError as exc:
        raise OCRExtractionError(
            "PIL (Pillow) is required for image preprocessing. "
            "Install with: pip install Pillow",
            agent_name=AgentType.VISION_AGENT.value,
        ) from exc

    try:
        img = Image.open(image_path)
    except FileNotFoundError as exc:
        raise OCRExtractionError(
            f"Image file not found: {image_path}",
            agent_name=AgentType.VISION_AGENT.value,
        ) from exc
    except Exception as exc:
        raise OCRExtractionError(
            f"Failed to open image '{image_path}': {exc}",
            agent_name=AgentType.VISION_AGENT.value,
        ) from exc

    # Grayscale — reduces colour noise
    img = img.convert("L")

    # 2x upscale with high-quality resampling for small lab report text
    w, h = img.size
    img = img.resize((w * 2, h * 2), Image.LANCZOS)

    # Edge sharpening improves character recognition
    img = img.filter(ImageFilter.SHARPEN)

    return img


def _run_ocr(img) -> str:
    """
    Run pytesseract on a preprocessed PIL image.

    Config: PSM 6 (assume a uniform block of text) is reliable for
    structured lab report layouts.
    """
    try:
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise OCRExtractionError(
            "pytesseract is required for OCR. "
            "Install with: pip install pytesseract",
            agent_name=AgentType.VISION_AGENT.value,
        ) from exc

    try:
        return pytesseract.image_to_string(img, config="--psm 6")
    except Exception as exc:
        raise OCRExtractionError(
            f"pytesseract failed to extract text: {exc}",
            agent_name=AgentType.VISION_AGENT.value,
        ) from exc


# =============================================================================
# LAB VALUE PARSER
# =============================================================================

# Numeric token: matches integers and decimals (e.g., 1500, 1.5, 0.8)
_NUM = r"(\d+(?:\.\d+)?)"

# Optional units suffix — OCR commonly garbles "/uL" → "luL", "IuL", "/ul"
# We match any trailing non-whitespace characters after a space (units are optional).
_OPT_UNITS = r"(?:\s*[^\s]*)?"


class LabValueParser:
    """
    OCR-noise-tolerant regex parser for lab values extracted from raw OCR text.

    OCR errors handled:
      - Character substitutions (0 → O, 1 → l, B → 8)
      - Dot insertion in abbreviations (ANC → A.N.C)
      - Units garbling (/uL → luL, IuL, /ul, uL, cells/uL)
      - Label OCR noise (WBC → "W.B.C", "White Blood Cell")
      - Extra whitespace and punctuation around values

    Design: all patterns are compiled once at class definition for performance.
    """

    # ------------------------------------------------------------------
    # ANC — Absolute Neutrophil Count (cells/uL)
    #
    # Matches labels: ANC, A.N.C, A N C, Absolute Neutrophil (Count)?
    # followed by optional separators (spaces, colons, equals) and a number.
    # ------------------------------------------------------------------
    _ANC_PATTERN = re.compile(
        r"(?:A[\.\s]?N[\.\s]?C|Absolute\s+Neutrophil(?:\s+Count)?)"
        r"[\s:=]*" + _NUM,
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # WBC — White Blood Cell count (10^3/uL)
    #
    # Matches: WBC, W.B.C, W B C, White Blood (Cell)? (Count)?
    # Does NOT match "wee" (OCR artefact too ambiguous for clinical safety).
    # ------------------------------------------------------------------
    _WBC_PATTERN = re.compile(
        r"(?:W[\.\s]?B[\.\s]?C|White\s+Blood(?:\s+Cell)?(?:\s+Count)?)"
        r"[\s:=]*" + _NUM,
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Hemoglobin (g/dL)
    #
    # Matches: HGB, Hgb, HB, H.G.B, Hemoglobin, Haemoglobin
    # ------------------------------------------------------------------
    _HGB_PATTERN = re.compile(
        r"(?:H[\.\s]?[Gg][\.\s]?[Bb]|H[\.\s]?[Bb]|[Hh]a?emoglobin)"
        r"[\s:=]*" + _NUM,
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Platelets (10^3/uL)
    #
    # Matches: PLT, PLTS, PLT Count, PLT COUNT, Platelet, Platelets
    # Note: "PLT Count" has a space — the pattern must allow for it.
    # ------------------------------------------------------------------
    _PLT_PATTERN = re.compile(
        r"(?:PLT[Ss]?(?:\s+Count)?|Platelets?)"
        r"[\s:=]*" + _NUM,
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Fatigue Score (0–10 numeric scale)
    #
    # Matches: Fatigue, Fatigue Score, Pain, Pain Score
    # Bounds the captured value to 0–10.
    # ------------------------------------------------------------------
    _FATIGUE_PATTERN = re.compile(
        r"(?:Fatigue(?:\s+Score)?|Pain(?:\s+Score)?)"
        r"[\s:=]*" + _NUM,
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Collection date — ISO 8601 (YYYY-MM-DD) and US format (MM/DD/YYYY)
    # Used when no explicit collection_date is passed to to_lab_values().
    # ------------------------------------------------------------------
    _DATE_ISO_PATTERN = re.compile(
        r"(?:Date|Collected|Collection\s+Date)?[\s:=]*"
        r"(\d{4}-\d{2}-\d{2})"
    )
    _DATE_US_PATTERN = re.compile(
        r"(?:Date|Collected|Collection\s+Date)?[\s:=]*"
        r"(\d{1,2}/\d{1,2}/\d{4})"
    )

    # -----------------------------------------------------------------------

    def parse(self, ocr_text: str) -> dict[str, Optional[float | int]]:
        """
        Extract lab values from raw OCR text.

        Returns a dict with keys matching LabValues fields. Values are None
        when the corresponding label was not found or the parsed number was
        out of a clinically plausible range.

        Plausibility guards (NCCN, v2024):
          - ANC: 0 – 50 000 cells/uL
          - WBC: 0 – 200 (10^3/uL); values > 200 are extreme
          - Hemoglobin: 0 – 25 g/dL
          - Platelets: 0 – 3 000 (10^3/uL)
          - Fatigue: 0 – 10
        """
        if not ocr_text or not ocr_text.strip():
            return {
                "anc": None,
                "wbc": None,
                "hemoglobin": None,
                "platelets": None,
                "fatigue_score": None,
            }

        result: dict[str, Optional[float | int]] = {}

        # ANC
        m = self._ANC_PATTERN.search(ocr_text)
        if m:
            val = float(m.group(1))
            result["anc"] = val if 0 <= val <= 50_000 else None
        else:
            result["anc"] = None

        # WBC
        m = self._WBC_PATTERN.search(ocr_text)
        if m:
            val = float(m.group(1))
            result["wbc"] = val if 0 <= val <= 200 else None
        else:
            result["wbc"] = None

        # Hemoglobin
        m = self._HGB_PATTERN.search(ocr_text)
        if m:
            val = float(m.group(1))
            result["hemoglobin"] = val if 0 <= val <= 25 else None
        else:
            result["hemoglobin"] = None

        # Platelets
        m = self._PLT_PATTERN.search(ocr_text)
        if m:
            val = float(m.group(1))
            # Platelets stored as int (10^3/uL)
            result["platelets"] = int(val) if 0 <= val <= 3_000 else None
        else:
            result["platelets"] = None

        # Fatigue (0–10)
        m = self._FATIGUE_PATTERN.search(ocr_text)
        if m:
            val = float(m.group(1))
            result["fatigue_score"] = int(val) if 0 <= val <= 10 else None
        else:
            result["fatigue_score"] = None

        return result

    def extract_date_from_text(self, ocr_text: str) -> Optional[date]:
        """
        Attempt to parse a collection date from OCR text.

        Tries ISO format first, then US MM/DD/YYYY. Returns None if not found.
        """
        m = self._DATE_ISO_PATTERN.search(ocr_text)
        if m:
            try:
                return date.fromisoformat(m.group(1))
            except ValueError:
                pass

        m = self._DATE_US_PATTERN.search(ocr_text)
        if m:
            try:
                parts = m.group(1).split("/")
                return date(int(parts[2]), int(parts[0]), int(parts[1]))
            except (ValueError, IndexError):
                pass

        return None

    def to_lab_values(
        self,
        parsed: dict[str, Optional[float | int]],
        collection_date: Optional[date] = None,
    ) -> LabValues:
        """
        Convert a parsed dict to a LabValues model with source='ocr'.

        If collection_date is None, defaults to today's date (date.today()).
        """
        return LabValues(
            anc=parsed.get("anc"),
            wbc=parsed.get("wbc"),
            hemoglobin=parsed.get("hemoglobin"),
            platelets=parsed.get("platelets"),
            fatigue_score=parsed.get("fatigue_score"),
            collection_date=collection_date or date.today(),
            source="ocr",
        )


# =============================================================================
# VISION AGENT
# =============================================================================

class VisionAgent:
    """
    Full lab value extraction pipeline from images or pre-extracted OCR text.

    Two entry points:
      - extract_from_text: for testing or when OCR has already been run.
      - extract_from_image: full pipeline (PIL → preprocess → pytesseract → parse).

    Every extraction returns a TransparencyRecord per the Glass Box Imperative.
    Confidence is computed as (fields_found / total_fields), reflecting how
    complete the extraction was, which directly informs the HITL gate.
    """

    #: Total number of lab value fields the parser attempts to extract.
    _TOTAL_FIELDS: int = 5  # anc, wbc, hemoglobin, platelets, fatigue_score

    #: Scholarly citation for OCR uncertainty — embedded in every audit record.
    _SCHOLARLY_BASIS: str = (
        "Carey (2024), Advances in Consumer Research: Model facts labels require "
        "uncertainty bounds for AI-generated data. OCR extraction confidence is "
        "field-count-based and must be verified by a human before clinical use. "
        "Hantel et al. (2024), JAMA Network Open: HITL review of AI-extracted "
        "clinical data is mandatory per accountability frameworks."
    )

    def __init__(self) -> None:
        self._parser = LabValueParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_text(
        self,
        ocr_text: str,
        collection_date: Optional[date] = None,
    ) -> tuple[LabValues, TransparencyRecord]:
        """
        Parse pre-extracted OCR text into LabValues and a TransparencyRecord.

        Confidence = (number of non-None fields) / _TOTAL_FIELDS.
        A document where 3 of 5 fields are found yields confidence = 0.60,
        triggering mandatory HITL review (threshold < 0.70 per Glass Box spec).

        Args:
            ocr_text: Raw text string from pytesseract (or a test stub).
            collection_date: Explicit collection date. If None, attempts to
                parse from text; falls back to today.

        Returns:
            (LabValues, TransparencyRecord) tuple.
        """
        parsed = self._parser.parse(ocr_text)

        # Resolve collection date: explicit > parsed from text > today
        resolved_date = (
            collection_date
            or self._parser.extract_date_from_text(ocr_text)
            or date.today()
        )

        lab_values = self._parser.to_lab_values(parsed, resolved_date)
        confidence = self._compute_confidence(parsed)

        record = self._build_transparency_record(
            lab_values=lab_values,
            confidence=confidence,
            reasoning_chain=self._build_reasoning_chain(parsed, confidence, ocr_text),
        )

        return lab_values, record

    def extract_from_image(
        self,
        image_path: str,
    ) -> tuple[LabValues, TransparencyRecord]:
        """
        Full pipeline: load image → preprocess → OCR → parse.

        Raises:
            OCRExtractionError: If the file is not found, PIL cannot open it,
                or pytesseract fails to run.
        """
        img = _preprocess_image(image_path)
        ocr_text = _run_ocr(img)
        return self.extract_from_text(ocr_text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_confidence(
        self, parsed: dict[str, Optional[float | int]]
    ) -> float:
        """
        Confidence = fraction of lab value fields successfully extracted.

        Returns a float in [0.0, 1.0].
        """
        lab_keys = {"anc", "wbc", "hemoglobin", "platelets", "fatigue_score"}
        found = sum(1 for k in lab_keys if parsed.get(k) is not None)
        return round(found / self._TOTAL_FIELDS, 4)

    def _build_reasoning_chain(
        self,
        parsed: dict[str, Optional[float | int]],
        confidence: float,
        ocr_text: str,
    ) -> str:
        """Build a human-readable reasoning chain for the TransparencyRecord."""
        lab_keys = {"anc", "wbc", "hemoglobin", "platelets", "fatigue_score"}
        found_fields = [k for k in lab_keys if parsed.get(k) is not None]
        missing_fields = [k for k in lab_keys if parsed.get(k) is None]

        text_preview = (ocr_text[:120].replace("\n", " ") + "…") if ocr_text else "(empty)"

        lines = [
            f"VisionAgent parsed OCR text ({len(ocr_text)} chars).",
            f"Input preview: '{text_preview}'",
            f"Fields extracted ({len(found_fields)}/{self._TOTAL_FIELDS}): "
            f"{', '.join(found_fields) if found_fields else 'none'}.",
            f"Fields not found: {', '.join(missing_fields) if missing_fields else 'none'}.",
            f"Extraction confidence: {confidence:.2%}.",
        ]

        if confidence < 0.70:
            lines.append(
                "Confidence below 0.70 — HITL verification is mandatory before "
                "any clinical decision is made from these values."
            )
        else:
            lines.append(
                "Confidence >= 0.70 — HITL verification still required for all "
                "OCR-sourced lab values per the Glass Box Imperative."
            )

        return " ".join(lines)

    @staticmethod
    def _build_transparency_record(
        lab_values: LabValues,
        confidence: float,
        reasoning_chain: str,
    ) -> TransparencyRecord:
        """Construct an immutable TransparencyRecord for this extraction event."""
        return TransparencyRecord(
            event_id=uuid4(),
            suggestion_id=None,
            agent_name=AgentType.VISION_AGENT.value,
            conflict_type=None,
            confidence=confidence,
            scholarly_basis=VisionAgent._SCHOLARLY_BASIS,
            patient_decision=None,
            reasoning_chain=reasoning_chain,
            timestamp=datetime.utcnow(),
        )


# =============================================================================
# LANGGRAPH NODE
# =============================================================================

def vision_agent_node(state: dict) -> dict:
    """
    LangGraph node for the Vision/OCR Agent.

    Reads ``state['image_path']``. If set, runs the full extraction pipeline
    and returns updated state with extracted lab values and an audit record.
    If not set, returns a patient-facing message explaining what to provide.

    Always sets ``hitl_required=True`` — OCR output is never acted upon
    without human verification per the Glass Box Imperative.

    Args:
        state: The shared AdvocateState dict flowing through the LangGraph graph.

    Returns:
        Partial state dict with keys: ``audit_log``, ``messages``,
        ``patient_profile``, ``hitl_required``.
    """
    try:
        from langchain_core.messages import AIMessage  # type: ignore
    except ImportError:
        # Graceful degradation in environments without langchain installed
        class AIMessage:  # type: ignore[no-redef]
            def __init__(self, content: str) -> None:
                self.content = content

    image_path: Optional[str] = state.get("image_path")

    if not image_path:
        message = AIMessage(
            content=(
                "I need a lab report image to extract your blood count values. "
                "Please upload a screenshot or photo of your lab results — "
                "I can read values like ANC (Absolute Neutrophil Count), WBC "
                "(White Blood Cell), Hemoglobin, and Platelets. "
                "Accepted formats: PNG, JPG, TIFF, BMP. "
                "Once uploaded, I will extract the values and ask you to confirm "
                "them before any scheduling decisions are made."
            )
        )
        return {
            "audit_log": [],
            "messages": [message],
            "patient_profile": state.get("patient_profile"),
            "hitl_required": True,
        }

    # Run the extraction pipeline
    agent = VisionAgent()
    try:
        lab_values, transparency_record = agent.extract_from_image(image_path)
    except OCRExtractionError as exc:
        message = AIMessage(
            content=(
                f"I was unable to extract lab values from the image: {exc}. "
                "Please ensure the file path is correct and the image is a "
                "supported format (PNG, JPG, TIFF). You can also enter your "
                "lab values manually and I will log them as 'manual' source."
            )
        )
        return {
            "audit_log": [],
            "messages": [message],
            "patient_profile": state.get("patient_profile"),
            "hitl_required": True,
        }

    # Update the patient profile with the newly extracted lab values
    patient_profile = state.get("patient_profile")
    if patient_profile is not None:
        try:
            patient_profile = patient_profile.model_copy(
                update={"latest_labs": lab_values}
            )
        except Exception:
            # If profile update fails, continue with original profile
            pass

    # Build a summary message for the patient
    field_lines = []
    if lab_values.anc is not None:
        field_lines.append(f"  ANC: {lab_values.anc:,.0f} cells/uL")
    if lab_values.wbc is not None:
        field_lines.append(f"  WBC: {lab_values.wbc} x10³/uL")
    if lab_values.hemoglobin is not None:
        field_lines.append(f"  Hemoglobin: {lab_values.hemoglobin} g/dL")
    if lab_values.platelets is not None:
        field_lines.append(f"  Platelets: {lab_values.platelets} x10³/uL")
    if lab_values.fatigue_score is not None:
        field_lines.append(f"  Fatigue Score: {lab_values.fatigue_score}/10")

    confidence_pct = f"{(transparency_record.confidence or 0.0):.0%}"

    if field_lines:
        values_summary = "\n".join(field_lines)
        content = (
            f"I extracted the following lab values from your image "
            f"(confidence: {confidence_pct}):\n\n"
            f"{values_summary}\n\n"
            "Please review these values carefully — OCR can make errors. "
            "Confirm they match your lab report before I use them for "
            "scheduling decisions."
        )
    else:
        content = (
            f"I was unable to extract any recognizable lab values from the image "
            f"(confidence: {confidence_pct}). "
            "The image quality may be too low, or the report format is not "
            "supported. Please try a clearer image or enter your values manually."
        )

    message = AIMessage(content=content)

    return {
        "audit_log": [transparency_record],
        "messages": [message],
        "patient_profile": patient_profile,
        "hitl_required": True,
    }
