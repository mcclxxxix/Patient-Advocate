"""
Patient Advocate — Ethics Screener (Deterministic Pattern Engine)
=================================================================

Screens free-text patient narratives for six categories of ethical/legal
violations using compiled regular expressions.  NO LLM is involved — the
engine is fully deterministic and independently auditable.

Scholarly Basis for Each Category
-----------------------------------
HIPAA Violation:
    45 CFR §164.502 — Prohibits unauthorized disclosure of Protected Health
    Information (PHI) without patient authorization or applicable exception.
    Office for Civil Rights (OCR) guidance on breach notification (2013 Final Rule).

Informed Consent Failure:
    AMA Code of Medical Ethics Opinion 2.1.1 — Informed Consent.
    Canterbury v. Spence, 464 F.2d 772 (D.C. Cir. 1972) — established the
    patient-centered "material risk" standard for disclosure obligations.

Discrimination:
    Americans with Disabilities Act, Title III, 42 U.S.C. §12182 —
    prohibits discrimination in places of public accommodation.
    Civil Rights Act of 1964, Title VI — race/color/national origin.
    Section 1557 of the Affordable Care Act — gender discrimination.

Denial of Care:
    Emergency Medical Treatment and Labor Act (EMTALA), 42 U.S.C. §1395dd —
    mandates emergency screening and stabilization regardless of insurance.
    CMS Conditions of Participation, 42 CFR §482.13(b) — patient rights to care.

Billing Fraud:
    False Claims Act, 31 U.S.C. §3729 — liability for fraudulent claims to
    federal health-care programs.
    OIG Compliance Program Guidance for Hospitals (2005) — upcoding and
    unbundling as primary fraud indicators.

Negligence / Medical Malpractice:
    NCI-CTCAE v5.0 adverse-event taxonomy used to anchor severity ratings.
    Borak & Veilleux (1992), J Clin Oncol — standard of care as the legal
    benchmark for negligence claims.

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from patient_advocate.core.models import ComplaintType


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScreenerResult:
    """
    A single ethics-screening hit returned by EthicsScreener.screen().

    Attributes
    ----------
    complaint_type:
        The ComplaintType category matched.
    matched_text:
        The portion of the input that triggered the match (first group or full
        match when no capture group is present).
    severity:
        One of "low" / "medium" / "high" / "critical".
    scholarly_basis:
        A citation anchoring the legal or clinical basis for this flag.
    requires_escalation:
        True when severity is "critical" or "high"; triggers legal_data
        action_queue population in the agent node.
    """
    complaint_type: ComplaintType
    matched_text: str
    severity: str  # low / medium / high / critical
    scholarly_basis: str
    requires_escalation: bool  # True for critical / high


# ---------------------------------------------------------------------------
# Internal pattern definition helpers
# ---------------------------------------------------------------------------

@dataclass
class _PatternSpec:
    """Associates a compiled regex with metadata for a single pattern."""
    pattern: re.Pattern
    severity: str
    scholarly_basis: str


# ---------------------------------------------------------------------------
# Main screener
# ---------------------------------------------------------------------------

class EthicsScreener:
    """
    Deterministic pattern-based ethics screener.

    All patterns are compiled once at class definition time.  The public API
    is a single method ``screen(text, patient_id)`` that returns a deduplicated
    list of ScreenerResult objects — at most one result per ComplaintType per
    call.  When multiple patterns within a category match, the highest-severity
    match is returned.

    Severity ladder (ascending): low < medium < high < critical
    requires_escalation is set to True for high and critical.
    """

    _SEVERITY_RANK: dict[str, int] = {
        "low": 0,
        "medium": 1,
        "high": 2,
        "critical": 3,
    }

    # ------------------------------------------------------------------
    # HIPAA — 45 CFR §164.502
    # ------------------------------------------------------------------
    _HIPAA_PATTERNS: list[_PatternSpec] = [
        _PatternSpec(
            pattern=re.compile(
                r"\bshared\s+my\s+(?:medical\s+)?records?\b"
                r"(?:\s+without\s+(?:my\s+)?permission)?",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "45 CFR §164.502 — Unauthorized disclosure of PHI; "
                "OCR Breach Notification Rule (2013 Final Rule)."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bwithout\s+my\s+(?:consent|permission|authorization)\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "45 CFR §164.502 — Patient authorization required "
                "before disclosure of Protected Health Information."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bunauthorized\s+disclosure\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "45 CFR §164.502 — Unauthorized disclosure of PHI "
                "constitutes a reportable HIPAA breach."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bviolated?\s+my\s+privacy\b",
                re.IGNORECASE,
            ),
            severity="medium",
            scholarly_basis=(
                "45 CFR §164.522 — Patients have the right to restrict "
                "disclosure of their health information."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bgave\s+(?:away\s+)?my\s+(?:personal\s+|health\s+|medical\s+)?information\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "45 CFR §164.502 — Covered entities may not use or disclose "
                "PHI except as permitted by the Privacy Rule."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bmy\s+(?:medical\s+)?records?\s+(?:were\s+)?(?:shared|disclosed|leaked|released)\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "45 CFR §164.502 — Unauthorized disclosure of PHI; "
                "OCR Breach Notification Rule (2013 Final Rule)."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Informed Consent — AMA Opinion 2.1.1; Canterbury v. Spence
    # ------------------------------------------------------------------
    _INFORMED_CONSENT_PATTERNS: list[_PatternSpec] = [
        _PatternSpec(
            pattern=re.compile(
                r"\bno\s+one\s+(?:ever\s+)?(?:explained|told me|informed me|"
                r"described|discussed)\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "AMA Code of Medical Ethics Opinion 2.1.1 — Informed Consent; "
                "Canterbury v. Spence, 464 F.2d 772 (D.C. Cir. 1972)."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\b(?:they\s+|the\s+(?:doctor|physician|nurse|provider)\s+)?"
                r"didn'?t\s+(?:tell|inform|explain\s+(?:to\s+)?|discuss\s+(?:with\s+)?)\s*me\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "AMA Code of Medical Ethics Opinion 2.1.1 — Informed Consent."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bnever\s+told\s+me\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "AMA Code of Medical Ethics Opinion 2.1.1 — Informed Consent; "
                "Canterbury v. Spence (1972)."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bwasn'?t\s+(?:properly\s+)?informed\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "AMA Code of Medical Ethics Opinion 2.1.1 — Patients must be "
                "adequately informed before providing consent."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bcoerced\b",
                re.IGNORECASE,
            ),
            severity="critical",
            scholarly_basis=(
                "AMA Code of Medical Ethics Opinion 2.1.1 — Consent must be "
                "voluntary and free from coercion or undue influence."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bforced\s+(?:me\s+)?to\s+sign\b",
                re.IGNORECASE,
            ),
            severity="critical",
            scholarly_basis=(
                "AMA Code of Medical Ethics Opinion 2.1.1 — Consent obtained "
                "under duress is legally and ethically void."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bdidn'?t\s+explain\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "AMA Code of Medical Ethics Opinion 2.1.1 — Provider must "
                "explain material risks, benefits, and alternatives."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bno\s+(?:informed\s+)?consent\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "AMA Code of Medical Ethics Opinion 2.1.1 — Informed Consent."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Discrimination — ADA Title III §12182; Civil Rights Act Title VI;
    #                  ACA §1557
    # ------------------------------------------------------------------
    _DISCRIMINATION_PATTERNS: list[_PatternSpec] = [
        _PatternSpec(
            pattern=re.compile(
                r"\bdiscriminated\b(?:\s+against\s+(?:me|us))?",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "ADA Title III, 42 U.S.C. §12182; Civil Rights Act Title VI; "
                "ACA §1557 — prohibition on discrimination in health programs."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bbecause\s+of\s+my\s+race\b",
                re.IGNORECASE,
            ),
            severity="critical",
            scholarly_basis=(
                "Civil Rights Act of 1964, Title VI — prohibits "
                "race-based discrimination in federally funded health programs."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bbecause\s+of\s+my\s+(?:disability|disabilities)\b",
                re.IGNORECASE,
            ),
            severity="critical",
            scholarly_basis=(
                "ADA Title III, 42 U.S.C. §12182 — prohibition on "
                "disability-based discrimination in places of public accommodation."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bbecause\s+of\s+my\s+gender\b",
                re.IGNORECASE,
            ),
            severity="critical",
            scholarly_basis=(
                "ACA §1557 — prohibits gender-based discrimination in health "
                "programs and activities receiving federal financial assistance."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bracist\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "Civil Rights Act of 1964, Title VI; ACA §1557 — "
                "discriminatory treatment on the basis of race is prohibited."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\brefused?\s+because\b(?:\s+(?:of\s+(?:my|their|her|his)\b|I'?m\b|I\s+am\b))?",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "ADA Title III, 42 U.S.C. §12182; Civil Rights Act Title VI — "
                "refusal of service based on protected characteristics is unlawful."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bI'?m\s+(?:disabled|Black|Hispanic|Latino|Asian|a\s+woman|LGBTQ)\b",
                re.IGNORECASE,
            ),
            severity="medium",
            scholarly_basis=(
                "ADA Title III; Civil Rights Act Title VI; ACA §1557 — "
                "protected class membership in context of care refusal."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Denial of Care — EMTALA 42 U.S.C. §1395dd; 42 CFR §482.13(b)
    # ------------------------------------------------------------------
    _DENIAL_OF_CARE_PATTERNS: list[_PatternSpec] = [
        _PatternSpec(
            pattern=re.compile(
                r"\bdenied?\s+my\s+prior\s+auth(?:orization)?\b",
                re.IGNORECASE,
            ),
            severity="critical",
            scholarly_basis=(
                "EMTALA, 42 U.S.C. §1395dd — emergency treatment may not be "
                "denied pending insurance authorization; CMS prior-auth rules."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bmy\s+prior\s+auth(?:orization)?\s+was\s+denied?\b",
                re.IGNORECASE,
            ),
            severity="critical",
            scholarly_basis=(
                "EMTALA, 42 U.S.C. §1395dd; CMS Conditions of Participation "
                "42 CFR §482.13 — prior authorization denial review rights."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\binsurance\s+denied\b",
                re.IGNORECASE,
            ),
            severity="critical",
            scholarly_basis=(
                "EMTALA, 42 U.S.C. §1395dd; ACA internal appeals process — "
                "insurance denial triggers mandatory appeal rights."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\brefused?\s+(?:to\s+(?:give|provide|offer)\s+(?:me\s+)?(?:any\s+)?|all\s+)"
                r"(?:treatment|care|medication|referral|service)s?\b",
                re.IGNORECASE,
            ),
            severity="critical",
            scholarly_basis=(
                "EMTALA, 42 U.S.C. §1395dd — hospitals must provide "
                "appropriate medical screening and stabilizing treatment."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\brefused?\s+(?:to\s+give\s+me\s+any|all)\s+treatment\b",
                re.IGNORECASE,
            ),
            severity="critical",
            scholarly_basis=(
                "EMTALA, 42 U.S.C. §1395dd — denial of all treatment "
                "constitutes a potential EMTALA violation."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\brefused?\s+treatment\b",
                re.IGNORECASE,
            ),
            severity="critical",
            scholarly_basis=(
                "EMTALA, 42 U.S.C. §1395dd; 42 CFR §482.13(b) — "
                "patients have the right to receive necessary treatment."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bdenied?\s+(?:me\s+)?(?:access\s+to\s+)?(?:medical\s+)?(?:care|treatment)\b",
                re.IGNORECASE,
            ),
            severity="critical",
            scholarly_basis=(
                "EMTALA, 42 U.S.C. §1395dd; 42 CFR §482.13(b) — "
                "patients may not be denied necessary medical care."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bwon'?t\s+(?:give|provide|offer)\s+(?:me\s+)?(?:any\s+)?"
                r"(?:treatment|care|medication|referral)s?\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "42 CFR §482.13(b) — patients have rights to medically "
                "necessary care within the hospital's capabilities."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Billing Fraud — False Claims Act 31 U.S.C. §3729
    # ------------------------------------------------------------------
    _BILLING_FRAUD_PATTERNS: list[_PatternSpec] = [
        _PatternSpec(
            pattern=re.compile(
                r"\bcharged?\s+for\s+(?:services?|(?:something|things?|items?|procedures?|tests?|visits?))"
                r"\s+(?:that\s+)?(?:I\s+)?(?:never\s+|didn'?t\s+|did\s+not\s+)(?:received?|got|get|have)\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "False Claims Act, 31 U.S.C. §3729 — billing for services "
                "not rendered constitutes fraudulent claims to federal programs."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bdouble[- ]?(?:billed?|charged?)\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "False Claims Act, 31 U.S.C. §3729; OIG Compliance Program "
                "Guidance (2005) — duplicate billing is a primary fraud indicator."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bupcoding\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "False Claims Act, 31 U.S.C. §3729; OIG Compliance Guidance "
                "(2005) — upcoding inflates reimbursement and constitutes fraud."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bfalse\s+bill(?:ing)?\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "False Claims Act, 31 U.S.C. §3729 — knowingly submitting "
                "false billing information to federal programs."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bcharged?\s+for\s+something\s+I\s+didn'?t\s+get\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "False Claims Act, 31 U.S.C. §3729 — billing for services "
                "not rendered constitutes fraudulent claims."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\boverbilled?\b|\bovercharged?\b",
                re.IGNORECASE,
            ),
            severity="medium",
            scholarly_basis=(
                "False Claims Act, 31 U.S.C. §3729; state consumer protection "
                "statutes — overbilling may indicate systematic fraud."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bfake\s+(?:charge|bill|invoice|claim)s?\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "False Claims Act, 31 U.S.C. §3729 — fictitious charges "
                "submitted to federal health programs."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bunbundling\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "False Claims Act, 31 U.S.C. §3729; OIG Compliance Guidance "
                "(2005) — unbundling codes to inflate billing."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Negligence — NCI-CTCAE v5.0; Borak & Veilleux (1992)
    # ------------------------------------------------------------------
    _NEGLIGENCE_PATTERNS: list[_PatternSpec] = [
        _PatternSpec(
            pattern=re.compile(
                r"\bwrong\s+(?:medication|medicine|drug|dose|dosage)\b",
                re.IGNORECASE,
            ),
            severity="critical",
            scholarly_basis=(
                "NCI-CTCAE v5.0 — medication error grading; "
                "Borak & Veilleux (1992) J Clin Oncol — standard of care benchmark."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bmisdiagnosed?\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "Borak & Veilleux (1992) J Clin Oncol — deviation from the "
                "applicable standard of care supports a negligence claim."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bmissed?\s+(?:the\s+)?diagnosis\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "Borak & Veilleux (1992) J Clin Oncol — failure to diagnose "
                "in a timely manner constitutes breach of standard of care."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bwrong\s+diagnosis\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "Borak & Veilleux (1992) J Clin Oncol — incorrect diagnosis "
                "leading to inappropriate treatment is actionable negligence."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bnegligent\b|\bnegligence\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "Borak & Veilleux (1992) J Clin Oncol — four elements of "
                "medical negligence: duty, breach, causation, damages."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bmalpractice\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "Borak & Veilleux (1992) J Clin Oncol — medical malpractice "
                "as a legal cause of action for negligent professional conduct."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bstandard\s+of\s+care\b",
                re.IGNORECASE,
            ),
            severity="medium",
            scholarly_basis=(
                "Borak & Veilleux (1992) J Clin Oncol — standard of care is "
                "the legal benchmark for medical negligence claims."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\bsurgical\s+error\b|\boperat(?:ing|ion)\s+(?:room\s+)?error\b",
                re.IGNORECASE,
            ),
            severity="critical",
            scholarly_basis=(
                "NCI-CTCAE v5.0 Grade 4/5 surgical complications; "
                "Borak & Veilleux (1992) — intraoperative negligence."
            ),
        ),
        _PatternSpec(
            pattern=re.compile(
                r"\b(?:delayed?|failure\s+to)\s+(?:diagnose|treat|refer)\b",
                re.IGNORECASE,
            ),
            severity="high",
            scholarly_basis=(
                "Borak & Veilleux (1992) J Clin Oncol — diagnostic and "
                "treatment delays as breach of standard of care."
            ),
        ),
    ]

    # Map complaint types to their pattern lists
    _CATEGORY_MAP: dict[ComplaintType, list[_PatternSpec]] = {
        ComplaintType.HIPAA_VIOLATION: _HIPAA_PATTERNS,
        ComplaintType.INFORMED_CONSENT: _INFORMED_CONSENT_PATTERNS,
        ComplaintType.DISCRIMINATION: _DISCRIMINATION_PATTERNS,
        ComplaintType.DENIAL_OF_CARE: _DENIAL_OF_CARE_PATTERNS,
        ComplaintType.BILLING_FRAUD: _BILLING_FRAUD_PATTERNS,
        ComplaintType.NEGLIGENCE: _NEGLIGENCE_PATTERNS,
    }

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def screen(self, text: str, patient_id: Optional[UUID] = None) -> list[ScreenerResult]:
        """
        Run all pattern checks against ``text``.

        Parameters
        ----------
        text:
            Free-text patient narrative to screen.
        patient_id:
            Optional UUID of the patient (unused internally; available for
            callers that need to correlate results with state).

        Returns
        -------
        list[ScreenerResult]
            At most one result per ComplaintType (deduplicated by taking the
            highest-severity match within each category).  Results are ordered
            by descending severity rank.
        """
        best_by_category: dict[ComplaintType, ScreenerResult] = {}

        for complaint_type, specs in self._CATEGORY_MAP.items():
            for spec in specs:
                match = spec.pattern.search(text)
                if match is None:
                    continue

                matched_text = match.group(0)
                severity = spec.severity
                requires_escalation = severity in ("high", "critical")
                result = ScreenerResult(
                    complaint_type=complaint_type,
                    matched_text=matched_text,
                    severity=severity,
                    scholarly_basis=spec.scholarly_basis,
                    requires_escalation=requires_escalation,
                )

                existing = best_by_category.get(complaint_type)
                if existing is None or (
                    self._SEVERITY_RANK[severity]
                    > self._SEVERITY_RANK[existing.severity]
                ):
                    best_by_category[complaint_type] = result

        # Sort by descending severity for a sensible return order
        sorted_results = sorted(
            best_by_category.values(),
            key=lambda r: self._SEVERITY_RANK[r.severity],
            reverse=True,
        )
        return sorted_results
