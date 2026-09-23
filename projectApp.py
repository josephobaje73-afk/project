"""
MedSimplify — Plain-Language Drug Information App
====================================================

A Streamlit application that lets a user type a medication name and:
  1. Fetches official drug labeling info from the openFDA Drug Labeling API
  2. Checks the openFDA Recall/Enforcement API for active recalls
  3. Uses the Gemini API to rewrite dense medical text into plain language
  4. Saves every search to a local JSON file so the user can revisit history

Python concepts demonstrated (per assignment spec):
  - File handling      -> SearchHistory reads/writes a JSON file on disk
  - Exception handling -> custom exceptions for invalid names, empty results,
                           network errors, and missing fields
  - Regular expressions -> Medication.clean_text() and extract_warning_keywords()
  - OOP                -> Medication, FDAClient, AITranslator, SearchHistory

Tech stack: Python, Streamlit, Requests, JSON, Gemini API, openFDA APIs.

Run with:
    streamlit run app.py

Environment variable required (or entered in the sidebar at runtime):
    GEMINI_API_KEY
"""

from __future__ import annotations

import json
import importlib
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, cast

JSONValue = dict[str, object] | list[object] | str | int | float | bool | None

try:
    st: Any = importlib.import_module("streamlit")
except ModuleNotFoundError:
    class _StreamlitFallback:
        def __call__(self, *args: object, **kwargs: object) -> None:
            return None

        def __getattr__(self, name: str) -> "_StreamlitFallback":
            return self

        def __enter__(self) -> "_StreamlitFallback":
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
            return False

    st = _StreamlitFallback()

import requests  # pyright: ignore[reportMissingModuleSource]


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class MedSimplifyError(Exception):
    """Base class for all app-specific errors."""


class InvalidMedicationNameError(MedSimplifyError):
    """Raised when the user's input fails basic validation."""


class MedicationNotFoundError(MedSimplifyError):
    """Raised when the openFDA label API returns no results."""


class FDANetworkError(MedSimplifyError):
    """Raised when a request to openFDA fails at the network/HTTP level."""


class AITranslationError(MedSimplifyError):
    """Raised when the Gemini API call fails or returns an unusable response."""


# ---------------------------------------------------------------------------
# Medication (OOP + regex)
# ---------------------------------------------------------------------------

@dataclass
class Medication:
    """Represents a single medication's parsed label data."""

    name: str
    generic_name: str = ""
    brand_names: list[str] = field(default_factory=list)
    usage: str = ""
    warnings: str = ""
    side_effects: str = ""
    dosage: str = ""
    raw: dict[str, object] = field(default_factory=dict)

    # --- regex-based text cleaning -----------------------------------
    _CITATION_RE: re.Pattern[str] = re.compile(r"\(\d+(\.\d+)*\)")  # e.g. "(2.1)"
    _WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")
    _BULLET_RE: re.Pattern[str] = re.compile(r"^\s*[\u2022\-\*]\s*", re.MULTILINE)
    _NAME_VALIDATION_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"^[A-Za-z0-9\s\-/]+$"
    )

    # A handful of clinically meaningful phrases we want to surface as
    # "keywords" to the user even before AI simplification.
    _WARNING_KEYWORD_PATTERNS: ClassVar[list[str]] = [
        r"do not use\b[^.]*",
        r"may cause\b[^.]*",
        r"risk of\b[^.]*",
        r"stop use and ask a doctor\b[^.]*",
        r"contraindicat\w*[^.]*",
        r"serious side effects?\b[^.]*",
        r"ask a doctor before use\b[^.]*",
        r"overdose\b[^.]*",
    ]

    @classmethod
    def validate_name(cls, name: str) -> str:
        """Validate and normalize a user-supplied medication name.

        Raises InvalidMedicationNameError on empty or malformed input.
        """
        cleaned = name.strip()
        if not cleaned:
            raise InvalidMedicationNameError("Medication name cannot be empty.")
        if len(cleaned) > 100:
            raise InvalidMedicationNameError("Medication name is too long.")
        if not cls._NAME_VALIDATION_RE.match(cleaned):
            raise InvalidMedicationNameError(
                "Medication name may only contain letters, numbers, spaces, hyphens, and slashes."
            )
        return cleaned

    @classmethod
    def clean_text(cls, text: str | list[str] | None) -> str:
        """Strip FDA-label boilerplate (citation markers, bullets, extra
        whitespace) from a raw label field using regular expressions."""
        if not text:
            return ""
        if isinstance(text, list):
            text = " ".join(text)
        text = cls._CITATION_RE.sub("", text)
        text = cls._BULLET_RE.sub("", text)
        text = cls._WHITESPACE_RE.sub(" ", text)
        return text.strip()

    @classmethod
    def from_openfda_result(cls, name: str, result: dict[str, object]) -> "Medication":
        """Build a Medication instance from one openFDA label API result."""
        openfda_value = result.get("openfda", {})
        openfda: dict[str, object] = {}
        if isinstance(openfda_value, dict):
            raw_openfda = cast(dict[object, object], openfda_value)
            openfda = {
                str(key): value
                for key, value in raw_openfda.items()
                if isinstance(key, str)
            }

        def text_field(key: str) -> str | list[str] | None:
            value = result.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                list_values = cast(list[object], value)
                return [item for item in list_values if isinstance(item, str)]
            return None

        def string_list_field(key: str) -> list[str]:
            value = openfda.get(key)
            if isinstance(value, list):
                list_values = cast(list[object], value)
                return [item for item in list_values if isinstance(item, str)]
            return []

        generic_names = string_list_field("generic_name")
        brand_names = string_list_field("brand_name")
        warnings = text_field("warnings") or text_field("warnings_and_cautions")

        return cls(
            name=name,
            generic_name=", ".join(generic_names) or name,
            brand_names=brand_names,
            usage=cls.clean_text(text_field("indications_and_usage")),
            warnings=cls.clean_text(warnings),
            side_effects=cls.clean_text(text_field("adverse_reactions")),
            dosage=cls.clean_text(text_field("dosage_and_administration")),
            raw=result,
        )

    def extract_warning_keywords(self) -> list[str]:
        """Use regex to pull short, human-readable warning snippets out of
        the raw warnings text (in addition to the AI-simplified version)."""
        if not self.warnings:
            return []
        found: list[str] = []
        for pattern in self._WARNING_KEYWORD_PATTERNS:
            raw_matches: list[str] = re.findall(pattern, self.warnings, flags=re.IGNORECASE)
            for match in raw_matches:
                snippet = match.strip()
                if snippet and snippet not in found:
                    found.append(snippet[:160])
        return found[:8]

    def has_missing_fields(self) -> list[str]:
        """Report which key sections came back empty from the API."""
        missing: list[str] = []
        for field_name in ("usage", "warnings", "side_effects", "dosage"):
            if not getattr(self, field_name):
                missing.append(field_name)
        return missing


# ---------------------------------------------------------------------------
# FDAClient (OOP + exception handling)
# ---------------------------------------------------------------------------

class FDAClient:
    """Thin wrapper around the openFDA Drug Labeling and Recall/Enforcement
    (Drug Enforcement) APIs."""

    LABEL_URL: str = "https://api.fda.gov/drug/label.json"
    ENFORCEMENT_URL: str = "https://api.fda.gov/drug/enforcement.json"

    def __init__(self, timeout: int = 10):
        self.timeout: int = timeout

    def _get(self, url: str, params: dict[str, str]) -> dict[str, object]:
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
        except requests.exceptions.RequestException as exc:
            raise FDANetworkError(f"Could not reach openFDA: {exc}") from exc

        if response.status_code == 404:
            return {"results": []}
        if not response.ok:
            raise FDANetworkError(
                f"openFDA returned an error (status {response.status_code})."
            )
        try:
            data = cast(object, response.json())
        except ValueError as exc:
            raise FDANetworkError("openFDA returned an unreadable response.") from exc

        if not isinstance(data, dict):
            raise FDANetworkError("openFDA returned an unreadable response.")
        return cast(dict[str, object], data)

    def fetch_label(self, drug_name: str) -> Medication:
        """Look up a medication by generic or brand name. Tries generic
        name first, falls back to brand name, then a free-text search."""
        queries = [
            f'openfda.generic_name:"{drug_name}"',
            f'openfda.brand_name:"{drug_name}"',
            f'indications_and_usage:"{drug_name}"',
        ]
        for query in queries:
            data = self._get(self.LABEL_URL, {"search": query, "limit": "1"})
            results_value = data.get("results")
            results_list: list[object] = results_value if isinstance(results_value, list) else []
            for candidate in results_list:
                if isinstance(candidate, dict):
                    return Medication.from_openfda_result(drug_name, cast(dict[str, object], candidate))

        raise MedicationNotFoundError(
            "No FDA label information found for "
            + f'"{drug_name}". Check the spelling or try the generic name.'
        )

    def fetch_recalls(self, drug_name: str, limit: int = 5) -> list[dict[str, str]]:
        """Return recent recall/enforcement records mentioning this drug."""
        query = f'product_description:"{drug_name}"'
        data = self._get(
            self.ENFORCEMENT_URL,
            {"search": query, "limit": str(limit), "sort": "report_date:desc"},
        )
        results_value = data.get("results")
        if not isinstance(results_value, list):
            return []

        recalls: list[dict[str, str]] = []
        for item in results_value:
            if not isinstance(item, dict):
                continue
            item_dict = cast(dict[str, object], item)
            product_description = item_dict.get("product_description")
            reason_for_recall = item_dict.get("reason_for_recall")
            classification = item_dict.get("classification", "Unknown")
            status = item_dict.get("status", "Unknown")
            report_date = item_dict.get("report_date", "")
            recalling_firm = item_dict.get("recalling_firm", "")

            product_description_text = ""
            if isinstance(product_description, str):
                product_description_text = product_description
            elif isinstance(product_description, list):
                product_description_parts = cast(list[object], product_description)
                product_description_text = " ".join(
                    str(part) for part in product_description_parts if isinstance(part, str)
                )

            reason_for_recall_text = ""
            if isinstance(reason_for_recall, str):
                reason_for_recall_text = reason_for_recall
            elif isinstance(reason_for_recall, list):
                reason_for_recall_parts = cast(list[object], reason_for_recall)
                reason_for_recall_text = " ".join(
                    str(part) for part in reason_for_recall_parts if isinstance(part, str)
                )

            recalls.append(
                {
                    "product_description": Medication.clean_text(product_description_text),
                    "reason_for_recall": Medication.clean_text(reason_for_recall_text),
                    "classification": str(classification),
                    "status": str(status),
                    "report_date": str(report_date),
                    "recalling_firm": str(recalling_firm),
                }
            )
        return recalls


# ---------------------------------------------------------------------------
# AITranslator (OOP + exception handling)
# ---------------------------------------------------------------------------

class AITranslator:
    """Calls the Gemini API to rewrite medical text in plain language."""

    API_URL_TEMPLATE: str = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:generateContent"
    )

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        if not api_key:
            raise AITranslationError("No Gemini API key was provided.")
        self.api_key: str = api_key
        self.model: str = model

    def simplify(self, text: str, section_name: str) -> str:
        """Rewrite one section of drug info in everyday language."""
        if not text:
            return "No information was provided by the FDA for this section."

        prompt = (
            "You are helping a patient with no medical background understand "
            f"their medication. Rewrite the following '{section_name}' section "
            "from an official FDA drug label in simple, clear, everyday "
            "language. Keep it accurate, use short sentences, and avoid "
            "jargon. Limit your answer to about 120 words.\n\n"
            f"Original text:\n{text}"
        )

        url = self.API_URL_TEMPLATE.format(model=self.model)
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}

        try:
            response = requests.post(
                url, headers=headers, params=params,
                data=json.dumps(payload), timeout=20,
            )
        except requests.exceptions.RequestException as exc:
            raise AITranslationError(f"Could not reach Gemini API: {exc}") from exc

        if not response.ok:
            raise AITranslationError(
                f"Gemini API returned status {response.status_code}: {response.text[:200]}"
            )

        try:
            payload = cast(object, response.json())
        except ValueError as exc:
            raise AITranslationError("Gemini API returned an unexpected response.") from exc

        if not isinstance(payload, dict):
            raise AITranslationError("Gemini API returned an unexpected response.")

        payload_dict = cast(dict[str, object], payload)
        candidates = payload_dict.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise AITranslationError("Gemini API returned an unexpected response.")

        first_candidate = cast(list[object], candidates)[0]
        if not isinstance(first_candidate, dict):
            raise AITranslationError("Gemini API returned an unexpected response.")

        candidate_dict = cast(dict[str, object], first_candidate)
        content = candidate_dict.get("content")
        if not isinstance(content, dict):
            raise AITranslationError("Gemini API returned an unexpected response.")

        content_dict = cast(dict[str, object], content)
        parts = content_dict.get("parts")
        if not isinstance(parts, list) or not parts:
            raise AITranslationError("Gemini API returned an unexpected response.")

        first_part = cast(list[object], parts)[0]
        if not isinstance(first_part, dict):
            raise AITranslationError("Gemini API returned an unexpected response.")

        part_dict = cast(dict[str, object], first_part)
        text_value = part_dict.get("text")
        if not isinstance(text_value, str):
            raise AITranslationError("Gemini API returned an unexpected response.")

        return text_value.strip()


# ---------------------------------------------------------------------------
# SearchHistory (OOP + file handling)
# ---------------------------------------------------------------------------

class SearchHistory:
    """Persists search results to a local JSON file."""

    def __init__(self, filepath: str = "search_history.json"):
        self.filepath: Path = Path(filepath)
        if not self.filepath.exists():
            self._write([])

    def _read(self) -> list[dict[str, object]]:
        try:
            with self.filepath.open("r", encoding="utf-8") as f:
                loaded = cast(object, json.load(f))
        except (json.JSONDecodeError, FileNotFoundError):
            return []

        if not isinstance(loaded, list):
            return []

        entries: list[dict[str, object]] = []
        for entry in cast(list[object], loaded):
            if isinstance(entry, dict):
                entries.append(cast(dict[str, object], entry))
        return entries

    def _write(self, entries: list[dict[str, object]]) -> None:
        with self.filepath.open("w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

    def add(self, entry: dict[str, object]) -> None:
        entries = self._read()
        entries.insert(0, entry)  # newest first
        self._write(entries[:100])  # cap history size

    def get_all(self) -> list[dict[str, object]]:
        return self._read()

    def clear(self) -> None:
        self._write([])


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def build_entry(
    drug_name: str,
    medication: Medication,
    recalls: list[dict[str, str]],
    simplified: dict[str, str],
) -> dict[str, object]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": drug_name,
        "generic_name": medication.generic_name,
        "brand_names": medication.brand_names,
        "recall_found": bool(recalls),
        "recall_count": len(recalls),
        "simplified": simplified,
    }


def render_recall_banner(recalls: list[dict[str, str]]) -> None:
    if recalls:
        st.error(f"⚠️ {len(recalls)} recall notice(s) found for this medication.")
        for r in recalls:
            title = (
                f"{r['recalling_firm'] or 'Unknown firm'} — {r['report_date']} "
                + f"(Class {r['classification']})"
            )
            with st.expander(title):
                st.write(f"**Reason:** {r['reason_for_recall'] or 'Not specified'}")
                st.write(f"**Status:** {r['status']}")
                st.write(f"**Product:** {r['product_description']}")
    else:
        st.success("✅ No recent recalls found for this medication.")


def main() -> None:
    st.set_page_config(page_title="MedSimplify", page_icon="💊", layout="centered")
    st.title("💊 MedSimplify")
    st.caption(
        "Look up a medication, get its FDA label info rewritten in plain "
        + "language, and check for active recalls."
    )

    # --- Sidebar: API key + history ------------------------------------
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input(
            "Gemini API Key",
            value=os.environ.get("GEMINI_API_KEY", ""),
            type="password",
            help="Get a key from Google AI Studio. Stored only for this session.",
        )

        st.divider()
        st.header("Search History")
        history = SearchHistory()
        entries = history.get_all()
        if entries:
            for e in entries[:15]:
                flag = "⚠️" if e["recall_found"] else ""
                st.write(f"{flag} **{e['query']}** — {e['timestamp']}")
            if st.button("Clear history"):
                history.clear()
                st.rerun()
        else:
            st.write("No searches yet.")

    # --- Main search form ------------------------------------------------
    drug_name_input = st.text_input("Enter a medication name", placeholder="e.g. ibuprofen")
    search_clicked = st.button("Search", type="primary")

    if not search_clicked:
        return

    # 1. Validate input
    try:
        drug_name = Medication.validate_name(drug_name_input)
    except InvalidMedicationNameError as exc:
        st.error(str(exc))
        return

    fda_client = FDAClient()

    # 2. Fetch label info
    with st.spinner(f"Looking up '{drug_name}'..."):
        try:
            medication = fda_client.fetch_label(drug_name)
        except MedicationNotFoundError as exc:
            st.warning(str(exc))
            return
        except FDANetworkError as exc:
            st.error(f"Network problem while fetching drug info: {exc}")
            return

        missing = medication.has_missing_fields()
        if missing:
            st.info(
                "Note: the FDA label was missing data for: " + ", ".join(missing)
            )

        # 3. Fetch recalls (non-fatal if it fails)
        try:
            recalls = fda_client.fetch_recalls(medication.generic_name or drug_name)
        except FDANetworkError as exc:
            st.warning(f"Could not check recalls right now: {exc}")
            recalls = []

    st.subheader(f"{medication.generic_name.title() or drug_name.title()}")
    if medication.brand_names:
        st.caption("Brand names: " + ", ".join(medication.brand_names))

    render_recall_banner(recalls)

    keywords = medication.extract_warning_keywords()
    if keywords:
        st.markdown("**Key warning phrases (raw extract):**")
        st.write(" • " + "\n • ".join(keywords))

    # 4. Simplify with Gemini
    simplified: dict[str, str] = {}
    sections = {
        "Usage": medication.usage,
        "Dosage & Administration": medication.dosage,
        "Warnings": medication.warnings,
        "Side Effects": medication.side_effects,
    }

    if not api_key:
        st.warning(
            "No Gemini API key provided — showing raw FDA text instead of "
            + "the simplified version."
        )
        for label, text in sections.items():
            st.markdown(f"### {label}")
            st.write(text or "_No data available._")
    else:
        translator = AITranslator(api_key=api_key)
        for label, text in sections.items():
            st.markdown(f"### {label}")
            try:
                with st.spinner(f"Simplifying '{label}'..."):
                    simple_text = translator.simplify(text, label)
                st.write(simple_text)
                simplified[label] = simple_text
            except AITranslationError as exc:
                st.error(f"Could not simplify this section: {exc}")
                st.write(text or "_No data available._")
            time.sleep(0.2)  # gentle pacing between API calls

    # 5. Save to history
    entry = build_entry(drug_name, medication, recalls, simplified)
    history.add(entry)
    st.toast("Search saved to history.")


if __name__ == "__main__":
    main()