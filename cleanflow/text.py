"""
Text data cleaning and normalization.

Provides configurable cleaning pipelines for different column types
(names, addresses, descriptions, codes). Adapted from Bala Priya C's
TextCleaner and KDNuggets techniques #3 and #8.
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from typing import Dict, List
from dataclasses import dataclass

from .base import BaseTransformer


@dataclass
class TextCleaningResult:
    """Report for a single column's text cleaning."""
    column: str
    transformations_applied: List[str]
    rows_modified: int


class TextCleaner(BaseTransformer):
    """Cleans and normalizes text data with type-specific pipelines.

    Built-in pipelines:
        - names: strip → title case → normalize unicode → remove accents
        - addresses: strip → lowercase → expand abbreviations
        - descriptions: strip → remove HTML → normalize unicode → lowercase
        - codes: strip → uppercase → remove spaces
        - emails: strip → lowercase → validate format
        - phones: strip → extract digits → reformat

    Example::

        cleaner = TextCleaner()
        cleaned_df = cleaner.clean_all(df, {
            "name": "name",
            "address": "address",
            "product_desc": "description",
        })
        print(cleaner.get_report())
    """

    ADDRESS_ABBREV = {
        r"\bst\.?\b": "street", r"\bave\.?\b": "avenue",
        r"\bblvd\.?\b": "boulevard", r"\bdr\.?\b": "drive",
        r"\bln\.?\b": "lane", r"\brd\.?\b": "road",
        r"\bct\.?\b": "court", r"\bpl\.?\b": "place",
        r"\bapt\.?\b": "apartment",
    }

    def __init__(self):
        self._results: List[TextCleaningResult] = []

    # -- static cleaning operations -----------------------------------------

    @staticmethod
    def strip_whitespace(text: str) -> str:
        """Remove leading/trailing whitespace and normalize internal spaces."""
        if pd.isna(text):
            return text
        return re.sub(r"\s+", " ", str(text).strip())

    @staticmethod
    def lowercase(text: str) -> str:
        if pd.isna(text):
            return text
        return str(text).lower()

    @staticmethod
    def uppercase(text: str) -> str:
        if pd.isna(text):
            return text
        return str(text).upper()

    @staticmethod
    def titlecase(text: str) -> str:
        if pd.isna(text):
            return text
        return str(text).title()

    @staticmethod
    def remove_html(text: str) -> str:
        """Remove HTML tags and decode entities."""
        if pd.isna(text):
            return text
        from html import unescape
        text = unescape(str(text))
        return re.sub(r"<[^>]+>", "", text).strip()

    @staticmethod
    def remove_special_chars(text: str, keep_chars: str = "") -> str:
        if pd.isna(text):
            return text
        pattern = rf"[^a-zA-Z0-9\s{re.escape(keep_chars)}]"
        return re.sub(pattern, "", str(text))

    @staticmethod
    def remove_urls(text: str) -> str:
        if pd.isna(text):
            return text
        return re.sub(r"https?://\S+|www\.\S+", "", str(text)).strip()

    @staticmethod
    def remove_emails_from_text(text: str) -> str:
        if pd.isna(text):
            return text
        return re.sub(r"\S+@\S+\.\S+", "", str(text)).strip()

    @staticmethod
    def normalize_unicode(text: str) -> str:
        if pd.isna(text):
            return text
        return unicodedata.normalize("NFKD", str(text))

    @staticmethod
    def remove_accents(text: str) -> str:
        if pd.isna(text):
            return text
        nfkd = unicodedata.normalize("NFKD", str(text))
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    # -- type-specific pipelines --------------------------------------------

    def clean_names(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Pipeline for person names: strip → title case → remove accents."""
        original = df[col].copy()
        df[col] = df[col].apply(self.strip_whitespace)
        df[col] = df[col].apply(self.titlecase)
        df[col] = df[col].apply(self.remove_accents)
        modified = (original != df[col]).sum()
        self._results.append(TextCleaningResult(col, ["strip", "titlecase", "remove_accents"], int(modified)))
        return df

    def clean_addresses(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Pipeline for addresses: strip → lowercase → expand abbreviations."""
        original = df[col].copy()
        df[col] = df[col].apply(self.strip_whitespace)
        df[col] = df[col].apply(self.lowercase)
        for pattern, replacement in self.ADDRESS_ABBREV.items():
            df[col] = df[col].str.replace(pattern, replacement, regex=True, flags=re.IGNORECASE)
        modified = (original != df[col]).sum()
        self._results.append(TextCleaningResult(col, ["strip", "lowercase", "expand_abbreviations"], int(modified)))
        return df

    def clean_descriptions(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Pipeline for free text: strip → remove HTML → remove URLs → lowercase."""
        original = df[col].copy()
        df[col] = df[col].apply(self.strip_whitespace)
        df[col] = df[col].apply(self.remove_html)
        df[col] = df[col].apply(self.remove_urls)
        df[col] = df[col].apply(self.lowercase)
        modified = (original != df[col]).sum()
        self._results.append(TextCleaningResult(col, ["strip", "remove_html", "remove_urls", "lowercase"], int(modified)))
        return df

    def clean_codes(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Pipeline for codes/IDs: strip → uppercase → remove spaces."""
        original = df[col].copy()
        df[col] = df[col].apply(self.strip_whitespace)
        df[col] = df[col].apply(self.uppercase)
        df[col] = df[col].str.replace(r"\s+", "", regex=True)
        modified = (original != df[col]).sum()
        self._results.append(TextCleaningResult(col, ["strip", "uppercase", "remove_spaces"], int(modified)))
        return df

    def clean_emails(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Pipeline for email addresses: strip → lowercase → validate."""
        original = df[col].copy()

        def _clean(email):
            if pd.isna(email):
                return np.nan
            email = str(email).strip().lower()
            pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if re.match(pattern, email):
                return email
            elif "@" not in email:
                return np.nan
            return email

        df[col] = df[col].apply(_clean)
        modified = (original.fillna("") != df[col].fillna("")).sum()
        self._results.append(TextCleaningResult(col, ["strip", "lowercase", "validate"], int(modified)))
        return df

    def clean_phones(self, df: pd.DataFrame, col: str, format_str: str = "({0}) {1}-{2}") -> pd.DataFrame:
        """Pipeline for phone numbers: extract digits → reformat."""
        original = df[col].copy()

        def _clean(phone):
            if pd.isna(phone):
                return np.nan
            digits = re.sub(r"\D", "", str(phone))
            if len(digits) == 10:
                return format_str.format(digits[:3], digits[3:6], digits[6:])
            elif len(digits) > 10:
                # Use last 10 digits (strip country code)
                d = digits[-10:]
                return format_str.format(d[:3], d[3:6], d[6:])
            return np.nan

        df[col] = df[col].apply(_clean)
        modified = (original.fillna("") != df[col].fillna("")).sum()
        self._results.append(TextCleaningResult(col, ["extract_digits", "reformat"], int(modified)))
        return df

    # -- batch cleaning ------------------------------------------------------

    def clean_all(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Clean multiple columns based on their declared types.

        Args:
            df: Input DataFrame.
            column_types: Dict mapping column names to types:
                "name", "address", "description", "code", "email", "phone"

        Returns:
            Cleaned DataFrame.
        """
        df = df.copy()
        self._results = []

        pipeline_map = {
            "name": self.clean_names,
            "address": self.clean_addresses,
            "description": self.clean_descriptions,
            "code": self.clean_codes,
            "email": self.clean_emails,
            "phone": self.clean_phones,
        }

        for col, col_type in column_types.items():
            if col not in df.columns:
                continue
            cleaner = pipeline_map.get(col_type)
            if cleaner:
                df = cleaner(df, col)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """BaseTransformer interface. For full control use clean_all() instead."""
        # Auto-clean: strip whitespace on all object columns
        df = df.copy()
        self._results = []
        for col in df.select_dtypes(include=["object", "string"]).columns:
            original = df[col].copy()
            df[col] = df[col].apply(self.strip_whitespace)
            modified = (original != df[col]).sum()
            if modified > 0:
                self._results.append(TextCleaningResult(col, ["strip_whitespace"], int(modified)))
        return df

    def get_report(self) -> pd.DataFrame:
        """Return text cleaning report."""
        if not self._results:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "column": r.column,
                "transformations": ", ".join(r.transformations_applied),
                "rows_modified": r.rows_modified,
            }
            for r in self._results
        ])
