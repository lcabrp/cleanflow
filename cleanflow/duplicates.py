"""
Duplicate record detection and resolution.

Finds exact and fuzzy duplicates, then resolves them using configurable
survivorship rules. Adapted from Bala Priya C's DuplicateDetector and
KDNuggets technique #2 (smart dedup keeping the most complete record).
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Literal
from dataclasses import dataclass
from difflib import SequenceMatcher

from .base import BaseTransformer


@dataclass
class DuplicateGroup:
    """A group of records identified as duplicates."""
    group_id: int
    record_indices: List[int]
    similarity_scores: List[float]
    match_type: str  # "exact" or "fuzzy"


@dataclass
class DuplicateReport:
    """Summary of duplicate detection and resolution."""
    total_duplicates_found: int
    groups_found: int
    rows_before: int
    rows_after: int
    method: str
    survivorship_rule: str


class DuplicateHandler(BaseTransformer):
    """Detects and resolves duplicate records.

    Supports exact matching (on specified columns) and fuzzy matching
    (using string similarity). Resolution strategies include keeping the
    first, last, or most complete record, or merging fields.

    Example::

        handler = DuplicateHandler()
        handler.find_exact(df, subset=["name", "email"])
        cleaned_df = handler.transform(df, survivorship="most_complete")
        print(handler.get_report())
    """

    def __init__(self):
        self.duplicate_groups: List[DuplicateGroup] = []
        self._report: Optional[DuplicateReport] = None

    # -- detection -----------------------------------------------------------

    def find_exact(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
    ) -> "DuplicateHandler":
        """Find exact duplicate records.

        Args:
            df: Input DataFrame.
            subset: Columns to check for duplicates. None = all columns.

        Returns:
            self (for chaining).
        """
        self.duplicate_groups = []
        dup_mask = df.duplicated(subset=subset, keep=False)
        if dup_mask.sum() == 0:
            return self

        dup_df = df[dup_mask]
        cols = subset or list(df.columns)
        group_id = 0

        for _, group in dup_df.groupby(cols, dropna=False):
            if len(group) > 1:
                self.duplicate_groups.append(DuplicateGroup(
                    group_id=group_id,
                    record_indices=group.index.tolist(),
                    similarity_scores=[1.0] * len(group),
                    match_type="exact",
                ))
                group_id += 1

        return self

    def find_fuzzy(
        self,
        df: pd.DataFrame,
        match_columns: List[str],
        threshold: float = 0.85,
        blocking_column: Optional[str] = None,
    ) -> "DuplicateHandler":
        """Find fuzzy duplicate records using string similarity.

        Args:
            df: Input DataFrame.
            match_columns: Columns to use for similarity comparison.
            threshold: Minimum similarity score (0-1) to consider a match.
            blocking_column: Optional column to partition data before comparing
                (dramatically reduces comparisons for large datasets).

        Returns:
            self (for chaining).
        """
        self.duplicate_groups = []
        group_id = 0

        if blocking_column and blocking_column in df.columns:
            blocks = df.groupby(blocking_column, dropna=False)
        else:
            blocks = [("_all", df)]

        for _, block in blocks:
            if len(block) < 2:
                continue

            indices = block.index.tolist()
            matched = set()

            for i in range(len(indices)):
                if indices[i] in matched:
                    continue
                group_indices = [indices[i]]
                group_scores = [1.0]

                for j in range(i + 1, len(indices)):
                    if indices[j] in matched:
                        continue

                    score = self._row_similarity(
                        block.loc[indices[i]], block.loc[indices[j]], match_columns
                    )
                    if score >= threshold:
                        group_indices.append(indices[j])
                        group_scores.append(score)
                        matched.add(indices[j])

                if len(group_indices) > 1:
                    matched.add(indices[i])
                    self.duplicate_groups.append(DuplicateGroup(
                        group_id=group_id,
                        record_indices=group_indices,
                        similarity_scores=group_scores,
                        match_type="fuzzy",
                    ))
                    group_id += 1

        return self

    # -- resolution ----------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame,
        survivorship: Literal["first", "last", "most_complete", "merge"] = "most_complete",
    ) -> pd.DataFrame:
        """Resolve detected duplicates using survivorship rules.

        Args:
            df: Input DataFrame (same one used for detection).
            survivorship: Rule for selecting the surviving record:
                - "first": keep the first occurrence
                - "last": keep the last occurrence
                - "most_complete": keep the row with fewest NaN values
                - "merge": combine non-null values from all duplicates

        Returns:
            Deduplicated DataFrame.
        """
        rows_before = len(df)
        df = df.copy()

        if not self.duplicate_groups:
            self._report = DuplicateReport(
                total_duplicates_found=0, groups_found=0,
                rows_before=rows_before, rows_after=len(df),
                method="none", survivorship_rule=survivorship,
            )
            return df

        indices_to_drop = set()
        method = self.duplicate_groups[0].match_type if self.duplicate_groups else "mixed"

        for group in self.duplicate_groups:
            keep_idx, drop_idxs = self._resolve_group(df, group, survivorship)

            # For merge strategy, update the surviving row
            if survivorship == "merge" and keep_idx is not None:
                for col in df.columns:
                    if pd.isna(df.at[keep_idx, col]):
                        for idx in group.record_indices:
                            if idx != keep_idx and pd.notna(df.at[idx, col]):
                                df.at[keep_idx, col] = df.at[idx, col]
                                break

            indices_to_drop.update(drop_idxs)

        df = df.drop(index=list(indices_to_drop)).reset_index(drop=True)

        total_dupes = sum(len(g.record_indices) - 1 for g in self.duplicate_groups)
        self._report = DuplicateReport(
            total_duplicates_found=total_dupes,
            groups_found=len(self.duplicate_groups),
            rows_before=rows_before,
            rows_after=len(df),
            method=method,
            survivorship_rule=survivorship,
        )

        return df

    def get_report(self) -> pd.DataFrame:
        """Return deduplication report."""
        if self._report is None:
            return pd.DataFrame()
        return pd.DataFrame([{
            "duplicates_found": self._report.total_duplicates_found,
            "groups": self._report.groups_found,
            "rows_before": self._report.rows_before,
            "rows_after": self._report.rows_after,
            "rows_removed": self._report.rows_before - self._report.rows_after,
            "method": self._report.method,
            "survivorship": self._report.survivorship_rule,
        }])

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings."""
        if pd.isna(s1) or pd.isna(s2):
            return 0.0
        return SequenceMatcher(None, str(s1).lower(), str(s2).lower()).ratio()

    def _row_similarity(self, row1: pd.Series, row2: pd.Series, columns: List[str]) -> float:
        """Calculate average similarity across specified columns."""
        scores = []
        for col in columns:
            if col in row1.index and col in row2.index:
                scores.append(self._string_similarity(row1[col], row2[col]))
        return np.mean(scores) if scores else 0.0

    def _resolve_group(
        self,
        df: pd.DataFrame,
        group: DuplicateGroup,
        survivorship: str,
    ) -> tuple:
        """Pick the surviving index and return (keep_idx, drop_indices)."""
        indices = group.record_indices

        if survivorship == "first":
            keep = indices[0]
        elif survivorship == "last":
            keep = indices[-1]
        elif survivorship in ("most_complete", "merge"):
            # Keep the row with the most non-null values
            completeness = {
                idx: df.loc[idx].notna().sum() for idx in indices if idx in df.index
            }
            keep = max(completeness, key=completeness.get) if completeness else indices[0]
        else:
            keep = indices[0]

        drop = [idx for idx in indices if idx != keep]
        return keep, drop
