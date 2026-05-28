"""
Comporator — compare multiple named data sources using a declarative rule tree.

Quick start::

    from comporator import Comporator, Source, Equal, Field

    result = Comporator(
        sources=[
            Source("prod",    {"id": 1, "city": "Kyiv"}, truth=True),
            Source("staging", {"id": 1, "city": "Lviv"}),
        ],
        schemas=[
            Equal(Field("id"),   Field("id")),
            Equal(Field("city"), Field("city"), strict=False),
        ],
    ).compare()

    print(result)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import cached_property

__all__ = [
    "Comporator",
    "Source",
    "Field",
    "Rule",
    "Equal",
    "NotEqual",
    "FieldResult",
    "ComparisonResult",
    "Status",
]


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class Status(Enum):
    """Result of a single rule evaluation."""

    MATCH = "match"
    MISMATCH = "mismatch"
    WARNING = "warning"  # mismatch in non-strict mode

    def icon(self) -> str:
        return {"match": "✓", "mismatch": "✗", "warning": "⚠"}[self.value]


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


@dataclass
class Source:
    """A named data source.

    Args:
        name:  identifier shown in reports.
        data:  a dict of field values (one row).
        truth: if True, this source is the ground truth;
               mismatches in other sources are reported relative to it.
    """

    name: str
    data: dict
    truth: bool = False


# ---------------------------------------------------------------------------
# Field  —  leaf node of the rule tree
# ---------------------------------------------------------------------------


class Field:
    """Reference to a single field inside one source dict.

    Args:
        name:  logical field name shown in reports.
        alias: actual key used to look up the value in the source dict.
               Falls back to *name* when omitted.

    Example::

        Field("city", alias="city_uid")
        # displayed as "city" but reads source["city_uid"]
    """

    def __init__(self, name: str, alias: str | None = None) -> None:
        self.name = name
        self.alias = alias

    @property
    def key(self) -> str:
        """The dict key actually used when reading the source."""
        return self.alias if self.alias is not None else self.name

    @property
    def _source_count(self) -> int:
        return 1

    def get(self, source: dict) -> object:
        return source.get(self.key)

    def __repr__(self) -> str:
        if self.alias is not None:
            return f"Field({self.name!r}, alias={self.alias!r})"
        return f"Field({self.name!r})"


# ---------------------------------------------------------------------------
# Rule  —  base class for internal tree nodes
# ---------------------------------------------------------------------------


class Rule(ABC):
    """A comparison rule node.  Children can be :class:`Field` instances or
    other nested :class:`Rule` instances.

    Args:
        *children: one child per source (Field or nested Rule).
        strict:    if ``True`` (default) a failed check produces MISMATCH;
                   if ``False`` it produces WARNING instead.
    """

    def __init__(self, *children: Field | Rule, strict: bool = True) -> None:
        self.children: list[Field | Rule] = list(children)
        self.strict = strict

    # --- tree structure ---

    @cached_property
    def _source_count(self) -> int:
        """Total number of sources consumed by this subtree.

        Cached: the tree is immutable after construction.
        """
        return sum(c._source_count for c in self.children)

    @cached_property
    def _leaf_fields(self) -> list[Field]:
        """All leaf Field nodes in DFS left-to-right order.

        Cached: traverses the full subtree; safe to call repeatedly.
        """
        result: list[Field] = []
        for c in self.children:
            result += [c] if isinstance(c, Field) else c._leaf_fields
        return result

    # --- value collection ---

    def _collect_values(self, sources: list[dict]) -> list:
        """Recursively gather leaf values, distributing sources across the tree."""
        values: list = []
        offset = 0
        for child in self.children:
            n = child._source_count
            chunk = sources[offset : offset + n]
            if isinstance(child, Field):
                values.append(child.get(chunk[0]))
            else:
                values.extend(child._collect_values(chunk))
            offset += n
        return values

    # --- evaluation ---

    @abstractmethod
    def _matches(self, values: list) -> bool: ...

    def _resolve_status(self, *, matched: bool) -> Status:
        if matched:
            return Status.MATCH
        return Status.MISMATCH if self.strict else Status.WARNING

    def check(
        self,
        sources: list[dict],
        source_names: list[str],
        truth: str | None = None,
        depth: int = 0,
    ) -> list[FieldResult]:
        """Evaluate this node and all nested rules (DFS).

        Returns a flat list of :class:`FieldResult` — one per rule node,
        with the current node first followed by its nested children.

        Args:
            sources:      raw data dicts, one per leaf Field in tree order.
            source_names: display names corresponding to *sources*.
            truth:        name of the ground-truth source, if any.
            depth:        recursion depth (used for indentation in reports).
        """
        all_values = self._collect_values(sources)
        fields = self._leaf_fields
        status = self._resolve_status(matched=self._matches(all_values))

        results = [
            FieldResult(
                rule=self,
                fields=fields,
                values=all_values,
                source_names=source_names,
                truth=truth,
                status=status,
                depth=depth,
            )
        ]

        offset = 0
        for child in self.children:
            n = child._source_count
            if isinstance(child, Rule):
                results.extend(
                    child.check(
                        sources[offset : offset + n],
                        source_names[offset : offset + n],
                        truth=truth,
                        depth=depth + 1,
                    )
                )
            offset += n

        return results

    def __repr__(self) -> str:
        suffix = "" if self.strict else ", strict=False"
        return f"{self.__class__.__name__}({self.children}{suffix})"


# ---------------------------------------------------------------------------
# Concrete rules
# ---------------------------------------------------------------------------


class Equal(Rule):
    """All fields in the subtree must have the same value."""

    def _matches(self, values: list) -> bool:
        return len(set(values)) <= 1


class NotEqual(Rule):
    """All fields in the subtree must have distinct values."""

    def _matches(self, values: list) -> bool:
        return len(set(values)) > 1


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class FieldResult:
    """Outcome of a single rule node evaluation.

    Attributes:
        rule:         the rule that produced this result.
        fields:       leaf Field nodes in DFS order.
        values:       extracted values corresponding to *fields*.
        source_names: display names corresponding to *values*.
        truth:        name of the ground-truth source, or ``None``.
        status:       MATCH / MISMATCH / WARNING.
        depth:        nesting depth (0 = top-level rule).
    """

    rule: Rule
    fields: list[Field]
    values: list
    source_names: list[str]
    truth: str | None
    status: Status
    depth: int = 0

    @property
    def is_match(self) -> bool:
        return self.status == Status.MATCH

    @property
    def is_mismatch(self) -> bool:
        return self.status == Status.MISMATCH

    @property
    def is_warning(self) -> bool:
        return self.status == Status.WARNING

    @property
    def _field_name(self) -> str:
        """Unique logical field names joined, preserving order."""
        unique = list(dict.fromkeys(f.name for f in self.fields))
        return ", ".join(unique)

    @property
    def _sources_str(self) -> str:
        """'src1, src2 and src3' — Oxford-comma-free English list."""
        if len(self.source_names) <= 1:
            return self.source_names[0] if self.source_names else ""
        return ", ".join(self.source_names[:-1]) + f" and {self.source_names[-1]}"

    def __repr__(self) -> str:
        indent = "  " * self.depth

        # With a ground-truth source, highlight which sources are wrong.
        if self.truth and not self.is_match and self.truth in self.source_names:
            truth_idx = self.source_names.index(self.truth)
            truth_val = self.values[truth_idx]
            wrong = [
                f"{sn}={v!r}"
                for sn, v in zip(self.source_names, self.values, strict=True)
                if sn != self.truth and v != truth_val
            ]
            return (
                f"{indent}{self.status.icon()} {self._field_name} = {self.status.value}"
                f" [truth ({self.truth})={truth_val!r}, wrong: {', '.join(wrong)}]"
            )

        # Default: show every source with its value.
        details = ", ".join(
            f"{sn}={v!r}" for sn, v in zip(self.source_names, self.values, strict=True)
        )
        return (
            f"{indent}{self.status.icon()} "
            f"{self._field_name} = {self.status.value} in {self._sources_str}"
            f" [{details}]"
        )


@dataclass
class ComparisonResult:
    """Aggregated outcome of a full :class:`Comporator` run.

    Attributes:
        results: flat list of :class:`FieldResult`, one per rule node evaluated.
    """

    results: list[FieldResult]

    @property
    def matches(self) -> int:
        return sum(1 for r in self.results if r.is_match)

    @property
    def mismatches(self) -> int:
        return sum(1 for r in self.results if r.is_mismatch)

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.is_warning)

    def __str__(self) -> str:
        lines = [
            f"Matches:    {self.matches}",
            f"Mismatches: {self.mismatches}",
            f"Warnings:   {self.warnings}",
            "",
            *[repr(r) for r in self.results],
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comporator
# ---------------------------------------------------------------------------


class Comporator:
    """Compare multiple named sources against a schema of rules.

    Args:
        sources: list of :class:`Source` objects; at most one may have
                 ``truth=True``.
        schemas: list of top-level :class:`Rule` nodes to evaluate.
    """

    def __init__(self, sources: list[Source], schemas: list[Rule]) -> None:
        self.sources = sources
        self.schemas = schemas

    def compare(self) -> ComparisonResult:
        """Run all schemas against all sources and return the aggregated result."""
        names = [s.name for s in self.sources]
        rows  = [s.data for s in self.sources]
        truth_sources = [s.name for s in self.sources if s.truth]
        truth = truth_sources[0] if truth_sources else None

        results: list[FieldResult] = []
        for schema in self.schemas:
            results.extend(schema.check(rows, names, truth=truth))
        return ComparisonResult(results=results)
