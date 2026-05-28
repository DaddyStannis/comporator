"""
Comporator — compare multiple named SQL-like data sources using a declarative rule tree.

Quick start::

    from comporator import Comporator, Source, Equal, Field

    result = Comporator(
        sources=[
            Source("prod",    key="id", data=[
                {"id": 1, "city": "Kyiv"},
                {"id": 2, "city": "Lviv"},
                {"id": 3, "city": "Odesa"},   # only in prod
            ], truth=True),
            Source("staging", key="id", data=[
                {"id": 1, "city": "Kyiv"},
                {"id": 2, "city": "Kharkiv"}, # mismatch
                {"id": 4, "city": "Dnipro"},  # only in staging
            ]),
        ],
        schemas=[
            Equal(Field("city", source="prod"), Field("city", source="staging")),
        ],
    ).compare()

    print(result)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    "UnmatchedRow",
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
    """A named table-like data source.

    Args:
        name:  identifier shown in reports and referenced by Field.source.
        key:   the dict field used to join rows across sources.
        data:  list of row dicts.
        truth: if True, this source is the ground truth;
               mismatches in other sources are reported relative to it.
    """

    name: str
    key: str
    data: list[dict]
    truth: bool = False


# ---------------------------------------------------------------------------
# Field  —  leaf node of the rule tree
# ---------------------------------------------------------------------------


class Field:
    """Reference to a single field inside a named source row.

    Args:
        name:   logical field name shown in reports.
        source: name of the :class:`Source` this field belongs to.
        alias:  actual key used to look up the value in the row dict.
                Falls back to *name* when omitted.

    Example::

        Field("city", source="prod", alias="city_uid")
        # displayed as "city", reads prod_row["city_uid"]
    """

    def __init__(self, name: str, source: str, alias: str | None = None) -> None:
        self.name = name
        self.source = source
        self.alias = alias

    @property
    def key(self) -> str:
        """The dict key actually used when reading the source row."""
        return self.alias if self.alias is not None else self.name

    def get(self, source_data: dict) -> object:
        return source_data.get(self.key)

    def __repr__(self) -> str:
        alias_part = f", alias={self.alias!r}" if self.alias is not None else ""
        return f"Field({self.name!r}, source={self.source!r}{alias_part})"


# ---------------------------------------------------------------------------
# Rule  —  base class for internal tree nodes
# ---------------------------------------------------------------------------


class Rule(ABC):
    """A comparison rule node.  Children can be :class:`Field` instances or
    other nested :class:`Rule` instances.

    Args:
        *children: Field or nested Rule nodes.
        strict:    if ``True`` (default) a failed check produces MISMATCH;
                   if ``False`` it produces WARNING instead.
    """

    def __init__(self, *children: Field | Rule, strict: bool = True) -> None:
        self.children: list[Field | Rule] = list(children)
        self.strict = strict

    # --- tree structure ---

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

    def _collect_values(self, sources: dict[str, dict]) -> list:
        """Collect leaf values by looking up each Field's named source."""
        values: list = []
        for child in self.children:
            if isinstance(child, Field):
                values.append(child.get(sources[child.source]))
            else:
                values.extend(child._collect_values(sources))
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
        sources: dict[str, dict],
        truth: str | None = None,
        depth: int = 0,
    ) -> list[FieldResult]:
        """Evaluate this node and all nested rules (DFS).

        Args:
            sources: mapping of source name → single row dict.
            truth:   name of the ground-truth source, if any.
            depth:   recursion depth used for indented report output.
        """
        all_values = self._collect_values(sources)
        fields = self._leaf_fields
        source_names = [f.source for f in fields]
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

        for child in self.children:
            if isinstance(child, Rule):
                results.extend(child.check(sources, truth=truth, depth=depth + 1))

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
    """Outcome of a single rule node evaluation for one matched row."""

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
        unique = list(dict.fromkeys(f.name for f in self.fields))
        return ", ".join(unique)

    @property
    def _sources_str(self) -> str:
        if len(self.source_names) <= 1:
            return self.source_names[0] if self.source_names else ""
        return ", ".join(self.source_names[:-1]) + f" and {self.source_names[-1]}"

    def __repr__(self) -> str:
        indent = "  " * self.depth

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

        details = ", ".join(
            f"{sn}={v!r}" for sn, v in zip(self.source_names, self.values, strict=True)
        )
        return (
            f"{indent}{self.status.icon()} "
            f"{self._field_name} = {self.status.value} in {self._sources_str}"
            f" [{details}]"
        )


@dataclass
class UnmatchedRow:
    """A row whose key value is present in only one source.

    Attributes:
        source:    name of the source that contains this row.
        key_value: value of the join key for this row.
        data:      the full row dict.
    """

    source: str
    key_value: object
    data: dict

    def __repr__(self) -> str:
        return f"  ↳ only in {self.source}: key={self.key_value!r}  {self.data}"


@dataclass
class ComparisonResult:
    """Aggregated outcome of a full :class:`Comporator` run."""

    results: list[FieldResult]
    unmatched: list[UnmatchedRow] = field(default_factory=list)

    @property
    def matches(self) -> int:
        return sum(1 for r in self.results if r.is_match)

    @property
    def mismatches(self) -> int:
        return sum(1 for r in self.results if r.is_mismatch)

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.is_warning)

    @property
    def only_in(self) -> dict[str, int]:
        """Count of unmatched rows per source."""
        counts: dict[str, int] = {}
        for u in self.unmatched:
            counts[u.source] = counts.get(u.source, 0) + 1
        return counts

    def __str__(self) -> str:
        lines = [
            f"Matches:    {self.matches}",
            f"Mismatches: {self.mismatches}",
            f"Warnings:   {self.warnings}",
            *[f"Only in {src}: {n}" for src, n in self.only_in.items()],
            "",
            *[repr(r) for r in self.results],
        ]
        if self.unmatched:
            lines += ["", *[repr(u) for u in self.unmatched]]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comporator
# ---------------------------------------------------------------------------


class Comporator:
    """Compare multiple named table sources against a schema of rules.

    Rows are matched across sources by a join key defined on each
    :class:`Source`.  Rows whose key value is absent in at least one source
    are collected as :class:`UnmatchedRow` entries.

    Args:
        sources: list of :class:`Source` objects; at most one may have
                 ``truth=True``.
        schemas: list of top-level :class:`Rule` nodes applied to every
                 matched row.
    """

    def __init__(self, sources: list[Source], schemas: list[Rule]) -> None:
        self.sources = sources
        self.schemas = schemas

    def compare(self) -> ComparisonResult:
        """Join all sources by key, run schemas on matched rows."""
        truth = next((s.name for s in self.sources if s.truth), None)

        # Build per-source index: {key_value: row}
        indices: dict[str, dict[object, dict]] = {
            s.name: {row.get(s.key): row for row in s.data} for s in self.sources
        }

        # Keys present in ALL sources → matched rows
        common_keys: set = set.intersection(*(set(idx) for idx in indices.values()))

        # Rows whose key is absent from at least one other source → unmatched
        unmatched: list[UnmatchedRow] = [
            UnmatchedRow(source=s.name, key_value=kv, data=indices[s.name][kv])
            for s in self.sources
            for kv in sorted(set(indices[s.name]) - common_keys, key=str)
        ]

        # Compare matched rows
        field_results: list[FieldResult] = []
        for key_val in sorted(common_keys, key=str):
            row_by_source = {s.name: indices[s.name][key_val] for s in self.sources}
            for schema in self.schemas:
                field_results.extend(schema.check(row_by_source, truth=truth))

        return ComparisonResult(results=field_results, unmatched=unmatched)
