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
from collections.abc import Callable
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
    "Guard",
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
    SKIP = "skip"        # not evaluated — Guard condition failed

    def icon(self) -> str:
        return {"match": "✓", "mismatch": "✗", "warning": "⚠", "skip": "–"}[self.value]


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
        name:       logical field name shown in reports.
        source:     name of the :class:`Source` this field belongs to.
        alias:      actual key used to look up the value in the row dict.
                    Falls back to *name* when omitted.
        normalizer: optional ``(value: object) -> object`` callable applied
                    to the raw value before comparison.  ``None`` values are
                    passed through unchanged.  Any plain function or lambda
                    works — e.g. ``str.lower``, ``str.strip``,
                    ``lambda v: v.strip().lower()``.

    Example::

        Field("city", source="prod",    alias="city_uid", normalizer=str.lower)
        Field("city", source="staging", normalizer=lambda v: v.strip().lower())
    """

    def __init__(
        self,
        name: str,
        source: str,
        alias: str | None = None,
        normalizer: Callable[[object], object] | None = None,
    ) -> None:
        self.name = name
        self.source = source
        self.alias = alias
        self.normalizer = normalizer

    @property
    def key(self) -> str:
        """The dict key actually used when reading the source row."""
        return self.alias if self.alias is not None else self.name

    def get(self, source_data: dict) -> object:
        value = source_data.get(self.key)
        if self.normalizer is not None and value is not None:
            return self.normalizer(value)
        return value

    # --- operator sugar ---

    def __and__(self, other: Field | Rule) -> Equal:
        """``f1 & f2``  →  ``Equal(f1, f2)``  (strict equality)"""
        return Equal(self, other)

    def __or__(self, other: Field | Rule) -> Equal:
        """``f1 | f2``  →  ``Equal(f1, f2, strict=False)``  (warning on mismatch)"""
        return Equal(self, other, strict=False)

    def __repr__(self) -> str:
        alias_part = f", alias={self.alias!r}" if self.alias is not None else ""
        norm_name = getattr(self.normalizer, "__name__", None) or repr(self.normalizer)
        norm_part = f", normalizer={norm_name}" if self.normalizer is not None else ""
        return f"Field({self.name!r}, source={self.source!r}{alias_part}{norm_part})"


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

    # --- skip helpers ---

    def _make_skip_results(self, depth: int = 0) -> list[FieldResult]:
        """Return SKIP results for this node and all nested rules, without evaluating."""
        fields = self._leaf_fields
        results: list[FieldResult] = [
            FieldResult(
                rule=self,
                fields=fields,
                values=[],
                source_names=[f.source for f in fields],
                truth=None,
                status=Status.SKIP,
                depth=depth,
            )
        ]
        for child in self.children:
            if isinstance(child, Rule):
                results.extend(child._make_skip_results(depth=depth + 1))
        return results

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

    # --- operator sugar ---

    def __and__(self, other: Field | Rule) -> Equal:
        """``rule & other``  →  ``Equal(rule, other)``  (strict equality)"""
        return Equal(self, other)

    def __or__(self, other: Field | Rule) -> Equal:
        """``rule | other``  →  ``Equal(rule, other, strict=False)``  (warning)"""
        return Equal(self, other, strict=False)

    def __rshift__(self, other: Rule | tuple[Rule, ...]) -> Guard:
        """``condition >> guarded``  →  ``Guard(condition, *guarded)``

        Parentheses are required due to Python operator precedence
        (``>>`` binds tighter than ``&`` and ``|``)::

            (f1 & f2) >> (f3 & f4)
        """
        if isinstance(other, tuple):
            return Guard(self, *other)
        return Guard(self, other)

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


class Guard(Rule):
    """Evaluate dependent rules only when the condition rule passes.

    The first argument is the *condition* — any :class:`Rule`.  The remaining
    arguments are *guarded* rules that run only when the condition produces
    :attr:`Status.MATCH`.  If the condition mismatches or warns, guarded rules
    are returned as :attr:`Status.SKIP` without being evaluated.

    Guard itself produces no result entry — only the condition results and the
    guarded (or skip) results are returned.

    Example — skip amount/city checks when statuses differ::

        Guard(
            Equal(Field("status", source="prod"), Field("status", source="staging")),
            Equal(Field("amount", source="prod"), Field("total",  source="staging")),
            Equal(Field("city",   source="prod"), Field("city",   source="staging")),
        )
    """

    def __init__(self, condition: Rule, *rules: Rule) -> None:
        super().__init__(condition, *rules)
        self._condition = condition
        self._guarded = list(rules)

    def _matches(self, values: list) -> bool:  # never called; Guard overrides check()
        return True

    def check(
        self,
        sources: dict[str, dict],
        truth: str | None = None,
        depth: int = 0,
    ) -> list[FieldResult]:
        condition_results = self._condition.check(sources, truth=truth, depth=depth)
        results: list[FieldResult] = list(condition_results)

        passed = condition_results[0].is_match
        for rule in self._guarded:
            if passed:
                results.extend(rule.check(sources, truth=truth, depth=depth))
            else:
                results.extend(rule._make_skip_results(depth=depth))

        return results


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
    def is_skip(self) -> bool:
        return self.status == Status.SKIP

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

        if self.is_skip:
            return f"{indent}{self.status.icon()} {self._field_name} = skip [{self._sources_str}]"

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
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.is_skip)

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
            f"Skipped:    {self.skipped}",
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
