"""Unit tests for Comporator."""

import pytest

from comporator import (
    Comporator,
    Equal,
    Field,
    FieldResult,
    NotEqual,
    Source,
    Status,
)


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------


class TestField:
    def test_key_defaults_to_name(self) -> None:
        assert Field("city").key == "city"

    def test_key_uses_alias_when_set(self) -> None:
        assert Field("city", alias="city_uid").key == "city_uid"

    def test_get_reads_by_key(self) -> None:
        assert Field("city", alias="city_uid").get({"city_uid": "Kyiv"}) == "Kyiv"

    def test_get_returns_none_for_missing_key(self) -> None:
        assert Field("missing").get({"other": 1}) is None

    def test_source_count_is_one(self) -> None:
        assert Field("x")._source_count == 1

    def test_repr_without_alias(self) -> None:
        assert repr(Field("city")) == "Field('city')"

    def test_repr_with_alias(self) -> None:
        assert repr(Field("city", alias="city_uid")) == "Field('city', alias='city_uid')"


# ---------------------------------------------------------------------------
# Equal
# ---------------------------------------------------------------------------


class TestEqual:
    def _run(self, *fields: Field, sources: list[dict]) -> list[FieldResult]:
        names = [f"src{i}" for i in range(len(sources))]
        return Equal(*fields).check(sources, names)

    def test_all_equal_is_match(self) -> None:
        result = self._run(
            Field("x"), Field("x"),
            sources=[{"x": 1}, {"x": 1}],
        )
        assert result[0].is_match

    def test_different_values_is_mismatch(self) -> None:
        result = self._run(
            Field("x"), Field("x"),
            sources=[{"x": 1}, {"x": 2}],
        )
        assert result[0].is_mismatch

    def test_strict_false_produces_warning(self) -> None:
        names = ["a", "b"]
        result = Equal(Field("x"), Field("x"), strict=False).check(
            [{"x": 1}, {"x": 2}], names
        )
        assert result[0].is_warning

    def test_single_source_is_match(self) -> None:
        result = Equal(Field("x")).check([{"x": 42}], ["src"])
        assert result[0].is_match

    def test_all_none_is_match(self) -> None:
        result = self._run(
            Field("x"), Field("x"),
            sources=[{"x": None}, {"x": None}],
        )
        assert result[0].is_match

    def test_source_count(self) -> None:
        rule = Equal(Field("a"), Field("b"), Field("c"))
        assert rule._source_count == 3

    def test_values_collected_correctly(self) -> None:
        names = ["s1", "s2"]
        result = Equal(Field("v"), Field("v")).check(
            [{"v": 10}, {"v": 20}], names
        )
        assert result[0].values == [10, 20]

    def test_alias_fields_read_correct_key(self) -> None:
        names = ["s1", "s2"]
        result = Equal(
            Field("city", alias="CityId"),
            Field("city", alias="city_uid"),
        ).check([{"CityId": "001"}, {"city_uid": "001"}], names)
        assert result[0].is_match


# ---------------------------------------------------------------------------
# NotEqual
# ---------------------------------------------------------------------------


class TestNotEqual:
    def test_all_different_is_match(self) -> None:
        names = ["a", "b"]
        result = NotEqual(Field("x"), Field("x")).check(
            [{"x": 1}, {"x": 2}], names
        )
        assert result[0].is_match

    def test_equal_values_is_mismatch(self) -> None:
        names = ["a", "b"]
        result = NotEqual(Field("x"), Field("x")).check(
            [{"x": 5}, {"x": 5}], names
        )
        assert result[0].is_mismatch

    def test_strict_false_produces_warning_on_equal(self) -> None:
        names = ["a", "b"]
        result = NotEqual(Field("x"), Field("x"), strict=False).check(
            [{"x": 5}, {"x": 5}], names
        )
        assert result[0].is_warning


# ---------------------------------------------------------------------------
# Nested rules
# ---------------------------------------------------------------------------


class TestNestedRules:
    def test_nested_check_returns_parent_and_child_results(self) -> None:
        rule = Equal(
            Field("a"),
            Equal(Field("b"), Field("c")),
        )
        names = ["s1", "s2", "s3"]
        results = rule.check(
            [{"a": 1}, {"b": 1}, {"c": 1}], names
        )
        assert len(results) == 2  # outer + inner

    def test_outer_mismatch_inner_warning(self) -> None:
        outer = Equal(
            Field("id"),
            Equal(Field("user_id"), Field("uid"), strict=False),
        )
        names = ["prod", "staging", "backup"]
        results = outer.check(
            [{"id": 1}, {"user_id": 2}, {"uid": 3}], names
        )
        outer_r, inner_r = results
        assert outer_r.is_mismatch
        assert inner_r.is_warning

    def test_depth_increments_for_nested(self) -> None:
        rule = Equal(Field("a"), Equal(Field("b"), Field("c")))
        names = ["s1", "s2", "s3"]
        results = rule.check(
            [{"a": 1}, {"b": 1}, {"c": 1}], names
        )
        assert results[0].depth == 0
        assert results[1].depth == 1

    def test_source_names_sliced_for_nested(self) -> None:
        rule = Equal(Field("a"), Equal(Field("b"), Field("c")))
        names = ["prod", "staging", "backup"]
        results = rule.check(
            [{"a": 1}, {"b": 1}, {"c": 1}], names
        )
        assert results[1].source_names == ["staging", "backup"]


# ---------------------------------------------------------------------------
# FieldResult
# ---------------------------------------------------------------------------


class TestFieldResult:
    def _make(self, status: Status, truth: str | None = None) -> FieldResult:
        return FieldResult(
            rule=Equal(Field("city"), Field("city")),
            fields=[Field("city"), Field("city")],
            values=["Kyiv", "Lviv"],
            source_names=["prod", "staging"],
            truth=truth,
            status=status,
        )

    def test_is_match(self) -> None:
        r = self._make(Status.MATCH)
        assert r.is_match and not r.is_mismatch and not r.is_warning

    def test_is_mismatch(self) -> None:
        r = self._make(Status.MISMATCH)
        assert r.is_mismatch and not r.is_match

    def test_is_warning(self) -> None:
        r = self._make(Status.WARNING)
        assert r.is_warning and not r.is_match

    def test_repr_without_truth(self) -> None:
        r = self._make(Status.MISMATCH)
        text = repr(r)
        assert "mismatch" in text
        assert "prod" in text
        assert "staging" in text

    def test_repr_with_truth_shows_wrong_source(self) -> None:
        r = self._make(Status.MISMATCH, truth="prod")
        text = repr(r)
        assert "truth (prod)='Kyiv'" in text
        assert "wrong: staging='Lviv'" in text

    def test_repr_match_with_truth_shows_normal_format(self) -> None:
        r = FieldResult(
            rule=Equal(Field("x"), Field("x")),
            fields=[Field("x"), Field("x")],
            values=[1, 1],
            source_names=["prod", "staging"],
            truth="prod",
            status=Status.MATCH,
        )
        assert "truth" not in repr(r)

    def test_repr_depth_indents(self) -> None:
        r = FieldResult(
            rule=Equal(Field("x"), Field("x")),
            fields=[Field("x"), Field("x")],
            values=[1, 1],
            source_names=["s1", "s2"],
            truth=None,
            status=Status.MATCH,
            depth=2,
        )
        assert repr(r).startswith("    ")  # 4 spaces = depth 2


# ---------------------------------------------------------------------------
# Comporator + ComparisonResult
# ---------------------------------------------------------------------------


class TestComporator:
    def _sources(self, truth: bool = False) -> list[Source]:
        return [
            Source("prod",    {"id": 1, "city": "Kyiv",  "CityId": "001"}, truth=truth),
            Source("staging", {"id": 1, "city": "Lviv",  "city_uid": "001"}),
            Source("backup",  {"id": 2, "city": "Kyiv",  "cityUid": "002"}),
        ]

    def _schemas(self) -> list[Equal]:
        return [
            Equal(Field("id"),   Field("id"),   Field("id")),
            Equal(Field("city"), Field("city"), Field("city"), strict=False),
            Equal(
                Field("city", alias="CityId"),
                Field("city", alias="city_uid"),
                Field("city", alias="cityUid"),
            ),
        ]

    def test_counts(self) -> None:
        result = Comporator(self._sources(), self._schemas()).compare()
        assert result.mismatches >= 1
        assert result.warnings >= 1

    def test_no_truth_repr_has_no_truth_label(self) -> None:
        result = Comporator(self._sources(truth=False), self._schemas()).compare()
        assert all("truth" not in repr(r) for r in result.results)

    def test_with_truth_repr_highlights_wrong(self) -> None:
        result = Comporator(self._sources(truth=True), self._schemas()).compare()
        mismatch_reprs = [repr(r) for r in result.results if r.is_mismatch]
        assert any("truth (prod)" in t for t in mismatch_reprs)

    def test_all_match(self) -> None:
        result = Comporator(
            sources=[
                Source("s1", {"x": 42}),
                Source("s2", {"x": 42}),
            ],
            schemas=[Equal(Field("x"), Field("x"))],
        ).compare()
        assert result.matches == 1
        assert result.mismatches == 0
        assert result.warnings == 0

    def test_str_output_contains_counts(self) -> None:
        result = Comporator(
            sources=[Source("a", {"v": 1}), Source("b", {"v": 2})],
            schemas=[Equal(Field("v"), Field("v"))],
        ).compare()
        text = str(result)
        assert "Matches:" in text
        assert "Mismatches:" in text
        assert "Warnings:" in text
