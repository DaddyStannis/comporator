"""Unit tests for Comporator."""

from comporator import (
    Comporator,
    Equal,
    Field,
    FieldResult,
    NotEqual,
    Source,
    Status,
    UnmatchedRow,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

SOURCES = {
    "prod": {"id": 1, "city": "Kyiv", "CityId": "001"},
    "staging": {"id": 1, "city": "Lviv", "city_uid": "001"},
    "backup": {"id": 2, "city": "Kyiv", "cityUid": "002"},
}


def run(rule: Equal | NotEqual, sources: dict = SOURCES) -> list[FieldResult]:
    return rule.check(sources)


def make_sources(truth: bool = False) -> list[Source]:
    return [
        Source(
            "prod",
            key="id",
            data=[
                {"id": 1, "city": "Kyiv"},
                {"id": 2, "city": "Lviv"},
                {"id": 3, "city": "Odesa"},
            ],
            truth=truth,
        ),
        Source(
            "staging",
            key="id",
            data=[
                {"id": 1, "city": "Kyiv"},
                {"id": 2, "city": "Kharkiv"},
                {"id": 4, "city": "Dnipro"},
            ],
        ),
    ]


def make_schemas() -> list[Equal]:
    return [Equal(Field("city", source="prod"), Field("city", source="staging"))]


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------


class TestField:
    def test_key_defaults_to_name(self) -> None:
        assert Field("city", source="prod").key == "city"

    def test_key_uses_alias_when_set(self) -> None:
        assert Field("city", source="prod", alias="CityId").key == "CityId"

    def test_get_reads_by_key(self) -> None:
        f = Field("city", source="prod", alias="CityId")
        assert f.get({"CityId": "001"}) == "001"

    def test_get_returns_none_for_missing_key(self) -> None:
        assert Field("missing", source="prod").get({"other": 1}) is None

    def test_source_attribute_stored(self) -> None:
        assert Field("id", source="staging").source == "staging"

    def test_repr_without_alias(self) -> None:
        assert repr(Field("city", source="prod")) == "Field('city', source='prod')"

    def test_repr_with_alias(self) -> None:
        r = repr(Field("city", source="prod", alias="CityId"))
        assert r == "Field('city', source='prod', alias='CityId')"


# ---------------------------------------------------------------------------
# Equal
# ---------------------------------------------------------------------------


class TestEqual:
    def test_all_equal_is_match(self) -> None:
        result = run(Equal(Field("id", source="prod"), Field("id", source="staging")))
        assert result[0].is_match

    def test_different_values_is_mismatch(self) -> None:
        result = run(Equal(Field("id", source="prod"), Field("id", source="backup")))
        assert result[0].is_mismatch

    def test_strict_false_produces_warning(self) -> None:
        result = run(
            Equal(
                Field("city", source="prod"),
                Field("city", source="staging"),
                strict=False,
            )
        )
        assert result[0].is_warning

    def test_single_field_is_match(self) -> None:
        result = run(Equal(Field("id", source="prod")))
        assert result[0].is_match

    def test_all_none_is_match(self) -> None:
        result = run(
            Equal(Field("missing", source="prod"), Field("missing", source="staging")),
        )
        assert result[0].is_match

    def test_values_collected_correctly(self) -> None:
        result = run(Equal(Field("id", source="prod"), Field("id", source="backup")))
        assert result[0].values == [1, 2]

    def test_alias_reads_correct_key(self) -> None:
        result = run(
            Equal(
                Field("city", source="prod", alias="CityId"),
                Field("city", source="staging", alias="city_uid"),
            )
        )
        assert result[0].is_match
        assert result[0].values == ["001", "001"]

    def test_source_names_match_field_sources(self) -> None:
        result = run(Equal(Field("id", source="prod"), Field("id", source="staging")))
        assert result[0].source_names == ["prod", "staging"]


# ---------------------------------------------------------------------------
# NotEqual
# ---------------------------------------------------------------------------


class TestNotEqual:
    def test_all_different_is_match(self) -> None:
        result = run(NotEqual(Field("id", source="prod"), Field("id", source="backup")))
        assert result[0].is_match

    def test_equal_values_is_mismatch(self) -> None:
        result = run(
            NotEqual(Field("id", source="prod"), Field("id", source="staging"))
        )
        assert result[0].is_mismatch

    def test_strict_false_produces_warning_on_equal(self) -> None:
        result = run(
            NotEqual(
                Field("id", source="prod"), Field("id", source="staging"), strict=False
            )
        )
        assert result[0].is_warning


# ---------------------------------------------------------------------------
# Nested rules
# ---------------------------------------------------------------------------


class TestNestedRules:
    def test_nested_returns_parent_and_child(self) -> None:
        rule = Equal(
            Field("id", source="prod"),
            Equal(Field("id", source="staging"), Field("id", source="backup")),
        )
        assert len(run(rule)) == 2

    def test_outer_mismatch_inner_warning(self) -> None:
        rule = Equal(
            Field("id", source="prod"),
            Equal(
                Field("id", source="staging"),
                Field("id", source="backup"),
                strict=False,
            ),
        )
        outer, inner = run(rule)
        assert outer.is_mismatch
        assert inner.is_warning

    def test_depth_increments(self) -> None:
        rule = Equal(
            Field("id", source="prod"),
            Equal(Field("id", source="staging"), Field("id", source="backup")),
        )
        results = run(rule)
        assert results[0].depth == 0
        assert results[1].depth == 1

    def test_nested_source_names_are_correct(self) -> None:
        rule = Equal(
            Field("id", source="prod"),
            Equal(Field("id", source="staging"), Field("id", source="backup")),
        )
        assert run(rule)[1].source_names == ["staging", "backup"]


# ---------------------------------------------------------------------------
# FieldResult
# ---------------------------------------------------------------------------


class TestFieldResult:
    def _make(self, status: Status, truth: str | None = None) -> FieldResult:
        return FieldResult(
            rule=Equal(Field("city", source="prod"), Field("city", source="staging")),
            fields=[Field("city", source="prod"), Field("city", source="staging")],
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
        text = repr(self._make(Status.MISMATCH))
        assert "mismatch" in text and "prod" in text and "staging" in text

    def test_repr_with_truth_shows_wrong(self) -> None:
        text = repr(self._make(Status.MISMATCH, truth="prod"))
        assert "truth (prod)='Kyiv'" in text
        assert "wrong: staging='Lviv'" in text

    def test_repr_match_with_truth_no_truth_label(self) -> None:
        assert "truth" not in repr(self._make(Status.MATCH, truth="prod"))

    def test_repr_depth_indents(self) -> None:
        r = self._make(Status.MATCH)
        r.depth = 2
        assert repr(r).startswith("    ")


# ---------------------------------------------------------------------------
# Comporator — join logic
# ---------------------------------------------------------------------------


class TestComporator:
    def test_matched_rows_are_compared(self) -> None:
        result = Comporator(make_sources(), make_schemas()).compare()
        # id=1 matches (Kyiv==Kyiv), id=2 mismatches (Lviv!=Kharkiv)
        assert result.matches == 1
        assert result.mismatches == 1

    def test_unmatched_rows_collected(self) -> None:
        result = Comporator(make_sources(), make_schemas()).compare()
        assert len(result.unmatched) == 2

    def test_only_in_counts(self) -> None:
        result = Comporator(make_sources(), make_schemas()).compare()
        assert result.only_in["prod"] == 1  # id=3
        assert result.only_in["staging"] == 1  # id=4

    def test_unmatched_row_attributes(self) -> None:
        result = Comporator(make_sources(), make_schemas()).compare()
        prod_unmatched = next(u for u in result.unmatched if u.source == "prod")
        assert prod_unmatched.key_value == 3
        assert prod_unmatched.data == {"id": 3, "city": "Odesa"}

    def test_truth_highlights_wrong_source(self) -> None:
        result = Comporator(make_sources(truth=True), make_schemas()).compare()
        mismatch_reprs = [repr(r) for r in result.results if r.is_mismatch]
        assert any("truth (prod)" in t for t in mismatch_reprs)

    def test_no_truth_no_truth_label(self) -> None:
        result = Comporator(make_sources(truth=False), make_schemas()).compare()
        assert all("truth" not in repr(r) for r in result.results)

    def test_all_rows_match(self) -> None:
        sources = [
            Source("a", key="id", data=[{"id": 1, "v": 42}, {"id": 2, "v": 7}]),
            Source("b", key="id", data=[{"id": 1, "v": 42}, {"id": 2, "v": 7}]),
        ]
        result = Comporator(
            sources, [Equal(Field("v", source="a"), Field("v", source="b"))]
        ).compare()
        assert result.matches == 2
        assert result.mismatches == 0
        assert result.unmatched == []

    def test_str_output_contains_only_in(self) -> None:
        text = str(Comporator(make_sources(), make_schemas()).compare())
        assert "Only in prod" in text
        assert "Only in staging" in text

    def test_unmatched_repr_shown_in_str(self) -> None:
        text = str(Comporator(make_sources(), make_schemas()).compare())
        assert "↳ only in" in text
