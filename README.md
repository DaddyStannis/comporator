# Comporator

Compare multiple named data sources using a declarative rule tree.

## Install

```bash
pip install comporator
```

## Quick start

```python
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
```

```
Matches:    1
Mismatches: 0
Warnings:   1

✓ id = match in prod and staging [prod=1, staging=1]
⚠ city = warning [truth (prod)='Kyiv', wrong: staging='Lviv']
```

## Concepts

### `Source(name, data, truth=False)`
A named dict of field values. Mark one source as `truth=True` to highlight which sources deviate from it.

### `Field(name, alias=None)`
A reference to a field inside a source dict. Use `alias` when the key differs across sources:

```python
Field("city", alias="city_uid")   # displayed as "city", reads source["city_uid"]
```

### `Equal` / `NotEqual`
Rules that check whether all fields across sources are equal (or distinct).
Set `strict=False` to downgrade a failure from **mismatch** to **warning**.

### Nested rules
Rules can contain other rules, each consuming the next N sources in order.
Every node in the tree generates its own result entry.

```python
Equal(
    Field("id"),                                    # prod
    Equal(Field("user_id"), Field("uid"), strict=False),  # staging, backup
)
```

```
✗ id = mismatch [truth (prod)=1, wrong: backup=2]
  ⚠ user_id, uid = warning in staging and backup [staging=1, backup=2]
```

## Status icons

| Icon | Meaning |
|------|---------|
| ✓ | All values match |
| ✗ | Mismatch (`strict=True`) |
| ⚠ | Mismatch (`strict=False`) |
