"""
The same scenario implemented with datacompy vs Comporator.

Data: orders table that exists in prod and staging with different field names.
"""

import os
import sys

import datacompy
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from comporator import Comporator, Equal, Field, Source

# ---------------------------------------------------------------------------
# Raw data — two sources with different field names
# ---------------------------------------------------------------------------

prod_rows = [
    {"id": 1, "client_name": "Alice", "CityId": "001", "amount": 100.0},
    {"id": 2, "client_name": "Bob",   "CityId": "002", "amount": 200.0},
    {"id": 3, "client_name": "Carol", "CityId": "001", "amount": 150.0},  # only in prod
]

staging_rows = [
    {"user_id": 1, "name": "Alice", "city_uid": "001", "total": 100.0},
    {"user_id": 2, "name": "Bob",   "city_uid": "999", "total": 200.0},  # city mismatch
    {"user_id": 4, "name": "Dave",  "city_uid": "001", "total":  50.0},  # only in staging
]

# ---------------------------------------------------------------------------
# datacompy — requires manual rename before comparison, only 2 sources
# ---------------------------------------------------------------------------

df_prod    = pd.DataFrame(prod_rows)
df_staging = pd.DataFrame(staging_rows).rename(columns={
    "user_id":  "id",
    "name":     "client_name",
    "city_uid": "CityId",
    "total":    "amount",
})

compare = datacompy.PandasCompare(
    df_prod,
    df_staging,
    join_columns=["id"],
    df1_name="prod",
    df2_name="staging",
)
print("=== datacompy ===")
print(compare.report())

# ---------------------------------------------------------------------------
# Comporator — aliases declared in schema, no rename needed
# ---------------------------------------------------------------------------

result = Comporator(
    sources=[
        Source("prod",    key="id", data=prod_rows,    truth=True),
        Source("staging", key="user_id", data=staging_rows),
    ],
    schemas=[
        Equal(Field("client_name", source="prod"),
              Field("name",        source="staging")),
        Equal(Field("city", source="prod",    alias="CityId"),
              Field("city", source="staging", alias="city_uid"),
              strict=False),
        Equal(Field("amount", source="prod"),
              Field("total",  source="staging")),
    ],
).compare()

print("=== Comporator ===")
print(result)
