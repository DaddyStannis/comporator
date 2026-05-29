"""
Benchmark: datacompy vs Comporator across different dataset sizes.
"""

import os
import sys
import time

import datacompy
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from comporator import Comporator, Equal, Field, Source

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def make_data(n: int, mismatch_rate: float = 0.1):
    """Generate prod/staging row lists of size n with ~mismatch_rate mismatches."""
    prod = [
        {
            "id": i,
            "client_name": f"User_{i}",
            "CityId": f"{i % 100:03d}",
            "amount": float(i * 10),
        }
        for i in range(n)
    ]
    staging = [
        {
            "use    r_id": i,
            "name": f"User_{i}",
            "city_uid": f"{(i % 100 + (1 if i % int(1 / mismatch_rate) == 0 else 0)):03d}",
            "total": float(i * 10),
        }
        for i in range(n)
    ]
    return prod, staging


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_datacompy(prod_rows, staging_rows):
    df_prod = pd.DataFrame(prod_rows)
    df_staging = pd.DataFrame(staging_rows).rename(
        columns={
            "user_id": "id",
            "name": "client_name",
            "city_uid": "CityId",
            "total": "amount",
        }
    )
    compare = datacompy.PandasCompare(df_prod, df_staging, join_columns=["id"])
    _ = compare.report()  # force full evaluation


def run_comporator(prod_rows, staging_rows):
    result = Comporator(
        sources=[
            Source("prod", key="id", data=prod_rows, truth=True),
            Source("staging", key="user_id", data=staging_rows),
        ],
        schemas=[
            Equal(Field("client_name", source="prod"), Field("name", source="staging")),
            Equal(
                Field("city", source="prod", alias="CityId"),
                Field("city", source="staging", alias="city_uid"),
                strict=False,
            ),
            Equal(Field("amount", source="prod"), Field("total", source="staging")),
        ],
    ).compare()
    _ = str(result)  # force full str rendering


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

SIZES = [100, 1_000, 10_000, 100_000]
REPEATS = 3

print(f"{'Rows':>10}  {'datacompy (ms)':>16}  {'comporator (ms)':>16}  {'ratio':>8}")
print("-" * 60)

for n in SIZES:
    prod, staging = make_data(n)

    dc_times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        run_datacompy(prod, staging)
        dc_times.append((time.perf_counter() - t0) * 1000)

    cp_times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        run_comporator(prod, staging)
        cp_times.append((time.perf_counter() - t0) * 1000)

    dc_ms = min(dc_times)
    cp_ms = min(cp_times)
    ratio = cp_ms / dc_ms if dc_ms else float("inf")

    print(f"{n:>10,}  {dc_ms:>14.1f}  {cp_ms:>14.1f}  {ratio:>7.2f}x")
