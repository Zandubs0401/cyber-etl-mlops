"""LOAD phase: write the feature table to a relational warehouse with SQLAlchemy.

Defaults to SQLite so the project runs anywhere. Point WAREHOUSE_DB_URL at PostgreSQL
(e.g. postgresql+psycopg2://etl:etl@localhost:5432/warehouse) to use Postgres instead;
no code changes are needed because every operation goes through SQLAlchemy.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from . import config
from .logging_config import get_logger, log_duration

log = get_logger("load")


def load_features(df: pd.DataFrame, db_url: str = config.WAREHOUSE_DB_URL,
                  table: str = config.FEATURES_TABLE, chunksize: int = 20_000) -> int:
    engine = create_engine(db_url)
    with log_duration(log, f"load {len(df)} rows into {table}"):
        # Full refresh inside one transaction: readers never see a half-written table.
        with engine.begin() as conn:
            df.to_sql(table, conn, if_exists="replace", index=False, chunksize=chunksize, method="multi"
                      if engine.dialect.name == "postgresql" else None)
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_event ON {table} (event_id)"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_user ON {table} (username)"))
        # Also keep a columnar copy for fast analytical reads / S3 upload.
        parquet_path = config.PROCESSED_DIR / f"{table}.parquet"
        df.to_parquet(parquet_path, index=False)

    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
    cols = [c["name"] for c in inspect(engine).get_columns(table)]
    log.info("Warehouse table %s now holds %d rows, %d columns", table, count, len(cols))
    if count != len(df):
        raise RuntimeError(f"Row count mismatch after load: wrote {len(df)}, table has {count}")
    return count


def read_features(db_url: str = config.WAREHOUSE_DB_URL, table: str = config.FEATURES_TABLE) -> pd.DataFrame:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        return pd.read_sql(text(f"SELECT * FROM {table}"), conn)
