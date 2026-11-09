import sqlite3
import pandas as pd

DB = "data/security.db"


def save(df: pd.DataFrame, table="events"):
    with sqlite3.connect(DB) as con:
        df.to_sql(table, con, if_exists="replace", index=False)


def load(table="events") -> pd.DataFrame:
    with sqlite3.connect(DB) as con:
        return pd.read_sql(f"SELECT * FROM {table}", con)
