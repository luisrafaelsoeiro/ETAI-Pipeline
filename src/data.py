"""Data loading."""
import os

import pandas as pd

# file extension -> pandas reader (week 9: not only CSV). Extra options (a different separator,
# encoding, decimal mark, ...) come from `data.read_options` in the config.
_READERS = {
    ".csv": pd.read_csv,
    ".txt": pd.read_csv,
    ".tsv": lambda path, **options: pd.read_csv(path, **{"sep": "\t", **options}),
    ".parquet": pd.read_parquet,          # needs pyarrow
    ".xlsx": pd.read_excel,               # needs openpyxl
    ".xls": pd.read_excel,
    ".json": pd.read_json,
}


def load_data(path: str, **read_options) -> pd.DataFrame:
    """
    Load the raw dataset at `path`. The reader follows the file extension (.csv, .txt, .tsv,
    .parquet, .xlsx, .xls, .json); `read_options` are passed on to it (e.g. `sep=";"` for a
    semicolon-separated CSV). Raises a clear error for a missing, unsupported or empty file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path} (paths are relative to the folder you run main.py from)")
    extension = os.path.splitext(path)[1].lower()
    if extension not in _READERS:
        raise ValueError(f"Unsupported data file type '{extension}' ({path}). Supported: {sorted(_READERS)}")
    try:
        df = _READERS[extension](path, **read_options)
    except ImportError as error:
        raise ImportError(f"Reading {extension} files needs an extra package: {error}") from error
    if df.empty:
        raise ValueError(f"The data file {path} has no rows")
    if df.columns.duplicated().any():
        raise ValueError(f"The data file {path} has repeated column names: "
                         f"{sorted(set(df.columns[df.columns.duplicated()]))}")
    return df
