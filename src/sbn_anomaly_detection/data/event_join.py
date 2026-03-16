"""Join TPC and PMT event arrays by (run, event) keys.

Both sub-detectors record events independently; this module aligns them so that
the fusion autoencoder sees matched (TPC, PMT) pairs.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Expected key columns present in every batch
_RUN_COL = "run"
_EVT_COL = "event"


def _batch_to_dataframe(
    batch: dict,
    feature_cols: list[str],
    key_cols: tuple[str, str] = (_RUN_COL, _EVT_COL),
) -> pd.DataFrame:
    """Convert a streaming batch dict to a pandas DataFrame.

    Parameters
    ----------
    batch:
        Mapping of branch name → 1-D numpy array.
    feature_cols:
        Branch names that carry physics features (not keys).
    key_cols:
        Names of the run and event-number branches used as join keys.

    Returns
    -------
    pd.DataFrame with columns = key_cols + feature_cols.
    """
    run_col, evt_col = key_cols
    all_cols = list(key_cols) + feature_cols
    data = {col: batch[col] for col in all_cols}
    return pd.DataFrame(data)


def join_tpc_pmt(
    tpc_batch: dict,
    pmt_batch: dict,
    tpc_feature_cols: list[str],
    pmt_feature_cols: list[str],
    key_cols: Tuple[str, str] = (_RUN_COL, _EVT_COL),
    how: str = "inner",
) -> Tuple[np.ndarray, np.ndarray]:
    """Inner-join TPC and PMT batches by run/event keys.

    Parameters
    ----------
    tpc_batch:
        Dict of arrays from the TPC streaming reader.
    pmt_batch:
        Dict of arrays from the PMT streaming reader.
    tpc_feature_cols:
        TPC feature branch names (excluding key columns).
    pmt_feature_cols:
        PMT feature branch names (excluding key columns).
    key_cols:
        Column names used as join keys (default: ``("run", "event")``).
    how:
        Join type passed to :func:`pandas.DataFrame.merge`. Defaults to ``"inner"``.

    Returns
    -------
    tpc_features : np.ndarray, shape (N, len(tpc_feature_cols))
    pmt_features : np.ndarray, shape (N, len(pmt_feature_cols))
        Aligned feature arrays for the matched events.
    """
    df_tpc = _batch_to_dataframe(tpc_batch, tpc_feature_cols, key_cols)
    df_pmt = _batch_to_dataframe(pmt_batch, pmt_feature_cols, key_cols)

    merged = df_tpc.merge(df_pmt, on=list(key_cols), how=how, suffixes=("_tpc", "_pmt"))

    n_matched = len(merged)
    n_tpc = len(df_tpc)
    n_pmt = len(df_pmt)
    logger.debug(
        "Event join: TPC=%d, PMT=%d → matched=%d (%.1f%%)",
        n_tpc,
        n_pmt,
        n_matched,
        100.0 * n_matched / max(n_tpc, 1),
    )

    tpc_arr = merged[tpc_feature_cols].to_numpy(dtype=np.float32)
    pmt_arr = merged[pmt_feature_cols].to_numpy(dtype=np.float32)
    return tpc_arr, pmt_arr


def join_batches_streaming(
    tpc_batches,
    pmt_batches,
    tpc_feature_cols: list[str],
    pmt_feature_cols: list[str],
    key_cols: Tuple[str, str] = (_RUN_COL, _EVT_COL),
):
    """Generator that zips two streaming sources and yields joined arrays.

    Parameters
    ----------
    tpc_batches, pmt_batches:
        Iterables of batch dicts (e.g. from :func:`~data.root_stream.stream_arrays`).

    Yields
    ------
    (tpc_features, pmt_features) : Tuple[np.ndarray, np.ndarray]
    """
    for tpc_batch, pmt_batch in zip(tpc_batches, pmt_batches):
        tpc_arr, pmt_arr = join_tpc_pmt(
            tpc_batch,
            pmt_batch,
            tpc_feature_cols,
            pmt_feature_cols,
            key_cols=key_cols,
        )
        if tpc_arr.shape[0] == 0:
            logger.warning("Empty batch after join; skipping.")
            continue
        yield tpc_arr, pmt_arr
