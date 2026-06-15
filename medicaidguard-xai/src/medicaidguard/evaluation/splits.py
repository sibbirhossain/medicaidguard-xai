"""Time-based and group-aware splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SplitConfig


def time_split(claims: pd.DataFrame, cfg: SplitConfig) -> pd.Series:
    """Label each claim train/valid/test by service date.

    Program-integrity models are always deployed forward in time, so a random
    split would report a number the deployed system can never achieve.
    """
    d = claims.sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    sd = d["service_date"]
    out = pd.Series("test", index=d.index, dtype="object")
    out[sd <= pd.Timestamp(cfg.train_end)] = "train"
    out[(sd > pd.Timestamp(cfg.train_end)) & (sd <= pd.Timestamp(cfg.valid_end))] = "valid"
    return out


def group_aware_folds(groups: pd.Series, n_splits: int = 5, seed: int = 0):
    """Yield (train_idx, valid_idx) keeping every entity wholly on one side.

    Used for the cross-validated variants. Without this, the same caregiver
    appears in train and validation and the model can memorise the entity
    rather than the behaviour.
    """
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(groups))
    yield from gkf.split(idx, groups=groups.to_numpy())
