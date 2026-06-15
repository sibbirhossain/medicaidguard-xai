"""Feature pipeline.

`FeaturePipeline.fit_transform(train_window)` then `.transform(all_rows)` is the
contract. Anything that learns a statistic (peer medians, network degrees,
category vocabularies) is fitted on the training window only; anything that is
purely row-local or strictly backward-looking is computed for all rows at once.

Feature blocks are tagged so the ablation study (Experiment 2) can switch whole
families on and off without re-implementing the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .billing import build_billing_features
from .evv import build_evv_features
from .network import NetworkFeatureTransformer
from .temporal import PeerDeviationTransformer, build_temporal_features

CATEGORICAL = ["service_code", "claim_status", "device_type", "verification_method"]

BLOCKS = ("billing", "evv", "temporal", "statistical", "network")


class FeaturePipeline:
    def __init__(self, blocks: tuple[str, ...] = BLOCKS):
        self.blocks = tuple(blocks)
        self.peer_ = PeerDeviationTransformer()
        self.net_ = NetworkFeatureTransformer()
        self.columns_: list[str] = []
        self.categories_: dict[str, list[str]] = {}
        self.medians_: pd.Series | None = None
        self._fitted = False

    # -- assembly -----------------------------------------------------------

    def _assemble(self, claims, evv, patients) -> pd.DataFrame:
        parts = []
        base = build_billing_features(claims)
        order = base[["claim_id"]].copy()
        if "billing" in self.blocks:
            parts.append(base.drop(columns=["claim_id"]))
        else:
            # Billing columns are still needed as inputs to peer deviations even
            # when the block is ablated out of the model; keep a minimal set.
            parts.append(base[["billed_hours", "billed_amount", "hourly_rate",
                               "service_code", "claim_status"]])
        if "evv" in self.blocks:
            ev = build_evv_features(claims, evv, patients)
            parts.append(ev.drop(columns=["claim_id"]))
        if "temporal" in self.blocks:
            tp = build_temporal_features(claims)
            parts.append(tp.drop(columns=["claim_id"]))
        out = pd.concat([order] + parts, axis=1)
        return out.loc[:, ~out.columns.duplicated()]

    # -- fit / transform ----------------------------------------------------

    def fit(self, claims, evv, patients, train_mask: pd.Series) -> FeaturePipeline:
        feats = self._assemble(claims, evv, patients)
        ordered_claims = claims.sort_values(["service_date", "claim_id"]).reset_index(drop=True)
        mask = train_mask.reindex(ordered_claims.index).fillna(False).to_numpy() \
            if isinstance(train_mask, pd.Series) else np.asarray(train_mask)

        self.peer_.fit(feats.loc[mask])
        self.net_.fit(ordered_claims.loc[mask])

        full = self._finalize(feats, ordered_claims, fitting=True)
        self.columns_ = [c for c in full.columns if c != "claim_id"]
        self.medians_ = full.loc[mask, self.columns_].median(numeric_only=True)
        self._fitted = True
        return self

    def transform(self, claims, evv, patients) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeaturePipeline.fit must be called before transform")
        feats = self._assemble(claims, evv, patients)
        ordered_claims = claims.sort_values(["service_date", "claim_id"]).reset_index(drop=True)
        full = self._finalize(feats, ordered_claims, fitting=False)
        for c in self.columns_:
            if c not in full.columns:
                full[c] = np.nan
        full = full[["claim_id"] + self.columns_]
        num = full[self.columns_].select_dtypes(include=[np.number]).columns
        full[num] = full[num].replace([np.inf, -np.inf], np.nan)
        full[num] = full[num].fillna(self.medians_.reindex(num))
        full[num] = full[num].fillna(0.0)
        return full

    def fit_transform(self, claims, evv, patients, train_mask) -> pd.DataFrame:
        return self.fit(claims, evv, patients, train_mask).transform(claims, evv, patients)

    # -- internals ----------------------------------------------------------

    def _finalize(self, feats: pd.DataFrame, ordered_claims: pd.DataFrame,
                  fitting: bool) -> pd.DataFrame:
        out = feats.copy()
        if "statistical" in self.blocks:
            out = pd.concat([out, self.peer_.transform(out)], axis=1)
        if "network" in self.blocks:
            net = self.net_.transform(ordered_claims)
            out = pd.concat([out, net.drop(columns=["claim_id"])], axis=1)

        for col in CATEGORICAL:
            if col not in out.columns:
                continue
            if fitting:
                self.categories_[col] = sorted(out[col].astype(str).unique().tolist())
            cats = self.categories_.get(col, [])
            s = out[col].astype(str)
            for c in cats:
                out[f"{col}__{c}"] = (s == c).astype("float64")
            out = out.drop(columns=[col])

        if "billing" not in self.blocks:
            drop = [c for c in ("billed_hours", "billed_amount", "hourly_rate")
                    if c in out.columns]
            out = out.drop(columns=drop)
        return out.loc[:, ~out.columns.duplicated()]
