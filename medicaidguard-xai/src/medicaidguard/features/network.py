"""Provider-caregiver-patient relationship features.

The interaction graph is built from the *training window only* and applied to
later rows as a lookup. Rebuilding it on the full period would let test-period
relationships describe themselves, and the coordinated-activity scenario would
become trivially detectable for the wrong reason.

Entities unseen during training get neutral fill values, which is also what
happens in production when a new caregiver appears.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class NetworkFeatureTransformer:
    def __init__(self):
        self.cg_degree_: pd.Series | None = None
        self.pt_degree_: pd.Series | None = None
        self.prov_cg_count_: pd.Series | None = None
        self.cg_share_in_provider_: pd.Series | None = None
        self.pair_count_: pd.Series | None = None
        self.pair_share_: pd.Series | None = None
        self.defaults_: dict[str, float] = {}

    def fit(self, claims: pd.DataFrame) -> NetworkFeatureTransformer:
        c = claims
        # Degree = number of distinct counterparties.
        self.cg_degree_ = c.groupby("caregiver_id", observed=True)["patient_id"].nunique()
        self.pt_degree_ = c.groupby("patient_id", observed=True)["caregiver_id"].nunique()
        self.prov_cg_count_ = c.groupby("provider_id", observed=True)["caregiver_id"].nunique()

        # Share of an agency's volume carried by one caregiver: a ring inside a
        # large agency shows up as an unusual concentration.
        vol = c.groupby(["provider_id", "caregiver_id"], observed=True).size().rename("n")
        prov_tot = vol.groupby("provider_id", observed=True).sum()
        self.cg_share_in_provider_ = (vol / prov_tot.reindex(vol.index.get_level_values(0)).to_numpy())

        pair = c.groupby(["caregiver_id", "patient_id"], observed=True).size().rename("n")
        cg_tot = pair.groupby("caregiver_id", observed=True).sum()
        self.pair_count_ = pair
        self.pair_share_ = pair / cg_tot.reindex(pair.index.get_level_values(0)).to_numpy()

        self.defaults_ = {
            "caregiver_degree": float(self.cg_degree_.median()),
            "patient_degree": float(self.pt_degree_.median()),
            "provider_caregiver_count": float(self.prov_cg_count_.median()),
            "caregiver_share_of_provider": float(self.cg_share_in_provider_.median()),
            "pair_claim_count": 0.0,
            "pair_share_of_caregiver": 0.0,
        }
        return self

    def transform(self, claims: pd.DataFrame) -> pd.DataFrame:
        c = claims
        out = pd.DataFrame(index=c.index)
        out["claim_id"] = c["claim_id"].to_numpy()

        out["caregiver_degree"] = self.cg_degree_.reindex(c["caregiver_id"]).to_numpy()
        out["patient_degree"] = self.pt_degree_.reindex(c["patient_id"]).to_numpy()
        out["provider_caregiver_count"] = self.prov_cg_count_.reindex(c["provider_id"]).to_numpy()

        idx = pd.MultiIndex.from_arrays([c["provider_id"], c["caregiver_id"]])
        out["caregiver_share_of_provider"] = self.cg_share_in_provider_.reindex(idx).to_numpy()

        pidx = pd.MultiIndex.from_arrays([c["caregiver_id"], c["patient_id"]])
        out["pair_claim_count"] = self.pair_count_.reindex(pidx).to_numpy()
        out["pair_share_of_caregiver"] = self.pair_share_.reindex(pidx).to_numpy()

        # Relationship novelty: a pairing never seen in the baseline period.
        out["pair_is_new"] = out["pair_claim_count"].isna().astype("float64")

        for col, val in self.defaults_.items():
            out[col] = out[col].fillna(val)
        # Inverse degree highlights caregivers serving very few patients very often.
        out["caregiver_concentration"] = out["pair_share_of_caregiver"] / np.maximum(
            out["caregiver_degree"], 1.0
        )
        return out
