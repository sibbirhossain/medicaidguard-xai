"""Reproducible synthetic Medicaid + EVV data generator.

Why synthetic data is needed at all
-----------------------------------
No public dataset contains linked Medicaid personal-care claims *and* EVV
check-in/check-out telemetry with verified fraud labels. EVV data is
operational PHI held by state agencies and their aggregators; it is not
published. The dataset discovery report (reports/dataset_discovery.md)
documents that search. This generator therefore exists to support controlled
EVV-specific experiments, and every label it produces is a *simulated* label.

Design constraints
------------------
* Normal behaviour must be genuinely normal: real caregivers work irregular
  hours, GPS drifts, visits run long, paperwork is late. If "normal" is too
  clean, every model scores near 1.0 and the benchmark is meaningless.
* Suspicious scenarios are injected as *behavioural* changes, never as a flag
  column. The model has to recover them from billing/EVV/temporal structure.
* Scenario metadata lives in a separate table (`scenario_labels`) so it cannot
  leak into the feature matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ALL_SCENARIOS, GeneratorConfig

# --- geography helpers -----------------------------------------------------

_EARTH_R_M = 6_371_000.0
_M_PER_DEG_LAT = 111_320.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in metres."""
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * _EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _td_min(x):
    """Timedelta from float minutes, rounded to whole seconds.

    pandas >= 3.0 defaults datetime columns to microsecond resolution, so
    nanosecond-precision offsets derived from float arithmetic cannot be
    assigned back losslessly. Rounding to seconds keeps every timestamp
    representable and matches the real world, where EVV clocks are not
    meaningful below the second.
    """
    return pd.to_timedelta(
        np.round(np.asarray(x, dtype=float) * 60.0).astype("int64"), unit="s"
    )


def _jitter_coords(rng, lat, lon, metres):
    """Add isotropic positional noise of the given scale (metres)."""
    dlat = rng.normal(0, metres, size=len(lat)) / _M_PER_DEG_LAT
    dlon = rng.normal(0, metres, size=len(lon)) / (
        _M_PER_DEG_LAT * np.cos(np.radians(np.asarray(lat, dtype=float)))
    )
    return lat + dlat, lon + dlon


SERVICE_CODES = ["T1019", "T1020", "S5125", "S5130", "G0156"]
_CODE_RATE = {"T1019": 27.5, "T1020": 31.0, "S5125": 24.0, "S5130": 22.5, "G0156": 29.0}
PROVIDER_TYPES = ["LHCSA", "FI", "CHHA", "MLTC-Contracted"]
SPECIALTIES = ["personal_care", "home_health_aide", "consumer_directed", "respite"]
DEVICES = ["mobile_app", "telephony_ivr", "fob", "web_portal"]
VERIFICATION = ["gps", "telephony", "fixed_device", "manual_entry"]
DEMOGRAPHIC_GROUPS = ["group_a", "group_b", "group_c"]
AGE_BANDS = ["18-44", "45-64", "65-79", "80+"]


class SyntheticMedicaidEVVGenerator:
    """Generates a linked provider / caregiver / patient / claim / EVV dataset."""

    def __init__(self, cfg: GeneratorConfig | None = None):
        self.cfg = cfg or GeneratorConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    # -- entity tables ------------------------------------------------------

    def _regions(self):
        rng = self.rng
        n = self.cfg.n_regions
        # Spread region centres across a plausible mid-latitude bounding box.
        lat = rng.uniform(40.5, 43.0, n)
        lon = rng.uniform(-79.0, -73.0, n)
        return pd.DataFrame({
            "service_region": [f"R{i:02d}" for i in range(n)],
            "center_lat": lat,
            "center_lon": lon,
        })

    def _providers(self, regions):
        rng, n = self.rng, self.cfg.n_providers
        return pd.DataFrame({
            "provider_id": [f"PRV{i:04d}" for i in range(n)],
            "provider_type": rng.choice(PROVIDER_TYPES, n, p=[0.45, 0.25, 0.20, 0.10]),
            "specialty": rng.choice(SPECIALTIES, n),
            "enrollment_date": pd.Timestamp("2015-01-01")
            + pd.to_timedelta(rng.integers(0, 3200, n), unit="D"),
            "service_region": rng.choice(regions["service_region"], n),
            "active_status": rng.random(n) > 0.03,
            "historical_claim_volume": rng.integers(200, 15_000, n),
        })

    def _caregivers(self, providers):
        rng, n = self.rng, self.cfg.n_caregivers
        # Caregiver counts per agency follow a heavy right tail: a few large
        # agencies employ most of the workforce.
        weights = rng.pareto(1.6, len(providers)) + 1.0
        weights /= weights.sum()
        prov = rng.choice(providers["provider_id"], n, p=weights)
        prov_region = providers.set_index("provider_id")["service_region"]
        return pd.DataFrame({
            "caregiver_id": [f"CG{i:05d}" for i in range(n)],
            "provider_id": prov,
            "employment_start_date": pd.Timestamp("2018-01-01")
            + pd.to_timedelta(rng.integers(0, 2200, n), unit="D"),
            "service_region": prov_region.loc[prov].to_numpy(),
            "qualification_status": rng.choice(
                ["certified", "certified", "certified", "provisional", "lapsed"], n
            ),
            "historical_visit_volume": rng.integers(10, 4000, n),
        })

    def _patients(self, regions):
        rng, n = self.rng, self.cfg.n_patients
        reg = rng.choice(regions["service_region"], n)
        centers = regions.set_index("service_region")
        clat = centers.loc[reg, "center_lat"].to_numpy()
        clon = centers.loc[reg, "center_lon"].to_numpy()
        # Homes scattered ~15 km around the regional centre.
        lat, lon = _jitter_coords(rng, clat, clon, 15_000)
        return pd.DataFrame({
            "patient_id": [f"PT{i:06d}" for i in range(n)],
            "service_region": reg,
            "demographic_group": rng.choice(DEMOGRAPHIC_GROUPS, n, p=[0.5, 0.3, 0.2]),
            "age_band": rng.choice(AGE_BANDS, n, p=[0.12, 0.22, 0.36, 0.30]),
            "authorized_hours_weekly": np.round(
                np.clip(rng.gamma(6.0, 4.0, n), 4, 84) / 2
            ) * 2,
            "service_type": rng.choice(["personal_care", "home_health", "respite"], n,
                                       p=[0.6, 0.3, 0.1]),
            "eligibility_status": rng.choice(["active", "active", "active", "pending"], n),
            "home_lat": lat,
            "home_lon": lon,
        })

    def _authorizations(self, patients):
        rng = self.rng
        start = pd.Timestamp(self.cfg.start_date)
        n = len(patients)
        offs = rng.integers(-120, 30, n)
        auth_start = start + pd.to_timedelta(offs, unit="D")
        return pd.DataFrame({
            "authorization_id": [f"AUTH{i:06d}" for i in range(n)],
            "patient_id": patients["patient_id"].to_numpy(),
            "authorized_service_code": rng.choice(SERVICE_CODES, n),
            "authorized_start_date": auth_start,
            "authorized_end_date": auth_start + pd.to_timedelta(rng.integers(180, 400, n), unit="D"),
            "authorized_hours_weekly": patients["authorized_hours_weekly"].to_numpy(),
            "authorization_status": rng.choice(["approved", "approved", "approved", "expired"], n,
                                               p=[0.35, 0.35, 0.24, 0.06]),
        })

    # -- baseline claims ----------------------------------------------------

    def _baseline_claims(self, providers, caregivers, patients, authorizations):
        """Create normal-looking claims from persistent caregiver-patient pairs."""
        rng = self.cfg_rng()
        n = self.cfg.n_claims
        start = pd.Timestamp(self.cfg.start_date)
        end = pd.Timestamp(self.cfg.end_date)
        span_days = (end - start).days

        # Persistent assignment: each patient has 1-3 regular caregivers drawn
        # from agencies operating in their region where possible.
        cg_by_region = {r: g["caregiver_id"].to_numpy()
                        for r, g in caregivers.groupby("service_region", observed=True)}
        all_cg = caregivers["caregiver_id"].to_numpy()
        assign_pt, assign_cg = [], []
        for pid, reg in zip(patients["patient_id"], patients["service_region"], strict=True):
            pool = cg_by_region.get(reg, all_cg)
            if len(pool) == 0:
                pool = all_cg
            k = int(rng.integers(1, 4))
            for c in rng.choice(pool, size=min(k, len(pool)), replace=False):
                assign_pt.append(pid)
                assign_cg.append(c)
        pairs = pd.DataFrame({"patient_id": assign_pt, "caregiver_id": assign_cg})

        # Patients with more authorised hours generate more visits.
        pt_hours = patients.set_index("patient_id")["authorized_hours_weekly"]
        w = pt_hours.loc[pairs["patient_id"]].to_numpy().astype(float)
        w = w / w.sum()
        idx = rng.choice(len(pairs), size=n, p=w)
        claims = pairs.iloc[idx].reset_index(drop=True)

        cg_prov = caregivers.set_index("caregiver_id")["provider_id"]
        claims["provider_id"] = cg_prov.loc[claims["caregiver_id"]].to_numpy()
        auth = authorizations.set_index("patient_id")
        claims["authorization_id"] = auth.loc[claims["patient_id"], "authorization_id"].to_numpy()
        claims["service_code"] = auth.loc[claims["patient_id"], "authorized_service_code"].to_numpy()
        claims["authorized_hours"] = (
            auth.loc[claims["patient_id"], "authorized_hours_weekly"].to_numpy() / 7.0
        )

        # Service dates: weekday-weighted, mild seasonality.
        day_off = rng.integers(0, span_days + 1, n)
        service_date = start + pd.to_timedelta(day_off, unit="D")
        dow = service_date.dayofweek.to_numpy()
        # Thin out weekends: keep ~35% of weekend draws, resample the rest.
        weekend = dow >= 5
        redraw = weekend & (rng.random(n) > 0.35)
        while redraw.any():
            day_off = np.where(redraw, rng.integers(0, span_days + 1, n), day_off)
            service_date = start + pd.to_timedelta(day_off, unit="D")
            dow = service_date.dayofweek.to_numpy()
            weekend = dow >= 5
            redraw = weekend & (rng.random(n) > 0.35)

        # Shift start: bimodal (morning / afternoon), minutes on a 15-min grid.
        morning = rng.random(n) < 0.62
        start_hour = np.where(morning, rng.normal(8.5, 1.1, n), rng.normal(14.0, 1.6, n))
        start_hour = np.clip(start_hour, 5.5, 21.0)
        start_min = np.round(start_hour * 4) / 4  # quarter-hour grid

        # Durations track authorised daily hours with real variability.
        base = np.clip(claims["authorized_hours"].to_numpy(), 0.5, 12.0)
        dur = np.clip(rng.normal(base, 0.45 + 0.12 * base), 0.5, 14.0)
        dur = np.round(dur * 4) / 4

        billing_start = service_date + _td_min(start_min)
        billing_end = billing_start + _td_min(dur)

        rate = pd.Series(_CODE_RATE)[claims["service_code"]].to_numpy()
        claims["service_date"] = service_date
        claims["billing_start_time"] = billing_start
        claims["billing_end_time"] = billing_end
        claims["billed_hours"] = dur
        claims["billed_amount"] = np.round(dur * rate * rng.normal(1.0, 0.02, n), 2)
        claims["claim_status"] = rng.choice(["paid", "paid", "paid", "pending", "denied"], n,
                                            p=[0.4, 0.3, 0.2, 0.07, 0.03])
        # Submission lag: usually days, occasionally weeks.
        lag_d = np.where(rng.random(n) < 0.85, rng.integers(1, 8, n), rng.integers(8, 45, n))
        claims["submission_timestamp"] = (
            claims["billing_end_time"]
            + pd.to_timedelta(lag_d, unit="D")
            + _td_min(rng.integers(0, 24 * 60, n))
        )
        claims["claim_id"] = [f"CLM{i:07d}" for i in range(n)]
        return claims

    def cfg_rng(self):
        return self.rng

    # -- EVV events ---------------------------------------------------------

    def _evv(self, claims, patients):
        rng = self.rng
        n = len(claims)
        homes = patients.set_index("patient_id")[["home_lat", "home_lon"]]
        lat = homes.loc[claims["patient_id"], "home_lat"].to_numpy()
        lon = homes.loc[claims["patient_id"], "home_lon"].to_numpy()

        # EVV clocks differ from billed times by a few minutes in either
        # direction; rounding to the billed quarter-hour is routine practice.
        #
        # The check-out offset is drawn RELATIVE to the check-in offset rather
        # than independently. Drawing them independently lets a positive
        # check-in jitter and a negative check-out jitter invert a short visit,
        # producing check-out-before-check-in rows that are a generator artefact
        # rather than a modelled behaviour. Real EVV systems reject such rows at
        # capture, so they must not appear in normal data: if they did, the
        # duration features would carry a spurious signal on ordinary claims.
        in_off = rng.normal(0, 6.5, n)
        dur_jitter = rng.normal(0, 8.0, n)
        billed_min = claims["billed_hours"].to_numpy() * 60.0
        # Verified duration stays positive and never collapses below a floor of
        # five minutes or a quarter of the billed time, whichever is smaller.
        floor = np.minimum(5.0, billed_min * 0.25)
        evv_minutes = np.maximum(billed_min + dur_jitter, floor)
        check_in = claims["billing_start_time"] + _td_min(in_off)
        check_out = check_in + _td_min(evv_minutes)

        acc = np.clip(rng.gamma(2.2, 14.0, n), 3, 400)
        in_lat, in_lon = _jitter_coords(rng, lat, lon, self.cfg.coordinate_noise_m)
        out_lat, out_lon = _jitter_coords(rng, lat, lon, self.cfg.coordinate_noise_m)

        device = rng.choice(DEVICES, n, p=[0.58, 0.22, 0.12, 0.08])
        method = np.where(device == "mobile_app", "gps",
                  np.where(device == "telephony_ivr", "telephony",
                  np.where(device == "fob", "fixed_device", "manual_entry")))

        evv = pd.DataFrame({
            "evv_event_id": [f"EVV{i:07d}" for i in range(n)],
            "claim_id": claims["claim_id"].to_numpy(),
            "caregiver_id": claims["caregiver_id"].to_numpy(),
            "patient_id": claims["patient_id"].to_numpy(),
            "check_in_time": check_in.to_numpy(),
            "check_out_time": check_out.to_numpy(),
            "check_in_latitude": in_lat,
            "check_in_longitude": in_lon,
            "check_out_latitude": out_lat,
            "check_out_longitude": out_lon,
            "device_type": device,
            "verification_method": method,
            "location_accuracy_m": acc,
            "event_status": "complete",
        })
        # Non-GPS methods legitimately carry no coordinates.
        no_gps = np.isin(evv["verification_method"], ["telephony", "manual_entry"])
        for c in ["check_in_latitude", "check_in_longitude",
                  "check_out_latitude", "check_out_longitude"]:
            evv.loc[no_gps, c] = np.nan
        evv.loc[no_gps, "location_accuracy_m"] = np.nan

        # Benign missing EVV (device failure, connectivity, late reconciliation).
        drop = rng.random(n) < self.cfg.missing_evv_rate
        evv = evv.loc[~drop].reset_index(drop=True)
        return evv

    # -- scenario injection -------------------------------------------------

    def _inject(self, claims, evv, patients, caregivers):
        """Mutate a subset of claims/EVV rows to embody suspicious scenarios.

        Returns (claims, evv, labels). Some scenarios append rows (duplicates),
        so the frames are rebuilt rather than edited in place everywhere.
        """
        rng = self.rng
        cfg = self.cfg
        scenarios = [s for s in cfg.scenarios if s in ALL_SCENARIOS]
        n_target = int(round(len(claims) * cfg.suspicious_prevalence))
        if not scenarios or n_target == 0:
            labels = pd.DataFrame({
                "claim_id": claims["claim_id"], "is_suspicious": 0,
                "scenario": "none", "severity": 0.0,
            })
            return claims, evv, labels

        claims = claims.set_index("claim_id", drop=False)
        evv = evv.set_index("claim_id", drop=False)
        homes = patients.set_index("patient_id")[["home_lat", "home_lon"]]

        pool = rng.permutation(claims.index.to_numpy())[:n_target]
        chunks = np.array_split(pool, len(scenarios))
        label_rows = []
        new_claims, new_evv = [], []

        for scenario, ids in zip(scenarios, chunks, strict=False):
            if len(ids) == 0:
                continue
            sev = rng.uniform(0.45, 1.0, len(ids))
            self._apply_scenario(scenario, ids, sev, claims, evv, homes,
                                 caregivers, new_claims, new_evv, rng)
            label_rows.append(pd.DataFrame({
                "claim_id": ids, "is_suspicious": 1,
                "scenario": scenario, "severity": sev,
            }))

        claims = claims.reset_index(drop=True)
        evv = evv.reset_index(drop=True)
        if new_claims:
            extra_c = pd.concat(new_claims, ignore_index=True)
            claims = pd.concat([claims, extra_c], ignore_index=True)
            label_rows.append(pd.DataFrame({
                "claim_id": extra_c["claim_id"],
                "is_suspicious": 1,
                "scenario": extra_c["_scenario"].to_numpy(),
                "severity": 1.0,
            }))
            claims = claims.drop(columns=["_scenario"])
        if new_evv:
            evv = pd.concat([evv] + new_evv, ignore_index=True)

        labels = pd.concat(label_rows, ignore_index=True) if label_rows else pd.DataFrame()
        labels = labels.drop_duplicates("claim_id")
        clean = claims.loc[~claims["claim_id"].isin(labels["claim_id"]), ["claim_id"]].copy()
        clean["is_suspicious"] = 0
        clean["scenario"] = "none"
        clean["severity"] = 0.0
        labels = pd.concat([labels, clean], ignore_index=True)
        return claims, evv, labels

    def _apply_scenario(self, scenario, ids, sev, claims, evv, homes,
                        caregivers, new_claims, new_evv, rng):
        has_evv = evv.index.intersection(ids)

        if scenario == "overlapping_visits":
            # Same caregiver, same time window, different patient. Physically
            # impossible; detectable only by cross-claim comparison.
            src = claims.loc[ids]
            dup = src.copy()
            dup["claim_id"] = [f"CLMX{i:06d}O" for i in range(len(dup))]
            other = rng.choice(claims["patient_id"].to_numpy(), len(dup))
            dup["patient_id"] = other
            shift = _td_min(rng.uniform(-0.5, 0.5, len(dup)))
            dup["billing_start_time"] = dup["billing_start_time"] + shift
            dup["billing_end_time"] = dup["billing_end_time"] + shift
            dup["_scenario"] = scenario
            new_claims.append(dup.reset_index(drop=True))

        elif scenario == "authorization_exceedance":
            mult = 1.6 + 2.4 * sev
            claims.loc[ids, "billed_hours"] = np.round(
                claims.loc[ids, "authorized_hours"].to_numpy() * mult * 4) / 4
            claims.loc[ids, "billing_end_time"] = (
                claims.loc[ids, "billing_start_time"]
                + _td_min(claims.loc[ids, "billed_hours"])
            )
            rate = pd.Series(_CODE_RATE)[claims.loc[ids, "service_code"]].to_numpy()
            claims.loc[ids, "billed_amount"] = np.round(
                claims.loc[ids, "billed_hours"].to_numpy() * rate, 2)

        elif scenario == "implausible_travel":
            # Move the check-in far from the patient's home so that the implied
            # travel speed from the previous visit is impossible.
            if len(has_evv):
                lat = evv.loc[has_evv, "check_in_latitude"].to_numpy()
                lon = evv.loc[has_evv, "check_in_longitude"].to_numpy()
                s = sev[: len(has_evv)] if len(sev) >= len(has_evv) else rng.uniform(.5, 1, len(has_evv))
                shift_m = 40_000 + 120_000 * s
                nl, no = _jitter_coords(rng, np.nan_to_num(lat, nan=41.5),
                                        np.nan_to_num(lon, nan=-75.0), shift_m)
                evv.loc[has_evv, "check_in_latitude"] = nl
                evv.loc[has_evv, "check_in_longitude"] = no
                evv.loc[has_evv, "check_out_latitude"] = nl
                evv.loc[has_evv, "check_out_longitude"] = no
                evv.loc[has_evv, "verification_method"] = "gps"
                evv.loc[has_evv, "device_type"] = "mobile_app"

        elif scenario == "evv_duration_mismatch":
            if len(has_evv):
                s = rng.uniform(0.45, 1.0, len(has_evv))
                # Verified visit is far shorter than the hours billed.
                billed = claims.loc[has_evv, "billed_hours"].to_numpy()
                actual = np.clip(billed * (1 - 0.45 - 0.4 * s), 0.25, None)
                evv.loc[has_evv, "check_out_time"] = (
                    evv.loc[has_evv, "check_in_time"]
                    + _td_min(actual)
                )

        elif scenario == "missing_evv":
            evv.drop(index=has_evv, inplace=True, errors="ignore")

        elif scenario == "duplicate_claim":
            src = claims.loc[ids]
            dup = src.copy()
            dup["claim_id"] = [f"CLMX{i:06d}D" for i in range(len(dup))]
            dup["submission_timestamp"] = dup["submission_timestamp"] + pd.Timedelta(hours=3)
            dup["_scenario"] = scenario
            new_claims.append(dup.reset_index(drop=True))

        elif scenario == "near_duplicate_claim":
            src = claims.loc[ids]
            dup = src.copy()
            dup["claim_id"] = [f"CLMX{i:06d}N" for i in range(len(dup))]
            jit = _td_min(rng.integers(5, 20, len(dup)))
            dup["billing_start_time"] = dup["billing_start_time"] + jit
            dup["billing_end_time"] = dup["billing_end_time"] + jit
            dup["billed_hours"] = dup["billed_hours"] * rng.uniform(0.97, 1.03, len(dup))
            dup["submission_timestamp"] = dup["submission_timestamp"] + pd.Timedelta(days=1)
            dup["_scenario"] = scenario
            new_claims.append(dup.reset_index(drop=True))

        elif scenario == "billing_spike":
            claims.loc[ids, "billed_hours"] = np.clip(
                claims.loc[ids, "billed_hours"].to_numpy() * (2.0 + 2.0 * sev), 0.5, 24)
            claims.loc[ids, "billing_end_time"] = (
                claims.loc[ids, "billing_start_time"]
                + _td_min(claims.loc[ids, "billed_hours"]))
            rate = pd.Series(_CODE_RATE)[claims.loc[ids, "service_code"]].to_numpy()
            claims.loc[ids, "billed_amount"] = np.round(
                claims.loc[ids, "billed_hours"].to_numpy() * rate, 2)

        elif scenario == "excessive_workload":
            # Re-assign these claims to a small set of caregivers so their daily
            # workload becomes implausible.
            hot = rng.choice(caregivers["caregiver_id"].to_numpy(),
                             max(2, len(ids) // 25), replace=False)
            pick = rng.choice(hot, len(ids))
            claims.loc[ids, "caregiver_id"] = pick
            cg_prov = caregivers.set_index("caregiver_id")["provider_id"]
            claims.loc[ids, "provider_id"] = cg_prov.loc[pick].to_numpy()
            if len(has_evv):
                evv.loc[has_evv, "caregiver_id"] = claims.loc[has_evv, "caregiver_id"].to_numpy()

        elif scenario == "provider_caregiver_concentration":
            hot_prov = rng.choice(caregivers["provider_id"].unique(), 2, replace=False)
            sub = caregivers[caregivers["provider_id"].isin(hot_prov)]["caregiver_id"].to_numpy()
            if len(sub):
                pick = rng.choice(sub, len(ids))
                claims.loc[ids, "caregiver_id"] = pick
                cg_prov = caregivers.set_index("caregiver_id")["provider_id"]
                claims.loc[ids, "provider_id"] = cg_prov.loc[pick].to_numpy()

        elif scenario == "after_hours_activity":
            base = claims.loc[ids, "service_date"]
            hr = rng.uniform(0.5, 4.5, len(ids))
            st = base + _td_min(hr)
            claims.loc[ids, "billing_start_time"] = st
            claims.loc[ids, "billing_end_time"] = (
                st + _td_min(claims.loc[ids, "billed_hours"]))
            if len(has_evv):
                evv.loc[has_evv, "check_in_time"] = claims.loc[has_evv, "billing_start_time"].to_numpy()
                evv.loc[has_evv, "check_out_time"] = claims.loc[has_evv, "billing_end_time"].to_numpy()

        elif scenario == "weekend_activity":
            d = claims.loc[ids, "service_date"]
            # Push each date forward to the nearest Saturday.
            add = (5 - d.dt.dayofweek) % 7
            newd = d + pd.to_timedelta(add, unit="D")
            delta = newd - d
            for c in ["service_date", "billing_start_time", "billing_end_time", "submission_timestamp"]:
                claims.loc[ids, c] = claims.loc[ids, c] + delta
            if len(has_evv):
                dd = delta.loc[has_evv]
                evv.loc[has_evv, "check_in_time"] = evv.loc[has_evv, "check_in_time"] + dd
                evv.loc[has_evv, "check_out_time"] = evv.loc[has_evv, "check_out_time"] + dd

        elif scenario == "provider_billing_shift":
            # Abrupt level shift in the second half of the year for one agency.
            late = claims.loc[ids, "service_date"] > pd.Timestamp(self.cfg.start_date) + pd.Timedelta(days=200)
            sel = np.asarray(ids)[late.to_numpy()]
            if len(sel):
                claims.loc[sel, "billed_hours"] = claims.loc[sel, "billed_hours"] * 1.85
                claims.loc[sel, "billing_end_time"] = (
                    claims.loc[sel, "billing_start_time"]
                    + _td_min(claims.loc[sel, "billed_hours"]))

        elif scenario == "suspicious_timestamps":
            # Perfectly round clock times and instant submission: the signature
            # of batch-entered rather than observed visits.
            st = claims.loc[ids, "service_date"] + pd.Timedelta(hours=9)
            claims.loc[ids, "billing_start_time"] = st
            claims.loc[ids, "billed_hours"] = np.round(claims.loc[ids, "billed_hours"])
            claims.loc[ids, "billing_end_time"] = (
                st + _td_min(claims.loc[ids, "billed_hours"]))
            claims.loc[ids, "submission_timestamp"] = (
                claims.loc[ids, "billing_end_time"] + pd.Timedelta(minutes=1))
            if len(has_evv):
                evv.loc[has_evv, "check_in_time"] = claims.loc[has_evv, "billing_start_time"].to_numpy()
                evv.loc[has_evv, "check_out_time"] = claims.loc[has_evv, "billing_end_time"].to_numpy()
                evv.loc[has_evv, "device_type"] = "web_portal"
                evv.loc[has_evv, "verification_method"] = "manual_entry"

        elif scenario == "coordinated_activity":
            # A ring: a handful of caregivers billing a handful of patients with
            # near-identical shapes. Individually unremarkable rows.
            ring_cg = rng.choice(caregivers["caregiver_id"].to_numpy(), 6, replace=False)
            ring_pt = rng.choice(claims["patient_id"].to_numpy(), 10, replace=False)
            claims.loc[ids, "caregiver_id"] = rng.choice(ring_cg, len(ids))
            claims.loc[ids, "patient_id"] = rng.choice(ring_pt, len(ids))
            cg_prov = caregivers.set_index("caregiver_id")["provider_id"]
            claims.loc[ids, "provider_id"] = cg_prov.loc[claims.loc[ids, "caregiver_id"]].to_numpy()
            claims.loc[ids, "billed_hours"] = 6.0
            claims.loc[ids, "billing_end_time"] = (
                claims.loc[ids, "billing_start_time"] + pd.Timedelta(hours=6))

        elif scenario == "authorization_inconsistency":
            # Service billed under a code the authorisation does not cover, or
            # dated outside the authorisation window.
            wrong = rng.choice(SERVICE_CODES, len(ids))
            claims.loc[ids, "service_code"] = wrong
            claims.loc[ids, "authorized_hours"] = np.clip(
                claims.loc[ids, "authorized_hours"].to_numpy() * 0.35, 0.1, None)

    # -- public API ---------------------------------------------------------

    def generate(self) -> dict[str, pd.DataFrame]:
        regions = self._regions()
        providers = self._providers(regions)
        caregivers = self._caregivers(providers)
        patients = self._patients(regions)
        authorizations = self._authorizations(patients)
        claims = self._baseline_claims(providers, caregivers, patients, authorizations)
        evv = self._evv(claims, patients)
        claims, evv, labels = self._inject(claims, evv, patients, caregivers)

        claims = claims.sort_values("service_date").reset_index(drop=True)
        claims["service_date"] = claims["billing_start_time"].dt.normalize()
        return {
            "regions": regions,
            "providers": providers,
            "caregivers": caregivers,
            "patients": patients,
            "authorizations": authorizations,
            "claims": claims,
            "evv_events": evv,
            "scenario_labels": labels,
        }


def generate_and_save(cfg: GeneratorConfig | None = None, out_dir=None) -> dict[str, pd.DataFrame]:
    """Generate a dataset and persist every table as Parquet."""
    from ..config import SYNTHETIC_DIR, ensure_dirs

    ensure_dirs()
    out_dir = out_dir or SYNTHETIC_DIR
    tables = SyntheticMedicaidEVVGenerator(cfg).generate()
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    return tables
