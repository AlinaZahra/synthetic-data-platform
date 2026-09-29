"""A small fictional healthcare dataset: patients, encounters, claims. No real people, providers or records."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdp.locale import get_locale

ANCHOR = pd.Timestamp("2025-12-31")
# ICD-10 codes are a public classification; the assignment to patients here is random
DIAGNOSES = {"J06.9": 0.20, "I10": 0.16, "E11.9": 0.12, "M54.5": 0.12, "K21.9": 0.08, "F41.9": 0.07, "N39.0": 0.06, "S93.4": 0.05, "J45.909": 0.07, "R51": 0.07}
ENCOUNTER_TYPES = {"outpatient": 0.72, "emergency": 0.18, "inpatient": 0.10}
CLAIM_STATUS = {"approved": 0.72, "partially_approved": 0.14, "denied": 0.10, "pending": 0.04}


def make_healthcare(patients: int = 300, seed: int = 0, locale: str = "en-US") -> dict[str, pd.DataFrame]:
    if patients < 1 or patients > 200_000:
        raise ValueError("patients must be between 1 and 200,000")
    rng = np.random.default_rng(seed)
    pack = get_locale(locale)
    n = patients
    pat = pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "name": [pack.person_name(rng) for _ in range(n)],
        "sex": rng.choice(["F", "M"], n, p=[0.51, 0.49]),
        "date_of_birth": pd.Timestamp("2025-12-31") - pd.to_timedelta(rng.integers(0, 90 * 365, n), unit="D"),
        "blood_type": rng.choice(["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"], n, p=[0.38, 0.34, 0.09, 0.03, 0.07, 0.06, 0.02, 0.01]),
        "phone": [pack.phone(rng) for _ in range(n)],
    })
    per = np.clip(rng.poisson(2.2, n), 0, 12)
    pid = np.repeat(pat["patient_id"].to_numpy(), per)
    m = len(pid)
    dob = pat.set_index("patient_id")["date_of_birth"].reindex(pid).to_numpy()
    days = rng.integers(0, 3 * 365, m)
    enc_date = pd.to_datetime(ANCHOR - pd.to_timedelta(days, unit="D"))
    enc_date = pd.DatetimeIndex(np.maximum(enc_date.to_numpy(), dob + np.timedelta64(1, "D")))
    typ = rng.choice(list(ENCOUNTER_TYPES), m, p=list(ENCOUNTER_TYPES.values()))
    los = np.where(typ == "inpatient", rng.integers(1, 15, m), 0)
    base = {"outpatient": 120.0, "emergency": 650.0, "inpatient": 1800.0}
    charge = np.round(np.array([base[t] for t in typ]) * np.exp(0.35 * rng.standard_normal(m)) + los * 420, 2)
    enc = pd.DataFrame({
        "encounter_id": np.arange(1, m + 1), "patient_id": pid, "encounter_date": enc_date, "encounter_type": typ,
        "diagnosis_code": rng.choice(list(DIAGNOSES), m, p=list(DIAGNOSES.values())), "length_of_stay_days": los.astype(int), "charge": charge,
    })
    has_claim = rng.random(m) < 0.93
    e = enc[has_claim].reset_index(drop=True)
    k = len(e)
    status = rng.choice(list(CLAIM_STATUS), k, p=list(CLAIM_STATUS.values()))
    frac = np.select([status == "approved", status == "partially_approved", status == "denied"], [1.0, rng.uniform(0.4, 0.9, k), 0.0], default=0.0)
    approved = np.where(status == "pending", np.nan, np.round(e["charge"].to_numpy() * frac, 2))
    cl = pd.DataFrame({
        "claim_id": np.arange(1, k + 1), "encounter_id": e["encounter_id"].to_numpy(),
        "submitted_date": e["encounter_date"] + pd.to_timedelta(rng.integers(1, 30, k), unit="D"),
        "status": status, "billed_amount": e["charge"].to_numpy(), "approved_amount": approved,
    })
    return {"patients": pat, "encounters": enc, "claims": cl}
