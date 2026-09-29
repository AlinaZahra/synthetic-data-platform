"""Train-on-Synthetic, Test-on-Real (TSTR) utility evaluation.

Baseline = train on real_train, test on real_test.  TSTR = train on synthetic, test on real_test.
gap_pct is oriented so that POSITIVE always means "synthetic is worse than baseline":
  higher-is-better metrics (accuracy, f1_macro, auc, r2): 100 * (baseline - synth) / |baseline|
  lower-is-better (rmse):                                  100 * (synth - baseline) / |baseline|
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOWER_IS_BETTER = {"rmse"}
DISCRETE_MAX_UNIQUE = 10


def detect_task(y: pd.Series) -> str:
    """'classification' for non-numeric/bool/low-cardinality whole-number targets, else 'regression'."""
    y = y.dropna()
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    whole = bool(np.all(np.equal(np.mod(y.to_numpy(dtype=float), 1), 0)))
    return "classification" if whole and y.nunique() <= DISCRETE_MAX_UNIQUE else "regression"


def _numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            t = X[c].astype("datetime64[ns]")
            X[c] = t.astype("int64").astype(float) / 1e9
            X.loc[t.isna(), c] = np.nan
        elif pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(str)
        elif pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(float)
        else:
            X[c] = X[c].astype(object).where(X[c].notna(), "__missing__").astype(str)
    return X


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = [c for c in X.columns if pd.api.types.is_float_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def _models(task: str, seed: int) -> dict[str, Any]:
    if task == "classification":
        return {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
            "LogisticRegression": LogisticRegression(max_iter=1000),
        }
    return {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1),
        "Ridge": Ridge(alpha=1.0),
    }


def _auc(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, labels: np.ndarray) -> float | None:
    try:
        proba = model.predict_proba(X_test)
        classes = model.classes_
        if len(labels) == 2:
            pos = labels[-1]
            if pos not in classes:
                return None
            return float(roc_auc_score(y_test == pos, proba[:, list(classes).index(pos)]))
        aligned = np.zeros((len(X_test), len(labels)))
        for j, c in enumerate(classes):
            if c in labels:
                aligned[:, list(labels).index(c)] = proba[:, j]
        keep = y_test.isin(labels).to_numpy()
        return float(roc_auc_score(y_test[keep], aligned[keep], multi_class="ovr", labels=labels))
    except ValueError:
        return None


def _fit_score(name: str, est: Any, task: str, train: pd.DataFrame, test: pd.DataFrame,
               target: str, labels: np.ndarray | None) -> dict[str, Any]:
    X_tr, y_tr = _numeric_features(train.drop(columns=target)), train[target]
    X_te, y_te = _numeric_features(test.drop(columns=target)), test[target]
    if task == "classification" and y_tr.nunique() < 2:
        return {"error": "training data has a single class"}
    model = Pipeline([("prep", _preprocessor(X_tr)), ("est", est)])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    if task == "classification":
        return {
            "accuracy": float(accuracy_score(y_te, pred)),
            "f1_macro": float(f1_score(y_te, pred, average="macro")),
            "auc": _auc(model, X_te, y_te, labels),
        }
    return {"r2": float(r2_score(y_te, pred)), "rmse": float(np.sqrt(mean_squared_error(y_te, pred)))}


def _gap(metric: str, base: float | None, synth: float | None) -> float | None:
    if base is None or synth is None or abs(base) < 1e-12:
        return None
    delta = (synth - base) if metric in LOWER_IS_BETTER else (base - synth)
    return float(100.0 * delta / abs(base))


def evaluate_tstr(real_train: pd.DataFrame, real_test: pd.DataFrame, synthetic: pd.DataFrame,
                  target: str, seed: int = 0) -> dict[str, Any]:
    """Return a JSON-serialisable scorecard block (see module docstring for gap semantics)."""
    for name, df in (("real_train", real_train), ("real_test", real_test), ("synthetic", synthetic)):
        if target not in df.columns:
            raise ValueError(f"target {target!r} missing from {name}")
    features = [c for c in real_train.columns if c != target]
    missing = [c for c in features + [target] if c not in synthetic.columns or c not in real_test.columns]
    if missing:
        raise ValueError(f"columns missing from synthetic/test data: {missing}")
    cols = features + [target]
    rt, te, sy = (d[cols].dropna(subset=[target]).reset_index(drop=True) for d in (real_train, real_test, synthetic))

    task = detect_task(rt[target])
    labels = np.array(sorted(rt[target].unique())) if task == "classification" else None
    if task == "classification":
        te = te[te[target].isin(labels)].reset_index(drop=True)

    models: dict[str, Any] = {}
    for name, est in _models(task, seed).items():
        base = _fit_score(name, est, task, rt, te, target, labels)
        est2 = _models(task, seed)[name]
        syn = _fit_score(name, est2, task, sy, te, target, labels)
        gaps = ({m: _gap(m, base.get(m), syn.get(m)) for m in base} if "error" not in syn and "error" not in base else {})
        models[name] = {"baseline_real": base, "synthetic": syn, "gap_pct": gaps}

    primary = "auc" if task == "classification" else "r2"  # threshold-free where possible
    prim_gaps = [m["gap_pct"][primary] for m in models.values() if m["gap_pct"].get(primary) is not None]
    if not prim_gaps and task == "classification":
        primary = "f1_macro"
        prim_gaps = [m["gap_pct"][primary] for m in models.values() if m["gap_pct"].get(primary) is not None]
    return {
        "target": target,
        "task": task,
        "n_train_real": len(rt), "n_train_synthetic": len(sy), "n_test_real": len(te),
        "models": models,
        "summary": {
            "primary_metric": primary,
            "mean_gap_pct": float(np.mean(prim_gaps)) if prim_gaps else None,
            "max_gap_pct": float(np.max(prim_gaps)) if prim_gaps else None,
            "gap_semantics": "positive = synthetic worse than real baseline",
        },
    }


def tstr_json(*args: Any, **kwargs: Any) -> str:
    return json.dumps(evaluate_tstr(*args, **kwargs), indent=2)
