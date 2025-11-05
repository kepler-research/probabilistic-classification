import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import xgboost as xgb
import yaml

YAML_PATH = Path(r"models/xgboost/xgb_config.yaml")

def run_predict(job_name: Optional[str] = None) -> Path:
    cfg = _load_yaml(YAML_PATH)
    base = YAML_PATH.parent

    model_path = _resolve(cfg["model"]["checkpoint"], base)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    feat_json = model_path.parent / "feature_names.json"
    feature_names = json.loads(feat_json.read_text())

    dd = cfg.get("data_defaults", {}) or {}
    key_X = dd.get("npz_key_X", "X")
    include_scalars = bool(dd.get("include_scalar_extras", True))
    scalar_keys = list(dd.get("scalar_keys", ["teff", "radius", "mass", "logg", "phot_g_mean_mag"]))
    fcfg = cfg.get("features", {}) or {}
    ac_lags = list(fcfg.get("ac_lags", [1, 2, 4, 8, 16, 32]))
    fft_cfg = fcfg.get("fft", {"enabled": True, "k": 3, "exclude_dc": True})
    fft_on = bool(fft_cfg.get("enabled", True))
    fft_k = int(fft_cfg.get("k", 3))
    fft_ex_dc = bool(fft_cfg.get("exclude_dc", True))

    job = _select_predict_job(cfg, job_name)
    eval_csv = _resolve(job["eval_csv"], base)
    npz_dir  = _resolve(job["npz_dir"], base)
    out_csv  = _resolve(job["output"]["preds_csv"], base)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(eval_csv)
    if "kepid" not in df.columns:
        raise ValueError(f"{eval_csv} must contain a 'kepid' column")

    rows: List[Dict[str, float]] = []
    keep_idx: List[int] = []
    miss = bad = 0
    for i, r in df.iterrows():
        feats = _build_features_for(
            _npz_path(npz_dir, int(r["kepid"])),
            key_X, include_scalars, scalar_keys,
            ac_lags, fft_on, fft_k, fft_ex_dc
        )
        if feats is None:
            p = _npz_path(npz_dir, int(r["kepid"]))
            miss += int(not p.exists()); bad += int(p.exists())
            continue
        rows.append(feats)
        keep_idx.append(i)

    if not rows:
        raise RuntimeError("No rows to score (NPZs missing/bad?)")

    X = pd.DataFrame(rows).reindex(columns=feature_names, fill_value=0.0)
    booster = xgb.Booster(); booster.load_model(str(model_path))
    prob = booster.predict(xgb.DMatrix(X.values, feature_names=feature_names))

    df_out = df.iloc[keep_idx].copy()
    df_out["proba"] = prob[: len(df_out)]
    df_out.to_csv(out_csv, index=False)

    print(f"[predict:{job.get('name','job')}] wrote -> {out_csv}  (kept={len(df_out)}/{len(df)} miss={miss} bad={bad})")
    return out_csv

if __name__ == "__main__":
    run_predict(None)
