import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score
import models.xgboost.helpers as helpers

YAML_PATH = Path(r"models/xgboost/xgb_config.yaml")

SummaryWriter = None
try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter
    SummaryWriter = _TBWriter
except Exception:
    try:
        from tensorboardX import SummaryWriter as _TBWriter
        SummaryWriter = _TBWriter
    except Exception:
        SummaryWriter = None

def train_one_model(cfg: dict) -> None:
    train_csv = Path(cfg["data"]["train_csv"])
    npz_dir   = Path(cfg["data"]["npz_dir"])
    key_X     = cfg["data"].get("npz_key_X", "X")
    include_scalars = bool(cfg["data"].get("include_scalar_extras", True))
    scalar_keys = list(cfg["data"].get("scalar_keys", ["teff", "radius", "mass", "logg", "phot_g_mean_mag"]))

    ac_lags = list(cfg.get("features", {}).get("ac_lags", [1, 2, 4, 8, 16, 32]))
    fft_cfg = cfg.get("features", {}).get("fft", {"enabled": True, "k": 3, "exclude_dc": True})
    fft_enabled = bool(fft_cfg.get("enabled", True))
    fft_k = int(fft_cfg.get("k", 3))
    fft_exclude_dc = bool(fft_cfg.get("exclude_dc", True))

    tb_cfg = cfg.get("tensorboard", {}) or {}
    tb_enabled = bool(tb_cfg.get("enabled", False)) and (SummaryWriter is not None)
    tag_prefix = str(tb_cfg.get("tag_prefix", "xgb"))
    log_pr_curve = bool(tb_cfg.get("log_pr_curve", True))
    top_importances = int(tb_cfg.get("top_importances", 0) or 0)
    if tb_enabled:
        log_dir = Path(tb_cfg.get("log_dir", "tb")).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
    else:
        writer = None

    dbg_cfg = cfg.get("debug", {}) or {}
    dbg_preview_n = int(dbg_cfg.get("preview_npz_stats_n", 0) or 0)
    dbg_print_every = int(dbg_cfg.get("print_every", 0) or 0)

    out_root = Path(cfg["output"]["out_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    model_name = cfg["output"].get("model_name", "xgb_quick.json")
    run_dir = out_root  # single run

    df = helpers.load_train_table(train_csv)

    rows: List[Dict[str, float]] = []
    ys: List[int] = []
    kept_ids: List[int] = []
    missing_npz = 0
    bad_npz = 0

    dbg_shown = 0
    for i, r in df.iterrows():
        kepid = int(r["kepid"])
        y = int(r["label"])
        p = helpers.npz_path_for_kepid(npz_dir, kepid)
        feats, _, dbg = helpers.load_npz_features(
            p, key_X, include_scalars, scalar_keys,
            ac_lags, fft_enabled, fft_k, fft_exclude_dc,
            dbg_capture_stats=(dbg_shown < dbg_preview_n)
        )
        if feats is None:
            if not p.exists():
                missing_npz += 1
            else:
                bad_npz += 1
            continue

        rows.append(feats)
        ys.append(y)
        kept_ids.append(kepid)

        if dbg is not None and dbg_shown < dbg_preview_n:
            print(f"[debug:X] kepid={kepid} len={dbg['len']} std={dbg['std']:.6g} p2p={dbg['p2p']:.6g} "
                  f"min={dbg['min']:.6g} max={dbg['max']:.6g}")
            dbg_shown += 1

        if dbg_print_every and (i + 1) % dbg_print_every == 0:
            print(f"[progress] processed {i+1}/{len(df)} rows...")

    if len(rows) == 0:
        raise RuntimeError("No training rows could be constructed (all NPZs missing/bad?).")

    X = pd.DataFrame(rows)
    y = np.array(ys, dtype=np.int32)

    tr = cfg["training"]
    test_size = float(tr.get("val_ratio", 0.2))
    seed = int(tr.get("seed", 42))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_train, idx_val = next(splitter.split(X, y))

    X_train, X_val = X.iloc[idx_train].copy(), X.iloc[idx_val].copy()
    y_train, y_val = y[idx_train].copy(), y[idx_val].copy()

    perm = np.random.RandomState(seed).permutation(len(X_train))
    X_train = X_train.iloc[perm]
    y_train = y_train[perm]

    cols = sorted(list(set(X_train.columns) | set(X_val.columns)))
    X_train = X_train.reindex(columns=cols, fill_value=0.0)
    X_val   = X_val.reindex(columns=cols,   fill_value=0.0)

    pos_ct = max(int((y_train == 1).sum()), 1)
    neg_ct = max(int((y_train == 0).sum()), 1)
    spw = float(neg_ct) / float(pos_ct)

    em = tr.get("eval_metric", "aucpr")
    eval_metrics = [em] if isinstance(em, str) else list(em)
    if "auc" not in eval_metrics:
        eval_metrics.append("auc")

    params = {
        "objective": "binary:logistic",
        "eval_metric": eval_metrics,
        "learning_rate": tr.get("learning_rate", 0.05),
        "max_depth": tr.get("max_depth", 6),
        "min_child_weight": tr.get("min_child_weight", 1.0),
        "subsample": tr.get("subsample", 0.8),
        "colsample_bytree": tr.get("colsample_bytree", 0.8),
        "reg_alpha": tr.get("reg_alpha", 0.0),
        "reg_lambda": tr.get("reg_lambda", 1.0),
        "tree_method": tr.get("tree_method", "hist"),
        "scale_pos_weight": tr.get("scale_pos_weight", spw),
        "verbosity": 1,
        "seed": seed,
        "random_state": seed,
    }
    num_boost_round = int(tr.get("n_estimators", 1000))
    early_stopping_rounds = int(tr.get("early_stopping_rounds", 100))

    assert "label" not in X_train.columns, "Feature matrix must not contain any 'label' column"

    print(f"[info] built {len(cols)} features from X (time/ac/fft)"
          f"{' + scalar extras' if include_scalars else ''}; rows used: {len(X)}")

    dtrain = xgb.DMatrix(X_train.values, label=y_train, feature_names=cols)
    dval   = xgb.DMatrix(X_val.values,   label=y_val,   feature_names=cols)
    watch  = [(dtrain, "train"), (dval, "val")]

    callbacks: List[xgb.callback.TrainingCallback] = []
    if early_stopping_rounds and early_stopping_rounds > 0:
        primary_metric = eval_metrics[0]  # 'aucpr' by default
        callbacks.append(xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            save_best=True,
            maximize=True,
            data_name="val",
            metric_name=primary_metric,
        ))

    evals_result: Dict[str, Dict[str, List[float]]] = {}
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=watch,
        callbacks=callbacks,
        evals_result=evals_result,   # <— works in xgboost 3.0.4
        verbose_eval=False,
    )

    best_iter = int(getattr(bst, "best_iteration", num_boost_round - 1))
    y_val_prob = bst.predict(dval, iteration_range=(0, best_iter + 1))
    try:
        ap = float(average_precision_score(y_val, y_val_prob))
    except Exception:
        ap = float("nan")
    try:
        roc = float(roc_auc_score(y_val, y_val_prob))
    except Exception:
        roc = float("nan")

    if tb_enabled and writer is not None:
        for split_name, metrics in (evals_result or {}).items():
            for metric_name, history in (metrics or {}).items():
                for i, v in enumerate(history):
                    writer.add_scalar(f"{tag_prefix}/{split_name}/{metric_name}", float(v), i)

        writer.add_scalar(f"{tag_prefix}/val/auprc_final", ap if np.isfinite(ap) else 0.0, 0)
        writer.add_scalar(f"{tag_prefix}/val/roc_auc_final", roc if np.isfinite(roc) else 0.0, 0)
        writer.add_scalar(f"{tag_prefix}/meta/best_iteration", best_iter, 0)
        writer.add_scalar(f"{tag_prefix}/meta/rows_used", len(X), 0)

        try:
            writer.add_pr_curve(f"{tag_prefix}/val_pr_curve",
                                y_val.astype(np.int32),
                                y_val_prob.astype(np.float32),
                                global_step=0)
        except Exception:
            pass

        if top_importances > 0:
            gain = bst.get_score(importance_type="gain") or {}
            top = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:top_importances]
            for name, val in top:
                writer.add_scalar(f"{tag_prefix}/feature_importance_gain/{name}", float(val), 0)

        try:
            writer.add_hparams(
                hparam_dict={
                    "learning_rate": params["learning_rate"],
                    "max_depth": params["max_depth"],
                    "min_child_weight": params["min_child_weight"],
                    "subsample": params["subsample"],
                    "colsample_bytree": params["colsample_bytree"],
                    "reg_alpha": params["reg_alpha"],
                    "reg_lambda": params["reg_lambda"],
                    "scale_pos_weight": params["scale_pos_weight"],
                    "seed": params.get("seed", 0),
                },
                metric_dict={
                    "val/auprc": ap if np.isfinite(ap) else 0.0,
                    "val/roc_auc": roc if np.isfinite(roc) else 0.0,
                }
            )
        except Exception:
            pass

        writer.flush()
        writer.close()

    model_path = run_dir / model_name
    bst.save_model(str(model_path))

    (run_dir / "feature_names.json").write_text(json.dumps(cols, indent=2))
    manifest = {
        "yaml_path": str(YAML_PATH),
        "train_csv": str(train_csv),
        "npz_dir": str(npz_dir),
        "npz_key_X": key_X,
        "include_scalar_extras": include_scalars,
        "scalar_keys": scalar_keys,
        "ac_lags": ac_lags,
        "fft": {"enabled": fft_enabled, "k": fft_k, "exclude_dc": fft_exclude_dc},
        "rows_in": int(len(df)),
        "rows_used": int(len(X)),
        "missing_npz": int(missing_npz),
        "bad_npz": int(bad_npz),
        "val_metrics": {"auprc": ap, "roc_auc": roc},
        "best_iteration": best_iter,
        "params": params,
        "model_path": str(model_path),
        "seed": seed,
        "tensorboard": {
            "enabled": tb_enabled,
            "log_dir": str(tb_cfg.get("log_dir", "")) if tb_enabled else None,
            "tag_prefix": tag_prefix if tb_enabled else None,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] saved model -> {model_path}")
    print(f"[val] AUPRC={ap:.5f}  ROC-AUC={roc:.5f}  used_rows={len(X)} / input_rows={len(df)}")

def main():
    cfg = helpers.load_config(YAML_PATH)
    print(f"[info] using config: {YAML_PATH}")
    train_one_model(cfg)

if __name__ == "__main__":
    main()
