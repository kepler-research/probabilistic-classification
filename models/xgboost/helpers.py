import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import yaml

def _safe(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

def time_feats(x: np.ndarray) -> Dict[str, float]:
    x = _safe(x)
    feats = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q90": float(np.quantile(x, 0.90)),
        "iqr": float(np.subtract(*np.percentile(x, [75, 25]))),
        "rms": float(np.sqrt(np.mean(x**2))),
        "abs_mean": float(np.mean(np.abs(x))),
        "p2p": float(np.ptp(x)),
        "zero_cross": float(((x[:-1] * x[1:]) < 0).sum()),
    }
    return feats

def autocorr_feats(x: np.ndarray, lags: List[int]) -> Dict[str, float]:
    x = _safe(x)
    xc = x - np.mean(x)
    denom = float(np.dot(xc, xc)) + 1e-12
    out = {}
    for L in lags:
        if L <= 0 or L >= len(xc):
            out[f"ac{L}"] = 0.0
        else:
            num = float(np.dot(xc[:-L], xc[L:]))
            out[f"ac{L}"] = num / denom
    return out

def _fft_power(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.fft.rfft(_safe(x))
    P = (X.real**2 + X.imag**2)
    n = len(P)
    freqs = np.linspace(0.0, 0.5, n)  # normalized frequency (Nyquist=0.5)
    return freqs.astype(np.float32), P.astype(np.float32)

def tiny_fft_feats(x: np.ndarray, k: int = 3, exclude_dc: bool = True) -> Dict[str, float]:
    freqs, P = _fft_power(x)
    total = float(np.sum(P) + 1e-12)
    idx = np.argsort(P)[::-1]
    if exclude_dc and len(idx) and idx[0] == 0:
        idx = idx[1:]
    idx = idx[:k]
    out: Dict[str, float] = {}
    for r, ix in enumerate(idx):
        out[f"fft_peak{r}_pow"] = float(P[ix] / total)
        out[f"fft_peak{r}_freq"] = float(freqs[ix])
    for r in range(len(idx), k):  # pad if < k bins
        out[f"fft_peak{r}_pow"] = 0.0
        out[f"fft_peak{r}_freq"] = 0.0
    centroid = float(np.sum(freqs * P) / total)
    out["spec_centroid"] = centroid
    out["spec_bandwidth"] = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * P) / total))
    return out

def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_train_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"kepid", "label"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required column(s): {sorted(missing)}")
    return df

def npz_path_for_kepid(npz_dir: Path, kepid: int) -> Path:
    return npz_dir / f"{int(kepid)}.npz"

def load_npz_features(
    npz_path: Path,
    key_X: str,
    include_scalars: bool,
    scalar_keys: List[str],
    ac_lags: List[int],
    fft_enabled: bool,
    fft_k: int,
    fft_exclude_dc: bool,
    dbg_capture_stats: bool = False,
) -> Tuple[Optional[Dict[str, float]], Optional[int], Optional[Dict[str, float]]]:
    if not npz_path.exists():
        return None, None, None
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            if key_X not in z:
                return None, None, None

            x = z[key_X].squeeze().astype(np.float32, copy=False)  # ignore any 'label' key in NPZ
            feats: Dict[str, float] = {}
            feats.update(time_feats(x))
            feats.update(autocorr_feats(x, ac_lags))
            if fft_enabled:
                feats.update(tiny_fft_feats(x, fft_k, fft_exclude_dc))

            if include_scalars:
                for k in scalar_keys:
                    if k in z:
                        feats[f"ex_{k}"] = float(np.array(z[k]).astype(np.float32))

            kepid_in = int(z["kepid"]) if "kepid" in z else None

            dbg = None
            if dbg_capture_stats:
                dbg = {
                    "len": int(x.shape[-1]),
                    "std": float(np.std(x)),
                    "p2p": float(np.ptp(x)),
                    "min": float(np.min(x)),
                    "max": float(np.max(x)),
                }

            return feats, kepid_in, dbg
    except Exception:
        return None, None, None