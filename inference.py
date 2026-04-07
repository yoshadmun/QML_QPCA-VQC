import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import joblib

__all__ = [
    "predict",
    "predict_batch",
    "format_result",
    "format_batch",
    "validate_input",
    "preprocess",
    "FEATURES",
]
import numpy as np
import pandas as pd

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

MODEL_VERSION = os.environ.get("QSVC_MODEL_VERSION", "v1")


def _resolve_model_path() -> str:
    candidates = [
        f"qsvc_model_{MODEL_VERSION}.pkl",
        "qsvc_model_v1.pkl",
        "qsvc_model.pkl",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "No QSVC model file found. Tried: " + ", ".join(candidates)
    )


def _load_artifacts() -> Dict[str, Any]:
    if os.path.isfile("pipeline.pkl"):
        pipeline = joblib.load("pipeline.pkl")
        features = joblib.load("features.pkl")
        logging.info("Loaded inference artifacts: pipeline.pkl + features.pkl")
        return {
            "use_pipeline": True,
            "pipeline": pipeline,
            "model": None,
            "scaler": None,
            "features": features,
            "model_artifact": "pipeline.pkl",
        }

    model_path = _resolve_model_path()
    model = joblib.load(model_path)
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    logging.info(
        "Loaded inference artifacts: %s, scaler.pkl, features.pkl", model_path
    )
    return {
        "use_pipeline": False,
        "pipeline": None,
        "model": model,
        "scaler": scaler,
        "features": features,
        "model_artifact": model_path,
    }


_ART = _load_artifacts()
FEATURES: List[str] = [str(c) for c in list(_ART["features"])]


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid for binary confidence from decision values."""
    if x >= 0:
        z = np.exp(-np.clip(x, -500, 500))
        return float(1.0 / (1.0 + z))
    z = np.exp(np.clip(x, -500, 500))
    return float(z / (1.0 + z))


def _confidence_from_decision(decision: np.ndarray) -> float:
    d = np.asarray(decision, dtype=float).ravel()
    if d.size == 1:
        p = _sigmoid(float(d[0]))
        return round(max(p, 1.0 - p), 4)
    d = d - np.max(d)
    expd = np.exp(np.clip(d, -500, 500))
    probs = expd / expd.sum()
    return round(float(np.max(probs)), 4)


def _as_dataframe(
    X: Union[pd.DataFrame, np.ndarray], columns: Optional[List[str]] = None
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    cols = columns if columns is not None else FEATURES
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return pd.DataFrame(arr, columns=cols)


def _transform_for_classifier(X: pd.DataFrame) -> tuple:
    """Return (matrix for classifier, classifier instance)."""
    if _ART["use_pipeline"]:
        pipe = _ART["pipeline"]
        if len(pipe.steps) <= 1:
            return X, pipe.steps[-1][1]
        X_model = pipe[:-1].transform(X)
        return X_model, pipe.steps[-1][1]

    X_scaled = _ART["scaler"].transform(X)
    return X_scaled, _ART["model"]


def _predict_labels_and_confidence(X: pd.DataFrame) -> tuple:
    if _ART["use_pipeline"]:
        pipe = _ART["pipeline"]
        pred = pipe.predict(X)
    else:
        X_scaled = _ART["scaler"].transform(X)
        pred = _ART["model"].predict(X_scaled)

    X_model, clf = _transform_for_classifier(X)

    confidence: Optional[float] = None
    try:
        proba = clf.predict_proba(X_model)
        confidence = round(float(np.max(proba[0])), 4)
    except (ValueError, AttributeError):
        try:
            decision = clf.decision_function(X_model)
            confidence = _confidence_from_decision(decision)
        except (ValueError, AttributeError):
            confidence = None

    return pred, confidence


def validate_input(data: List[Any]) -> pd.DataFrame:
    if len(data) != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} features")

    clean: List[float] = []
    for x in data:
        if x is None:
            raise ValueError("Missing values not allowed")
        try:
            val = float(x)
        except (TypeError, ValueError):
            raise ValueError("All inputs must be numeric")
        if val < 0:
            raise ValueError("Negative values not allowed")
        clean.append(val)

    return pd.DataFrame([clean], columns=FEATURES)


def preprocess(data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    X = _as_dataframe(data)
    if _ART["use_pipeline"]:
        if len(_ART["pipeline"].steps) <= 1:
            return X.to_numpy(dtype=np.float64)
        return np.asarray(_ART["pipeline"][:-1].transform(X))
    return np.asarray(_ART["scaler"].transform(X))


def format_result(payload: Dict[str, Any]) -> str:
    """Pretty, human-readable rendering for notebooks and CLI."""
    if payload.get("status") == "error":
        return f"Status : error\nMessage: {payload.get('message', 'unknown')}"

    lines = [
        "Status     : success",
        f"Prediction : {payload['prediction']}",
    ]
    if "confidence" in payload:
        lines.append(f"Confidence : {payload['confidence']}")
    lines.extend(
        [
            f"Latency    : {payload['latency_sec']} s",
            f"Model      : {payload['model_artifact']}",
        ]
    )
    return "\n".join(lines)


def format_batch(payloads: List[Dict[str, Any]]) -> str:
    parts = []
    for i, p in enumerate(payloads, start=1):
        parts.append(f"── Sample {i} ──")
        parts.append(format_result(p))
    return "\n".join(parts)


def predict(data: List[Any]) -> Dict[str, Any]:
    try:
        logging.info("Input received: %s", data)
        t0 = time.perf_counter()

        X = validate_input(data)
        pred, confidence = _predict_labels_and_confidence(X)

        latency = time.perf_counter() - t0
        result = "Malware" if pred[0] == 1 else "Benign"

        payload: Dict[str, Any] = {
            "status": "success",
            "prediction": result,
            "latency_sec": round(latency, 4),
            "model_artifact": _ART["model_artifact"],
        }
        if confidence is not None:
            payload["confidence"] = confidence

        logging.info("Prediction: %s", result)
        if confidence is not None:
            logging.info("Confidence: %s", confidence)
        logging.info("Latency: %.4fs", latency)

        return payload

    except Exception as e:
        logging.error("%s", e)
        return {"status": "error", "message": str(e)}


def predict_batch(data_list: List[List[Any]]) -> List[Dict[str, Any]]:
    return [predict(row) for row in data_list]
