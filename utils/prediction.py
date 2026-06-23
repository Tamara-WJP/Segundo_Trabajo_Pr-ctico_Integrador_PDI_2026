from pathlib import Path

import joblib
import numpy as np

from config import (
    CONFIANZA_MINIMA_PREDICCION,
    ETIQUETA_DESCONOCIDA,
    LARGO_SECUENCIA,
    RUTA_MODELO,
    SEÑA_DESCONOCIDA,
)
from utils.features import FEATURE_VERSION, build_feature_vector
from utils.landmarks import TOTAL_KEYPOINTS


class ModelNotFoundError(FileNotFoundError):
    """Error claro cuando el modelo entrenado todavia no existe."""


def load_model(model_path: Path = RUTA_MODELO) -> dict:
    if not Path(model_path).exists():
        raise ModelNotFoundError(
            "El modelo todavia no fue entrenado. Ejecutar primero "
            "scripts/collect_dataset.py y luego train_model.py."
        )

    return joblib.load(model_path)


def _validate_sequence(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    expected_shape = (LARGO_SECUENCIA, TOTAL_KEYPOINTS)

    if sequence.shape != expected_shape:
        raise ValueError(
            f"La secuencia debe tener forma {expected_shape}, "
            f"pero se recibio {sequence.shape}."
        )

    return sequence


def _prepare_features(artifact: dict, sequence: np.ndarray) -> np.ndarray:
    artifact_feature_version = artifact.get("feature_version")
    if artifact_feature_version != FEATURE_VERSION:
        raise ValueError(
            "El modelo fue entrenado con una version anterior de features. "
            "Ejecuta nuevamente train_model.py."
        )

    sequence = _validate_sequence(sequence)
    X = build_feature_vector(sequence).reshape(1, -1)
    expected_input_size = artifact.get("input_size")

    if expected_input_size is not None and X.shape[1] != expected_input_size:
        raise ValueError(
            "El modelo entrenado no coincide con el extractor de features actual. "
            "Ejecuta nuevamente train_model.py."
        )

    return X


def predict_sequence_with_scores(
    sequence: np.ndarray,
    model_path: Path = RUTA_MODELO,
) -> tuple[str, float, dict[str, float]]:
    artifact = load_model(model_path)
    model = artifact["model"]
    min_confidence = CONFIANZA_MINIMA_PREDICCION
    unknown_action = artifact.get("seña_desconocida", artifact.get("unknown_action", SEÑA_DESCONOCIDA))
    X = _prepare_features(artifact, sequence)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        best_index = int(np.argmax(probabilities))
        prediction = model.classes_[best_index]
        confidence = float(probabilities[best_index])
        scores = {
            str(class_name): float(probability)
            for class_name, probability in zip(model.classes_, probabilities)
        }

        if str(prediction) == unknown_action or confidence < min_confidence:
            return ETIQUETA_DESCONOCIDA, confidence, scores

        return str(prediction), confidence, scores

    prediction = model.predict(X)[0]
    confidence = 1.0
    scores = {str(prediction): confidence}
    if str(prediction) == unknown_action:
        return ETIQUETA_DESCONOCIDA, confidence, scores

    return str(prediction), confidence, scores


def predict_sequence(sequence: np.ndarray, model_path: Path = RUTA_MODELO) -> tuple[str, float]:
    label, confidence, _ = predict_sequence_with_scores(sequence, model_path)
    return label, confidence
