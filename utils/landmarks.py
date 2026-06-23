import numpy as np
import mediapipe as mp
import os
from pathlib import Path

from config import CONFIANZA_MINIMA_DETECCION, CONFIANZA_MINIMA_SEGUIMIENTO


def _require_legacy_solutions():
    if not hasattr(mp, "solutions"):
        version = getattr(mp, "__version__", "desconocida")
        raise ImportError(
            "Esta version de MediaPipe no incluye la API clasica mp.solutions "
            f"(version instalada: {version}). Para usar MediaPipe Holistic en "
            "este proyecto, recrea el entorno con Python 3.11 e instala las "
            "dependencias de requirements.txt."
        )

    return mp.solutions


mp_solutions = _require_legacy_solutions()
mp_holistic = mp_solutions.holistic


def _warn_windows_unicode_path() -> None:
    package_path = str(Path(mp.__file__).resolve())
    if os.name == "nt" and not package_path.isascii():
        raise RuntimeError(
            "MediaPipe esta instalado dentro de una ruta con caracteres no ASCII: "
            f"{package_path}. En Windows esto puede impedir que Holistic cargue "
            "sus archivos internos .binarypb. Crea el entorno virtual en una ruta "
            "sin acentos ni letras especiales, por ejemplo "
            "C:\\venvs\\tp-senias-mediapipe, o mueve "
            "el proyecto a una carpeta como C:\\tp-senias-mediapipe."
        )

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
POSE_VALUES = POSE_LANDMARKS * 4
HAND_VALUES = HAND_LANDMARKS * 3
TOTAL_KEYPOINTS = POSE_VALUES + HAND_VALUES + HAND_VALUES


def create_holistic():
    """Crea una instancia de MediaPipe Holistic con la configuracion del TP."""
    _warn_windows_unicode_path()
    return mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=CONFIANZA_MINIMA_DETECCION,
        min_tracking_confidence=CONFIANZA_MINIMA_SEGUIMIENTO,
    )


def _extract_pose(results) -> np.ndarray:
    if not results.pose_landmarks:
        return np.zeros(POSE_VALUES, dtype=np.float32)

    return np.array(
        [
            value
            for landmark in results.pose_landmarks.landmark
            for value in (landmark.x, landmark.y, landmark.z, landmark.visibility)
        ],
        dtype=np.float32,
    )


def _extract_hand(hand_landmarks) -> np.ndarray:
    if not hand_landmarks:
        return np.zeros(HAND_VALUES, dtype=np.float32)

    return np.array(
        [
            value
            for landmark in hand_landmarks.landmark
            for value in (landmark.x, landmark.y, landmark.z)
        ],
        dtype=np.float32,
    )


def extract_keypoints(results) -> np.ndarray:
    """Devuelve un vector fijo con pose, mano izquierda y mano derecha."""
    pose = _extract_pose(results)
    left_hand = _extract_hand(results.left_hand_landmarks)
    right_hand = _extract_hand(results.right_hand_landmarks)

    keypoints = np.concatenate([pose, left_hand, right_hand]).astype(np.float32)

    if keypoints.shape[0] != TOTAL_KEYPOINTS:
        raise ValueError(
            f"Vector de landmarks invalido: {keypoints.shape[0]} valores. "
            f"Se esperaban {TOTAL_KEYPOINTS}."
        )

    return keypoints
