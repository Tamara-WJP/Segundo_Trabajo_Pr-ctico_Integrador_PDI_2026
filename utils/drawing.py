from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
mp_drawing = mp_solutions.drawing_utils
mp_holistic = mp_solutions.holistic

RUTAS_FUENTES = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
]


def draw_holistic_landmarks(frame, results) -> None:
    """Dibuja landmarks de pose y manos sobre el frame recibido."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(80, 255, 121), thickness=2, circle_radius=2),
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


def _tamano_fuente(scale: float) -> int:
    return max(12, int(round(32 * scale)))


@lru_cache(maxsize=32)
def _cargar_fuente(tamano: int) -> ImageFont.ImageFont:
    for ruta in RUTAS_FUENTES:
        if ruta.exists():
            return ImageFont.truetype(str(ruta), tamano)

    return ImageFont.load_default()


def medir_texto(texto: str, scale: float = 0.7) -> tuple[int, int]:
    fuente = _cargar_fuente(_tamano_fuente(scale))
    caja = fuente.getbbox(texto)
    ancho = caja[2] - caja[0]
    alto = caja[3] - caja[1]
    return ancho, alto


def draw_text(
    frame,
    text: str,
    position: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Dibuja texto Unicode sobre un frame BGR usando Pillow."""
    if frame is None:
        return

    fuente = _cargar_fuente(_tamano_fuente(scale))
    x, y = position
    caja = fuente.getbbox(text)
    alto_texto = caja[3] - caja[1]
    posicion_pillow = (x, max(0, y - alto_texto))
    color_rgb = (color[2], color[1], color[0])

    imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen_pillow = Image.fromarray(imagen_rgb)
    dibujante = ImageDraw.Draw(imagen_pillow)
    dibujante.text(posicion_pillow, text, font=fuente, fill=color_rgb)
    frame[:] = cv2.cvtColor(np.array(imagen_pillow), cv2.COLOR_RGB2BGR)


def draw_status_text(
    frame,
    action: str,
    sequence_number: int,
    frame_number: int,
    total_frames: int,
) -> None:
    """Agrega textos de guia durante la captura."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (25, 25, 25), -1)
    draw_text(frame, f"Seña: {action}", (16, 30), scale=0.8)
    draw_text(frame, f"Secuencia: {sequence_number}", (16, 60), scale=0.65)
    draw_text(frame, f"Frame: {frame_number}/{total_frames}", (280, 60), scale=0.65)
    draw_text(frame, "Presiona q para salir", (16, frame.shape[0] - 20), scale=0.6)


def draw_wait_text(frame, action: str, sequence_number: int) -> None:
    """Agrega instrucciones mientras se espera el inicio manual."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (25, 25, 25), -1)
    draw_text(frame, f"Seña: {action}", (16, 30), scale=0.8)
    draw_text(frame, f"Secuencia: {sequence_number}", (16, 60), scale=0.65)
    draw_text(frame, "ESPACIO: iniciar captura", (16, 92), scale=0.65)
    draw_text(frame, "Q: salir", (16, frame.shape[0] - 20), scale=0.6)


def draw_countdown_text(frame, action: str, sequence_number: int, remaining: int) -> None:
    """Muestra la cuenta regresiva antes de capturar la secuencia."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (25, 25, 25), -1)
    draw_text(frame, f"Seña: {action}", (16, 30), scale=0.8)
    draw_text(frame, f"Secuencia: {sequence_number}", (16, 60), scale=0.65)
    draw_text(frame, f"Captura en {remaining}...", (16, 92), scale=0.8)
    draw_text(frame, "Q: salir", (16, frame.shape[0] - 20), scale=0.6)
