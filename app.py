from collections import deque
from pathlib import Path
import time

import cv2
import gradio as gr
import numpy as np

from config import (
    CONFIANZA_MINIMA_PREDICCION,
    ETIQUETA_DESCONOCIDA,
    LARGO_SECUENCIA,
    RUTA_MODELO,
    SEÑA_DESCONOCIDA,
    SEÑAS,
)
from utils.drawing import draw_text as dibujar_texto_pillow
from utils.drawing import medir_texto
from utils.landmarks import HAND_VALUES, POSE_VALUES
from utils.landmarks import create_holistic as crear_detector_holistic
from utils.landmarks import extract_keypoints as extraer_puntos_clave
from utils.prediction import ModelNotFoundError, predict_sequence_with_scores


TITULO_PROYECTO = "Seña a Texto (LSA)"
MENSAJE_MODELO_NO_LISTO = (
    "El modelo todavia no fue entrenado. Ejecutar primero "
    "scripts/collect_dataset.py y luego train_model.py."
)

SALTOS_ENTRE_PREDICCIONES = 3
INTERVALO_STREAM_SEGUNDOS = 0.08
FRAMES_MOVIMIENTO = LARGO_SECUENCIA
UMBRAL_MOVIMIENTO = 0.006
MIN_PUNTOS_MANO = 6
PAUSA_LECTURA_SEGUNDOS = 3.0

ETIQUETA_NO_RECONOCIDA = "NO RECONOCIDA"
ESTADO_ESPERANDO = "ESPERANDO SEÑA"
ESTADO_LEYENDO = "LEYENDO SEÑA"
ESTADO_LEIDA = "SEÑA LEIDA"

detector_holistic_compartido = None


def obtener_detector_holistic():
    global detector_holistic_compartido

    if detector_holistic_compartido is None:
        detector_holistic_compartido = crear_detector_holistic()

    return detector_holistic_compartido


def crear_estado_deteccion() -> dict:
    return {
        "secuencia": deque(maxlen=LARGO_SECUENCIA),
        "frames_vistos": 0,
        "pausar_lectura_hasta": 0.0,
        "frame_con_resultado": None,
        "estado_con_resultado": ESTADO_ESPERANDO,
    }


def obtener_secuencia(estado_deteccion: dict) -> deque:
    secuencia = estado_deteccion.get("secuencia")

    if not isinstance(secuencia, deque):
        secuencia = deque(secuencia or [], maxlen=LARGO_SECUENCIA)
        estado_deteccion["secuencia"] = secuencia

    return secuencia


def reiniciar_lectura_estado(estado_deteccion: dict) -> None:
    secuencia = obtener_secuencia(estado_deteccion)
    secuencia.clear()
    estado_deteccion["frames_vistos"] = 0
    estado_deteccion["pausar_lectura_hasta"] = 0.0
    estado_deteccion["frame_con_resultado"] = None
    estado_deteccion["estado_con_resultado"] = ESTADO_ESPERANDO


def procesar_frame_webcam(frame_rgb: np.ndarray, detector_holistic) -> tuple[np.ndarray, np.ndarray]:
    """Espeja un frame RGB del navegador, ejecuta MediaPipe y devuelve landmarks."""
    imagen_rgb = cv2.flip(frame_rgb, 1)
    imagen_rgb = np.ascontiguousarray(imagen_rgb)
    imagen_rgb.flags.writeable = False
    resultados = detector_holistic.process(imagen_rgb)
    puntos_clave = extraer_puntos_clave(resultados)

    return imagen_rgb.copy(), puntos_clave


def nombre_para_mostrar(etiqueta: str) -> str:
    if etiqueta == ETIQUETA_DESCONOCIDA:
        return ETIQUETA_NO_RECONOCIDA

    return etiqueta


def armar_lineas_confianza(confianzas: dict[str, float]) -> list[str]:
    filas = [(seña, confianzas.get(seña, 0.0)) for seña in SEÑAS]
    filas.append((ETIQUETA_NO_RECONOCIDA, confianzas.get(SEÑA_DESCONOCIDA, 0.0)))

    return [f"{nombre}: {confianza:.2%}" for nombre, confianza in filas]


def armar_estado(mensaje: str, confianzas: dict[str, float] | None = None) -> str:
    if not confianzas:
        return mensaje

    return "\n".join(
        [
            mensaje,
            f"Umbral: {CONFIANZA_MINIMA_PREDICCION:.0%}",
            *armar_lineas_confianza(confianzas),
        ]
    )


def calcular_movimiento_manos(secuencia: deque) -> float:
    if len(secuencia) < 2:
        return 0.0

    recientes = np.array(list(secuencia)[-FRAMES_MOVIMIENTO:], dtype=np.float32)
    inicio_mano_izquierda = POSE_VALUES
    inicio_mano_derecha = POSE_VALUES + HAND_VALUES
    manos = np.concatenate(
        [
            recientes[:, inicio_mano_izquierda:inicio_mano_derecha],
            recientes[:, inicio_mano_derecha : inicio_mano_derecha + HAND_VALUES],
        ],
        axis=1,
    ).reshape(len(recientes), 42, 3)

    puntos_visibles = np.any(np.abs(manos) > 1e-6, axis=2)
    movimientos_por_frame = []

    for indice in range(1, len(recientes)):
        visibles = puntos_visibles[indice] & puntos_visibles[indice - 1]
        if np.count_nonzero(visibles) < MIN_PUNTOS_MANO:
            continue

        movimientos = np.linalg.norm(
            manos[indice, visibles, :2] - manos[indice - 1, visibles, :2],
            axis=1,
        )
        movimientos_por_frame.append(float(np.mean(movimientos)))

    return max(movimientos_por_frame, default=0.0)


def hay_movimiento_de_seña(secuencia: deque) -> bool:
    return calcular_movimiento_manos(secuencia) >= UMBRAL_MOVIMIENTO


def dibujar_resultado_centrado(imagen_rgb: np.ndarray, etiqueta: str) -> np.ndarray:
    imagen_bgr = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)
    alto, ancho = imagen_bgr.shape[:2]
    texto = nombre_para_mostrar(etiqueta)
    escala = 1.35
    margen_x = 24
    margen_y = 18
    ancho_texto, alto_texto = medir_texto(texto, scale=escala)
    linea_base = max(6, int(alto_texto * 0.2))
    x = max(16, (ancho - ancho_texto) // 2)
    y = max(alto_texto + 16, alto // 2)

    inicio_caja = (x - margen_x, y - alto_texto - margen_y)
    fin_caja = (x + ancho_texto + margen_x, y + linea_base + margen_y)
    capa = imagen_bgr.copy()
    cv2.rectangle(capa, inicio_caja, fin_caja, (20, 20, 20), -1)
    imagen_bgr = cv2.addWeighted(capa, 0.58, imagen_bgr, 0.42, 0)
    dibujar_texto_pillow(imagen_bgr, texto, (x, y), scale=escala)

    return cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)


def analizar_frame_webcam(frame_rgb: np.ndarray | None, estado_deteccion: dict | None):
    if estado_deteccion is None:
        estado_deteccion = crear_estado_deteccion()

    if frame_rgb is None:
        return None, ESTADO_ESPERANDO, estado_deteccion

    if not Path(RUTA_MODELO).exists():
        return frame_rgb, MENSAJE_MODELO_NO_LISTO, estado_deteccion

    try:
        ahora = time.time()

        if estado_deteccion["pausar_lectura_hasta"] > 0:
            if ahora < estado_deteccion["pausar_lectura_hasta"]:
                return (
                    estado_deteccion["frame_con_resultado"],
                    estado_deteccion["estado_con_resultado"],
                    estado_deteccion,
                )

            reiniciar_lectura_estado(estado_deteccion)

        detector_holistic = obtener_detector_holistic()
        imagen_camara, puntos_clave = procesar_frame_webcam(frame_rgb, detector_holistic)
        secuencia = obtener_secuencia(estado_deteccion)
        secuencia.append(puntos_clave)
        estado_deteccion["frames_vistos"] += 1

        hay_movimiento = hay_movimiento_de_seña(secuencia)
        frame_salida = imagen_camara

        if not hay_movimiento:
            estado = ESTADO_ESPERANDO
        elif len(secuencia) < LARGO_SECUENCIA:
            estado = ESTADO_LEYENDO
        else:
            estado = ESTADO_LEYENDO

            if estado_deteccion["frames_vistos"] % SALTOS_ENTRE_PREDICCIONES == 0:
                secuencia_array = np.array(secuencia, dtype=np.float32)
                etiqueta, _, confianzas = predict_sequence_with_scores(secuencia_array)
                nombre = nombre_para_mostrar(etiqueta)
                estado = armar_estado(f"{ESTADO_LEIDA}: {nombre}", confianzas)
                frame_salida = dibujar_resultado_centrado(imagen_camara, etiqueta)
                estado_deteccion["frame_con_resultado"] = frame_salida
                estado_deteccion["estado_con_resultado"] = estado
                estado_deteccion["pausar_lectura_hasta"] = (
                    time.time() + PAUSA_LECTURA_SEGUNDOS
                )
                estado_deteccion["frames_vistos"] = 0
                secuencia.clear()

        return frame_salida, estado, estado_deteccion
    except ModelNotFoundError:
        return frame_rgb, MENSAJE_MODELO_NO_LISTO, estado_deteccion
    except ValueError as error:
        return frame_rgb, f"No se pudo realizar la prediccion: {error}", estado_deteccion
    except Exception as error:
        return frame_rgb, f"Ocurrio un error al procesar la camara: {error}", estado_deteccion


def reiniciar_lectura():
    return None, ESTADO_ESPERANDO, crear_estado_deteccion()


def crear_interfaz() -> gr.Blocks:
    with gr.Blocks(title=TITULO_PROYECTO) as interfaz:
        gr.Markdown(
            f"""
            # {TITULO_PROYECTO}

            Prototipo educativo de reconocimiento basico de señas usando
            MediaPipe Holistic, landmarks de pose y manos, y un modelo simple
            entrenado con scikit-learn.

            Esta version reconoce HOLA, GRACIAS, SI, NO y AYUDA. Si la
            confianza del modelo es baja, la app devuelve NO RECONOCIDA.
            """
        )

        estado_deteccion = gr.State(crear_estado_deteccion())
        estado_camara = gr.Textbox(
            label="Estado",
            lines=8,
            value=ESTADO_ESPERANDO,
        )

        with gr.Row():
            entrada_camara = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="numpy",
                image_mode="RGB",
                label="Camara",
            )
            salida_camara = gr.Image(
                type="numpy",
                image_mode="RGB",
                label="Prediccion en vivo",
                interactive=False,
            )

        boton_reiniciar = gr.Button("Reiniciar lectura")

        entrada_camara.stream(
            fn=analizar_frame_webcam,
            inputs=[entrada_camara, estado_deteccion],
            outputs=[salida_camara, estado_camara, estado_deteccion],
            concurrency_limit=1,
            trigger_mode="always_last",
            stream_every=INTERVALO_STREAM_SEGUNDOS,
        )
        boton_reiniciar.click(
            fn=reiniciar_lectura,
            inputs=None,
            outputs=[salida_camara, estado_camara, estado_deteccion],
        )

        if not Path(RUTA_MODELO).exists():
            gr.Markdown(f"**Estado del modelo:** {MENSAJE_MODELO_NO_LISTO}")

    return interfaz


if __name__ == "__main__":
    app = crear_interfaz()
    app.queue(default_concurrency_limit=1)
    app.launch(server_name="0.0.0.0", server_port=7860)
