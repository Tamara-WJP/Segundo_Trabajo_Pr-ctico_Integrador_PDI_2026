from collections import deque
import html
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
from utils.landmarks import HAND_LANDMARKS, HAND_VALUES, POSE_LANDMARKS, POSE_VALUES
from utils.landmarks import create_holistic as crear_detector_holistic
from utils.landmarks import extract_keypoints as extraer_puntos_clave
from utils.prediction import ModelNotFoundError, predict_sequence_with_scores

try:
    from fastrtc import VideoStreamHandler, WebRTC
except ImportError:
    VideoStreamHandler = None
    WebRTC = None


TITULO_PROYECTO = "Seña a Texto (LSA)"
MENSAJE_MODELO_NO_LISTO = (
    "El modelo todavia no fue entrenado. Ejecutar primero "
    "scripts/collect_dataset.py y luego train_model.py."
)

SALTOS_ENTRE_PREDICCIONES = 3
FRAMES_MOVIMIENTO = LARGO_SECUENCIA
UMBRAL_MOVIMIENTO = 0.015
MIN_PUNTOS_MANO = 8
MIN_FRAMES_CON_MANOS = 4
PAUSA_LECTURA_SEGUNDOS = 3.0
FPS_SALIDA = 24
DEMORA_INICIO_SEGUNDOS = 1.0
FRAMES_PREVIOS_POSE_INICIAL = 8
FRAMES_MOVIMIENTO_PARA_INICIAR = 4
ANCHO_CAMARA = 560
ALTO_CAMARA = 420
INTERVALO_REENVIO_RESULTADO_SEGUNDOS = 1.0
INTERVALO_ACTUALIZACION_ESTADO_SEGUNDOS = 0.25

ETIQUETA_NO_RECONOCIDA = "NO RECONOCIDA"
ESTADO_ESPERANDO = "ESPERANDO SEÑA"
ESTADO_LEYENDO = "LEYENDO SEÑA"
ESTADO_LEIDA = "SEÑA LEIDA"

POSE_HOMBRO_IZQUIERDO = 11
POSE_HOMBRO_DERECHO = 12

CSS_INTERFAZ = """
#camara-fastrtc {
    max-width: 620px;
    margin: 0 auto;
}
#camara-fastrtc video {
    max-height: 420px !important;
    object-fit: contain !important;
}
.estado-panel {
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    background: var(--block-background-fill);
    padding: 12px 14px;
    margin-bottom: 12px;
}
.estado-label {
    color: var(--body-text-color-subdued);
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 8px;
}
.estado-texto {
    color: var(--body-text-color);
    font-family: var(--font-mono);
    font-size: 15px;
    line-height: 1.45;
    margin: 0;
    min-height: 150px;
    white-space: pre-wrap;
}
"""


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


def limpiar_texto_estado(estado_nuevo) -> str:
    texto = str(estado_nuevo or ESTADO_ESPERANDO)

    if texto.startswith("replace,value,"):
        texto = texto.split("replace,value,", 1)[1]

    if texto.startswith("replace,,"):
        texto = texto.split("replace,,", 1)[1]

    return texto


def armar_html_estado(estado_nuevo) -> str:
    texto = html.escape(limpiar_texto_estado(estado_nuevo))
    return (
        '<div class="estado-panel">'
        '<div class="estado-label">Estado</div>'
        f'<pre class="estado-texto">{texto}</pre>'
        "</div>"
    )


class DetectorSeñasWebRTC:
    """Estado de lectura para un stream WebRTC de video."""

    def __init__(self) -> None:
        self.secuencia = deque(maxlen=LARGO_SECUENCIA)
        self.frames_previos = deque(maxlen=FRAMES_PREVIOS_POSE_INICIAL)
        self.frames_vistos = 0
        self.leyendo = False
        self.frames_movimiento_consecutivos = 0
        self.esperar_lectura_hasta = time.time() + DEMORA_INICIO_SEGUNDOS
        self.pausar_lectura_hasta = 0.0
        self.frame_con_resultado_bgr = None
        self.estado_con_resultado = ESTADO_ESPERANDO
        self.estado_actual = ESTADO_ESPERANDO
        self.estado_mostrado = None
        self.ultimo_envio_estado = 0.0
        self.detector_holistic = None

    def obtener_detector_holistic(self):
        if self.detector_holistic is None:
            self.detector_holistic = crear_detector_holistic()

        return self.detector_holistic

    def reiniciar_lectura(self) -> None:
        self.secuencia.clear()
        self.frames_previos.clear()
        self.frames_vistos = 0
        self.leyendo = False
        self.frames_movimiento_consecutivos = 0
        self.esperar_lectura_hasta = time.time() + DEMORA_INICIO_SEGUNDOS
        self.pausar_lectura_hasta = 0.0
        self.frame_con_resultado_bgr = None
        self.estado_con_resultado = ESTADO_ESPERANDO
        self.estado_actual = ESTADO_ESPERANDO
        self.estado_mostrado = None
        self.ultimo_envio_estado = 0.0

    def separar_puntos_clave(self, secuencia: np.ndarray):
        pose = secuencia[:, :POSE_VALUES].reshape(len(secuencia), POSE_LANDMARKS, 4)
        inicio_mano_derecha = POSE_VALUES + HAND_VALUES
        mano_izquierda = secuencia[:, POSE_VALUES:inicio_mano_derecha].reshape(
            len(secuencia),
            HAND_LANDMARKS,
            3,
        )
        mano_derecha = secuencia[:, inicio_mano_derecha:].reshape(
            len(secuencia),
            HAND_LANDMARKS,
            3,
        )

        return pose, mano_izquierda, mano_derecha

    def normalizar_manos_respecto_hombros(
        self,
        pose: np.ndarray,
        manos: np.ndarray,
        puntos_visibles: np.ndarray,
    ) -> np.ndarray:
        manos_normalizadas = np.zeros_like(manos, dtype=np.float32)

        for indice in range(len(manos)):
            hombro_izquierdo = pose[indice, POSE_HOMBRO_IZQUIERDO]
            hombro_derecho = pose[indice, POSE_HOMBRO_DERECHO]

            if hombro_izquierdo[3] <= 0.2 or hombro_derecho[3] <= 0.2:
                continue

            centro_hombros = (hombro_izquierdo[:3] + hombro_derecho[:3]) / 2.0
            ancho_hombros = float(
                np.linalg.norm(hombro_izquierdo[:2] - hombro_derecho[:2])
            )

            if ancho_hombros <= 1e-5:
                continue

            visibles = puntos_visibles[indice]
            manos_normalizadas[indice, visibles] = (
                manos[indice, visibles] - centro_hombros
            ) / ancho_hombros

        return manos_normalizadas

    def calcular_movimiento_manos(self, secuencia: deque) -> float:
        if len(secuencia) < 2:
            return 0.0

        recientes = np.array(list(secuencia)[-FRAMES_MOVIMIENTO:], dtype=np.float32)
        pose, mano_izquierda, mano_derecha = self.separar_puntos_clave(recientes)
        manos = np.concatenate([mano_izquierda, mano_derecha], axis=1)

        puntos_visibles = np.any(np.abs(manos) > 1e-6, axis=2)
        frames_con_manos = np.count_nonzero(
            np.count_nonzero(puntos_visibles, axis=1) >= MIN_PUNTOS_MANO
        )

        if frames_con_manos < MIN_FRAMES_CON_MANOS:
            return 0.0

        manos_normalizadas = self.normalizar_manos_respecto_hombros(
            pose,
            manos,
            puntos_visibles,
        )
        movimientos_por_frame = []

        for indice in range(1, len(recientes)):
            visibles = puntos_visibles[indice] & puntos_visibles[indice - 1]
            if np.count_nonzero(visibles) < MIN_PUNTOS_MANO:
                continue

            movimientos = np.linalg.norm(
                manos_normalizadas[indice, visibles, :2]
                - manos_normalizadas[indice - 1, visibles, :2],
                axis=1,
            )
            movimientos_por_frame.append(float(np.percentile(movimientos, 75)))

        return max(movimientos_por_frame, default=0.0)

    def hay_movimiento_de_seña(self, secuencia: deque) -> bool:
        return self.calcular_movimiento_manos(secuencia) >= UMBRAL_MOVIMIENTO

    def preparar_inicio_lectura(self) -> None:
        self.leyendo = True
        self.secuencia.clear()
        self.secuencia.extend(self.frames_previos)
        self.frames_vistos = len(self.secuencia)

    def responder(self, frame_bgr: np.ndarray, estado: str | None = None, forzar: bool = False):
        if estado is None:
            return frame_bgr

        ahora = time.time()
        reenviar_resultado = (
            estado.startswith(ESTADO_LEIDA)
            and ahora - self.ultimo_envio_estado >= INTERVALO_REENVIO_RESULTADO_SEGUNDOS
        )

        if forzar or estado != self.estado_mostrado or reenviar_resultado:
            self.estado_actual = estado
            self.estado_mostrado = estado
            self.ultimo_envio_estado = ahora

        return frame_bgr

    def procesar_frame(self, frame_bgr: np.ndarray):
        frame_bgr = cv2.flip(frame_bgr, 1)

        if not Path(RUTA_MODELO).exists():
            return self.responder(frame_bgr, MENSAJE_MODELO_NO_LISTO)

        try:
            ahora = time.time()

            if self.pausar_lectura_hasta > 0:
                if ahora < self.pausar_lectura_hasta:
                    return self.responder(
                        self.frame_con_resultado_bgr,
                        self.estado_con_resultado,
                    )

                self.reiniciar_lectura()

            if ahora < self.esperar_lectura_hasta:
                return self.responder(frame_bgr, ESTADO_ESPERANDO)

            detector_holistic = self.obtener_detector_holistic()
            imagen_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            imagen_rgb.flags.writeable = False
            resultados = detector_holistic.process(imagen_rgb)
            puntos_clave = extraer_puntos_clave(resultados)

            frame_salida_bgr = frame_bgr

            if not self.leyendo:
                self.frames_previos.append(puntos_clave)
                hay_movimiento = self.hay_movimiento_de_seña(self.frames_previos)

                if hay_movimiento:
                    self.frames_movimiento_consecutivos += 1
                else:
                    self.frames_movimiento_consecutivos = 0

                if self.frames_movimiento_consecutivos < FRAMES_MOVIMIENTO_PARA_INICIAR:
                    return self.responder(frame_salida_bgr, ESTADO_ESPERANDO)

                self.preparar_inicio_lectura()

            self.secuencia.append(puntos_clave)
            self.frames_vistos += 1
            estado = ESTADO_LEYENDO

            if len(self.secuencia) >= LARGO_SECUENCIA and (
                self.frames_vistos % SALTOS_ENTRE_PREDICCIONES == 0
            ):
                secuencia_array = np.array(self.secuencia, dtype=np.float32)
                etiqueta, _, confianzas = predict_sequence_with_scores(secuencia_array)
                nombre = nombre_para_mostrar(etiqueta)
                estado = armar_estado(f"{ESTADO_LEIDA}: {nombre}", confianzas)
                frame_rgb_resultado = dibujar_resultado_centrado(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                    etiqueta,
                )
                frame_salida_bgr = cv2.cvtColor(frame_rgb_resultado, cv2.COLOR_RGB2BGR)
                self.frame_con_resultado_bgr = frame_salida_bgr
                self.estado_con_resultado = estado
                self.pausar_lectura_hasta = time.time() + PAUSA_LECTURA_SEGUNDOS
                self.secuencia.clear()
                self.frames_previos.clear()
                self.leyendo = False
                self.frames_movimiento_consecutivos = 0
                self.frames_vistos = 0

            return self.responder(
                frame_salida_bgr,
                estado,
                forzar=estado.startswith(ESTADO_LEIDA),
            )
        except ModelNotFoundError:
            return self.responder(frame_bgr, MENSAJE_MODELO_NO_LISTO, forzar=True)
        except ValueError as error:
            return self.responder(
                frame_bgr,
                f"No se pudo realizar la prediccion: {error}",
                forzar=True,
            )
        except Exception as error:
            return self.responder(
                frame_bgr,
                f"Ocurrio un error al procesar la camara: {error}",
                forzar=True,
            )


def actualizar_estado(estado_nuevo):
    return armar_html_estado(estado_nuevo)


def actualizar_estado_desde_detector(detector: DetectorSeñasWebRTC):
    return armar_html_estado(detector.estado_actual)


def crear_interfaz() -> gr.Blocks:
    if WebRTC is None or VideoStreamHandler is None:
        with gr.Blocks(title=TITULO_PROYECTO) as interfaz:
            gr.Markdown(
                f"""
                # {TITULO_PROYECTO}

                Falta instalar FastRTC en el entorno.

                ```powershell
                C:\\venvs\\tp-senias-mediapipe\\Scripts\\Activate.ps1
                pip install -r requirements.txt
                python app.py
                ```
                """
            )

        return interfaz

    detector = DetectorSeñasWebRTC()

    with gr.Blocks(title=TITULO_PROYECTO, css=CSS_INTERFAZ) as interfaz:
        gr.Markdown(
            f"""
            # {TITULO_PROYECTO}

            Prototipo educativo de reconocimiento basico de señas usando
            MediaPipe Holistic, landmarks de pose y manos, y un modelo simple
            entrenado con scikit-learn.
            """
        )
        estado_camara = gr.HTML(value=armar_html_estado(ESTADO_ESPERANDO))
        temporizador_estado = gr.Timer(
            value=INTERVALO_ACTUALIZACION_ESTADO_SEGUNDOS,
            active=True,
        )
        camara = WebRTC(
            label="Camara en vivo",
            mode="send-receive",
            modality="video",
            width=ANCHO_CAMARA,
            height=ALTO_CAMARA,
            full_screen=False,
            mirror_webcam=False,
            elem_id="camara-fastrtc",
            button_labels={
                "start": "Iniciar deteccion",
                "stop": "Detener deteccion",
                "waiting": "Conectando",
            },
            track_constraints={
                "facingMode": "user",
                "width": {"ideal": ANCHO_CAMARA},
                "height": {"ideal": ALTO_CAMARA},
                "frameRate": {"ideal": FPS_SALIDA, "max": 30},
            },
            rtp_params={"degradationPreference": "maintain-framerate"},
        )

        camara.stream(
            fn=VideoStreamHandler(
                detector.procesar_frame,
                fps=FPS_SALIDA,
                skip_frames=True,
            ),
            inputs=[camara],
            outputs=[camara],
            concurrency_limit=1,
        )
        temporizador_estado.tick(
            fn=lambda: actualizar_estado_desde_detector(detector),
            outputs=[estado_camara],
            show_progress="hidden",
            queue=False,
        )

        if not Path(RUTA_MODELO).exists():
            gr.Markdown(f"**Estado del modelo:** {MENSAJE_MODELO_NO_LISTO}")

    return interfaz


if __name__ == "__main__":
    app = crear_interfaz()
    app.queue(default_concurrency_limit=1)
    app.launch(server_name="0.0.0.0", server_port=7860)
