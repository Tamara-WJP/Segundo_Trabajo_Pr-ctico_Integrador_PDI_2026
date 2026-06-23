import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CARPETA_PROYECTO = Path(__file__).resolve().parents[1]
if str(CARPETA_PROYECTO) not in sys.path:
    sys.path.insert(0, str(CARPETA_PROYECTO))

from config import (
    LARGO_SECUENCIA,
    MUESTRAS_OBJETIVO_POR_SEÑA,
    RUTA_DATOS,
    SEÑAS_ENTRENAMIENTO,
)
from utils.drawing import (
    draw_holistic_landmarks as dibujar_landmarks_holistic,
    draw_status_text as dibujar_texto_captura,
    draw_text as dibujar_texto,
)
from utils.landmarks import create_holistic as crear_detector_holistic
from utils.landmarks import extract_keypoints as extraer_puntos_clave


NOMBRE_VENTANA = "Sena a texto - Captura de dataset"
ANCHO_PANEL = 360
TECLA_Q = ord("q")
TECLA_ESPACIO = 32
TAMANO_LOTE = 10
PAUSA_ENTRE_CAPTURAS_SEGUNDOS = 1.0
SEGUNDOS_CONFIRMAR_LIMPIEZA = 5.0


@dataclass
class EstadoColector:
    indice_seña_seleccionada: int = 0
    menu_abierto: bool = False
    comando_pendiente: str | None = None
    esta_ocupado: bool = False
    mensaje_estado: str = "Selecciona una seña y usa los botones."
    seña_limpieza_pendiente: str | None = None
    confirmar_limpieza_hasta: float = 0.0
    rectangulos_botones: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    rectangulos_opciones: list[tuple[str, tuple[int, int, int, int]]] = field(
        default_factory=list
    )

    @property
    def seña_seleccionada(self) -> str:
        return SEÑAS_ENTRENAMIENTO[self.indice_seña_seleccionada]

    def limpiar_confirmacion(self) -> None:
        self.seña_limpieza_pendiente = None
        self.confirmar_limpieza_hasta = 0.0


def crear_carpetas_dataset() -> None:
    for seña in SEÑAS_ENTRENAMIENTO:
        (Path(RUTA_DATOS) / seña).mkdir(parents=True, exist_ok=True)


def carpeta_de_seña(seña: str) -> Path:
    return Path(RUTA_DATOS) / seña


def contar_muestras(seña: str) -> int:
    return sum(1 for ruta in carpeta_de_seña(seña).glob("*.npy"))


def proximo_indice_muestra(seña: str) -> int:
    indices_existentes = [
        int(ruta.stem)
        for ruta in carpeta_de_seña(seña).glob("*.npy")
        if ruta.stem.isdigit()
    ]
    return max(indices_existentes, default=-1) + 1


def limpiar_muestras(seña: str) -> int:
    borradas = 0
    for ruta in carpeta_de_seña(seña).glob("*.npy"):
        ruta.unlink()
        borradas += 1
    return borradas


def leer_frame_procesado(camara, detector_holistic):
    lectura_ok, frame = camara.read()
    if not lectura_ok:
        return False, None, None, None

    frame = cv2.flip(frame, 1)
    imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen_rgb.flags.writeable = False
    resultados = detector_holistic.process(imagen_rgb)
    imagen_rgb.flags.writeable = True
    puntos_clave = extraer_puntos_clave(resultados)

    return True, frame, resultados, puntos_clave


def dibujar_boton(lienzo, rectangulo, texto: str, color) -> None:
    x1, y1, x2, y2 = rectangulo
    cv2.rectangle(lienzo, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(lienzo, (x1, y1), (x2, y2), (210, 210, 210), 1)
    dibujar_texto(lienzo, texto, (x1 + 14, y1 + 32), scale=0.58)


def dibujar_panel(frame, estado: EstadoColector) -> np.ndarray:
    alto, ancho = frame.shape[:2]
    panel = np.full((alto, ANCHO_PANEL, 3), (32, 34, 38), dtype=np.uint8)
    lienzo = np.hstack([frame, panel])

    inicio_panel_x = ancho
    margen = 18
    izquierda = inicio_panel_x + margen
    derecha = inicio_panel_x + ANCHO_PANEL - margen

    estado.rectangulos_botones = {}
    estado.rectangulos_opciones = []

    dibujar_texto(lienzo, "Seña a texto Dataset", (izquierda, 34), scale=0.74)
    dibujar_texto(lienzo, "Control de captura", (izquierda, 64), scale=0.55)

    y = 100
    dibujar_texto(lienzo, "Seña a capturar", (izquierda, y), scale=0.55)
    rectangulo_menu = (izquierda, y + 14, derecha, y + 54)
    estado.rectangulos_botones["menu"] = rectangulo_menu
    cv2.rectangle(lienzo, rectangulo_menu[:2], rectangulo_menu[2:], (64, 69, 78), -1)
    cv2.rectangle(lienzo, rectangulo_menu[:2], rectangulo_menu[2:], (220, 220, 220), 1)
    dibujar_texto(lienzo, estado.seña_seleccionada, (izquierda + 12, y + 42), scale=0.64)
    dibujar_texto(lienzo, "v", (derecha - 24, y + 42), scale=0.64)

    y += 84
    total = contar_muestras(estado.seña_seleccionada)
    objetivo = MUESTRAS_OBJETIVO_POR_SEÑA
    color_contador = (120, 210, 140) if total >= objetivo else (245, 205, 95)
    dibujar_texto(lienzo, "Muestras registradas", (izquierda, y), scale=0.55)
    cv2.rectangle(lienzo, (izquierda, y + 14), (derecha, y + 70), (45, 48, 54), -1)
    dibujar_texto(lienzo, f"{total} / {objetivo}", (izquierda + 14, y + 54), scale=1.0)
    cv2.circle(lienzo, (derecha - 22, y + 42), 8, color_contador, -1)

    y += 108
    rectangulo_capturar_una = (izquierda, y, derecha, y + 48)
    estado.rectangulos_botones["capturar_una"] = rectangulo_capturar_una
    dibujar_boton(lienzo, rectangulo_capturar_una, "Capturar 1", (54, 118, 196))

    y += 62
    rectangulo_capturar_lote = (izquierda, y, derecha, y + 48)
    estado.rectangulos_botones["capturar_lote"] = rectangulo_capturar_lote
    dibujar_boton(lienzo, rectangulo_capturar_lote, f"Capturar {TAMANO_LOTE}", (42, 148, 118))

    y += 62
    limpieza_armada = (
        estado.seña_limpieza_pendiente == estado.seña_seleccionada
        and time.time() < estado.confirmar_limpieza_hasta
    )
    texto_limpieza = "Confirmar limpieza" if limpieza_armada else "Limpiar seña"
    color_limpieza = (52, 84, 130) if limpieza_armada else (136, 65, 65)
    rectangulo_limpiar = (izquierda, y, derecha, y + 48)
    estado.rectangulos_botones["limpiar"] = rectangulo_limpiar
    dibujar_boton(lienzo, rectangulo_limpiar, texto_limpieza, color_limpieza)

    y += 74
    cv2.rectangle(lienzo, (izquierda, y), (derecha, min(y + 92, alto - 64)), (45, 48, 54), -1)
    dibujar_texto(lienzo, "Estado", (izquierda + 12, y + 28), scale=0.55)
    dibujar_texto(lienzo, estado.mensaje_estado[:34], (izquierda + 12, y + 60), scale=0.5)
    if len(estado.mensaje_estado) > 34:
        dibujar_texto(lienzo, estado.mensaje_estado[34:68], (izquierda + 12, y + 84), scale=0.5)

    dibujar_texto(lienzo, "Q: salir | Espacio: capturar 1", (izquierda, alto - 24), scale=0.48)

    if estado.menu_abierto:
        inicio_opciones = rectangulo_menu[3] + 4
        for indice, seña in enumerate(SEÑAS_ENTRENAMIENTO):
            rectangulo_opcion = (
                rectangulo_menu[0],
                inicio_opciones + indice * 38,
                rectangulo_menu[2],
                inicio_opciones + (indice + 1) * 38,
            )
            estado.rectangulos_opciones.append((seña, rectangulo_opcion))
            color = (86, 96, 112) if seña == estado.seña_seleccionada else (52, 56, 64)
            cv2.rectangle(lienzo, rectangulo_opcion[:2], rectangulo_opcion[2:], color, -1)
            cv2.rectangle(lienzo, rectangulo_opcion[:2], rectangulo_opcion[2:], (200, 200, 200), 1)
            dibujar_texto(lienzo, seña, (rectangulo_opcion[0] + 12, rectangulo_opcion[1] + 27), scale=0.55)

    return lienzo


def punto_en_rectangulo(x: int, y: int, rectangulo: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rectangulo
    return x1 <= x <= x2 and y1 <= y <= y2


def manejar_mouse(evento, x, y, _opciones_mouse, estado: EstadoColector) -> None:
    if evento != cv2.EVENT_LBUTTONDOWN or estado.esta_ocupado:
        return

    if estado.menu_abierto:
        for seña, rectangulo in estado.rectangulos_opciones:
            if punto_en_rectangulo(x, y, rectangulo):
                estado.indice_seña_seleccionada = SEÑAS_ENTRENAMIENTO.index(seña)
                estado.menu_abierto = False
                estado.limpiar_confirmacion()
                estado.mensaje_estado = f"Seleccion actual: {seña}."
                return
        estado.menu_abierto = False

    if punto_en_rectangulo(x, y, estado.rectangulos_botones.get("menu", (0, 0, 0, 0))):
        estado.menu_abierto = not estado.menu_abierto
        return

    if punto_en_rectangulo(x, y, estado.rectangulos_botones.get("capturar_una", (0, 0, 0, 0))):
        estado.comando_pendiente = "capturar_una"
        return

    if punto_en_rectangulo(x, y, estado.rectangulos_botones.get("capturar_lote", (0, 0, 0, 0))):
        estado.comando_pendiente = "capturar_lote"
        return

    if punto_en_rectangulo(x, y, estado.rectangulos_botones.get("limpiar", (0, 0, 0, 0))):
        estado.comando_pendiente = "limpiar"


def mostrar_frame(frame, estado: EstadoColector) -> int:
    lienzo = dibujar_panel(frame, estado)
    cv2.imshow(NOMBRE_VENTANA, lienzo)
    return cv2.waitKey(10) & 0xFF


def guardar_muestra(seña: str, frames: list[np.ndarray]) -> Path:
    numero_muestra = proximo_indice_muestra(seña)
    ruta_salida = carpeta_de_seña(seña) / f"{numero_muestra}.npy"
    np.save(ruta_salida, np.array(frames, dtype=np.float32))
    return ruta_salida


def capturar_muestra(camara, detector_holistic, estado: EstadoColector) -> tuple[list[np.ndarray], bool]:
    frames = []
    seña = estado.seña_seleccionada
    numero_muestra = proximo_indice_muestra(seña)

    for numero_frame in range(LARGO_SECUENCIA):
        lectura_ok, frame, resultados, puntos_clave = leer_frame_procesado(camara, detector_holistic)
        if not lectura_ok:
            estado.mensaje_estado = "No se pudo leer la webcam."
            return frames, False

        frames.append(puntos_clave)
        dibujar_landmarks_holistic(frame, resultados)
        dibujar_texto_captura(
            frame,
            action=seña,
            sequence_number=numero_muestra,
            frame_number=numero_frame + 1,
            total_frames=LARGO_SECUENCIA,
        )
        estado.mensaje_estado = f"Grabando {seña}: frame {numero_frame + 1}/{LARGO_SECUENCIA}"

        if mostrar_frame(frame, estado) == TECLA_Q:
            return frames, False

    return frames, True


def esperar_entre_capturas(camara, detector_holistic, estado: EstadoColector) -> bool:
    tiempo_fin = time.time() + PAUSA_ENTRE_CAPTURAS_SEGUNDOS

    while time.time() < tiempo_fin:
        lectura_ok, frame, resultados, _ = leer_frame_procesado(camara, detector_holistic)
        if not lectura_ok:
            estado.mensaje_estado = "No se pudo leer la webcam."
            return False

        dibujar_landmarks_holistic(frame, resultados)
        restante = max(0.0, tiempo_fin - time.time())
        estado.mensaje_estado = f"Siguiente captura en {restante:.1f}s"

        if mostrar_frame(frame, estado) == TECLA_Q:
            return False

    return True


def capturar_lote(camara, detector_holistic, estado: EstadoColector, cantidad: int) -> bool:
    estado.esta_ocupado = True
    estado.menu_abierto = False
    estado.limpiar_confirmacion()

    try:
        for indice in range(cantidad):
            seña = estado.seña_seleccionada
            estado.mensaje_estado = f"Capturando {indice + 1}/{cantidad} para {seña}."
            frames, completa = capturar_muestra(camara, detector_holistic, estado)

            if not completa or len(frames) != LARGO_SECUENCIA:
                estado.mensaje_estado = "Captura cancelada."
                return False

            ruta_salida = guardar_muestra(seña, frames)
            estado.mensaje_estado = f"Guardado: {ruta_salida.name}"
            print(f"[OK] Guardado: {ruta_salida}")

            if indice < cantidad - 1 and not esperar_entre_capturas(camara, detector_holistic, estado):
                estado.mensaje_estado = "Captura cancelada."
                return False

    finally:
        estado.esta_ocupado = False

    total = contar_muestras(estado.seña_seleccionada)
    estado.mensaje_estado = f"Listo. {total} muestras para {estado.seña_seleccionada}."
    return True


def manejar_comando_limpiar(estado: EstadoColector) -> None:
    ahora = time.time()
    seña = estado.seña_seleccionada
    limpieza_confirmada = (
        estado.seña_limpieza_pendiente == seña and ahora < estado.confirmar_limpieza_hasta
    )

    if not limpieza_confirmada:
        estado.seña_limpieza_pendiente = seña
        estado.confirmar_limpieza_hasta = ahora + SEGUNDOS_CONFIRMAR_LIMPIEZA
        estado.mensaje_estado = "Volver a tocar para confirmar limpieza."
        return

    borradas = limpiar_muestras(seña)
    estado.limpiar_confirmacion()
    estado.mensaje_estado = f"Se eliminaron {borradas} muestras de {seña}."
    print(f"[INFO] Se eliminaron {borradas} muestras de {seña}.")


def manejar_comando_pendiente(camara, detector_holistic, estado: EstadoColector) -> bool:
    comando = estado.comando_pendiente
    estado.comando_pendiente = None

    if comando == "capturar_una":
        return capturar_lote(camara, detector_holistic, estado, cantidad=1)

    if comando == "capturar_lote":
        return capturar_lote(camara, detector_holistic, estado, cantidad=TAMANO_LOTE)

    if comando == "limpiar":
        manejar_comando_limpiar(estado)
        return True

    return True


def ejecutar_colector() -> None:
    crear_carpetas_dataset()

    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        print("[ERROR] No se pudo abrir la webcam.")
        return

    estado = EstadoColector()
    cv2.namedWindow(NOMBRE_VENTANA)
    cv2.setMouseCallback(NOMBRE_VENTANA, manejar_mouse, estado)

    print("[INFO] Captura iniciada.")
    print("[INFO] Usa los botones del panel lateral para capturar o limpiar.")
    print("[INFO] Presiona 'q' en la ventana de video para salir.")

    detener = False

    with crear_detector_holistic() as detector_holistic:
        while not detener:
            lectura_ok, frame, resultados, _ = leer_frame_procesado(camara, detector_holistic)
            if not lectura_ok:
                print("[ERROR] No se pudo leer un frame de la webcam.")
                break

            if estado.seña_limpieza_pendiente and time.time() >= estado.confirmar_limpieza_hasta:
                estado.limpiar_confirmacion()

            dibujar_landmarks_holistic(frame, resultados)
            tecla = mostrar_frame(frame, estado)

            if tecla == TECLA_Q:
                detener = True
            elif tecla == TECLA_ESPACIO:
                estado.comando_pendiente = "capturar_una"

            if estado.comando_pendiente:
                detener = not manejar_comando_pendiente(camara, detector_holistic, estado)

    camara.release()
    cv2.destroyAllWindows()
    print("[INFO] Captura finalizada.")


if __name__ == "__main__":
    ejecutar_colector()
