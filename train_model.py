from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import (
    CONFIANZA_MINIMA_PREDICCION,
    LARGO_SECUENCIA,
    RUTA_DATOS,
    RUTA_MODELO,
    SEÑA_DESCONOCIDA,
    SEÑAS_ENTRENAMIENTO,
)
from utils.features import FEATURE_VERSION as VERSION_CARACTERISTICAS
from utils.features import build_feature_matrix as armar_matriz_caracteristicas


def cargar_dataset() -> tuple[np.ndarray, np.ndarray]:
    secuencias = []
    etiquetas = []

    for seña in SEÑAS_ENTRENAMIENTO:
        carpeta_seña = Path(RUTA_DATOS) / seña
        if not carpeta_seña.exists():
            print(f"[AVISO] No existe la carpeta para {seña}: {carpeta_seña}")
            continue

        for ruta_archivo in sorted(carpeta_seña.glob("*.npy")):
            secuencia = np.load(ruta_archivo)

            if secuencia.shape[0] != LARGO_SECUENCIA:
                print(
                    f"[AVISO] Se ignora {ruta_archivo.name}: "
                    f"se esperaban {LARGO_SECUENCIA} frames y tiene {secuencia.shape[0]}."
                )
                continue

            secuencias.append(secuencia)
            etiquetas.append(seña)

    if not secuencias:
        return np.array([]), np.array([])

    return np.array(secuencias, dtype=np.float32), np.array(etiquetas)


def entrenar_modelo() -> None:
    print("[INFO] Leyendo dataset...")
    secuencias, etiquetas = cargar_dataset()

    if secuencias.size == 0 or etiquetas.size == 0:
        print(
            "[ERROR] No se encontraron muestras validas. "
            "Ejecuta primero scripts/collect_dataset.py."
        )
        return

    if len(set(etiquetas)) < 2:
        print("[ERROR] Se necesitan muestras de al menos dos clases para entrenar.")
        return

    if SEÑA_DESCONOCIDA not in set(etiquetas):
        print(
            f"[AVISO] No hay muestras de {SEÑA_DESCONOCIDA}. "
            "El rechazo de gestos no reconocidos sera poco confiable."
        )

    señas_unicas, cantidades = np.unique(etiquetas, return_counts=True)
    print("[INFO] Muestras por clase:")
    for seña, cantidad in zip(señas_unicas, cantidades):
        print(f"  - {seña}: {cantidad}")

    matriz_caracteristicas = armar_matriz_caracteristicas(secuencias)
    codigos_etiquetas = np.unique(etiquetas, return_inverse=True)[1]
    puede_estratificar = min(np.bincount(codigos_etiquetas)) >= 2
    estratificacion = etiquetas if puede_estratificar else None

    caracteristicas_entrenamiento, caracteristicas_prueba, etiquetas_entrenamiento, etiquetas_prueba = (
        train_test_split(
            matriz_caracteristicas,
            etiquetas,
            test_size=0.25,
            random_state=42,
            stratify=estratificacion,
        )
    )

    print(f"[INFO] Muestras de entrenamiento: {len(caracteristicas_entrenamiento)}")
    print(f"[INFO] Muestras de prueba: {len(caracteristicas_prueba)}")

    modelo = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    print("[INFO] Entrenando RandomForestClassifier...")
    modelo.fit(caracteristicas_entrenamiento, etiquetas_entrenamiento)

    etiquetas_predichas = modelo.predict(caracteristicas_prueba)
    precision = accuracy_score(etiquetas_prueba, etiquetas_predichas)

    print(f"[RESULTADO] Accuracy: {precision:.4f}")
    print("[RESULTADO] Classification report:")
    print(classification_report(etiquetas_prueba, etiquetas_predichas, zero_division=0))
    print("[RESULTADO] Matriz de confusion:")
    nombres_etiquetas = [str(etiqueta) for etiqueta in sorted(set(etiquetas))]
    print(nombres_etiquetas)
    print(confusion_matrix(etiquetas_prueba, etiquetas_predichas, labels=nombres_etiquetas))

    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": modelo,
            "classes": list(modelo.classes_),
            "sequence_length": LARGO_SECUENCIA,
            "input_size": matriz_caracteristicas.shape[1],
            "feature_version": VERSION_CARACTERISTICAS,
            "min_prediction_confidence": CONFIANZA_MINIMA_PREDICCION,
            "unknown_action": SEÑA_DESCONOCIDA,
            "seña_desconocida": SEÑA_DESCONOCIDA,
        },
        RUTA_MODELO,
    )
    print(f"[INFO] Modelo guardado en: {RUTA_MODELO}")


if __name__ == "__main__":
    entrenar_modelo()
