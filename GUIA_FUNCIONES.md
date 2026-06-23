# Guia de funciones

Resumen rapido de las funciones propias del proyecto: que hacen y donde se usan.

## app.py

| Funcion/clase | Que hace | Donde se usa |
|---|---|---|
| `nombre_para_mostrar` | Convierte la etiqueta interna de rechazo en `NO RECONOCIDA`. | Estado y texto sobre la camara. |
| `armar_lineas_confianza` | Arma una linea por clase con su porcentaje de confianza. | `armar_estado`. |
| `armar_estado` | Construye el texto del panel Estado con umbral y porcentajes. | Resultado de prediccion. |
| `dibujar_resultado_centrado` | Dibuja solo la palabra detectada sobre la imagen de la camara usando Pillow. | `DetectorSeñasWebRTC.procesar_frame`. |
| `limpiar_texto_estado` | Limpia prefijos internos de streaming como `replace,value,`. | `armar_html_estado`. |
| `armar_html_estado` | Convierte el texto de estado en un panel HTML estable. | `actualizar_estado` e interfaz inicial. |
| `DetectorSeñasWebRTC` | Mantiene estado del stream: frames previos, secuencia, pausas, MediaPipe y resultado congelado. | Handler de FastRTC. |
| `obtener_detector_holistic` | Crea una unica instancia de MediaPipe Holistic para el stream. | `procesar_frame`. |
| `reiniciar_lectura` | Limpia lectura despues de la pausa de resultado. | `procesar_frame`. |
| `separar_puntos_clave` | Separa el vector de landmarks en pose, mano izquierda y mano derecha. | `calcular_movimiento_manos`. |
| `normalizar_manos_respecto_hombros` | Expresa las manos respecto del centro/ancho de hombros para ignorar movimientos de cabeza/cara/camara. | `calcular_movimiento_manos`. |
| `calcular_movimiento_manos` | Mide movimiento relativo de manos entre frames. | `hay_movimiento_de_seña`. |
| `hay_movimiento_de_seña` | Decide si hay movimiento suficiente para empezar a leer. | `procesar_frame`. |
| `preparar_inicio_lectura` | Arma la secuencia inicial usando frames previos para conservar la pose inicial. | `procesar_frame`. |
| `responder` | Devuelve el frame procesado y guarda el ultimo Estado para la interfaz. | `procesar_frame`. |
| `procesar_frame` | Recibe frames WebRTC, procesa landmarks, predice y pausa 3 segundos al leer una seña. | `VideoStreamHandler`. |
| `actualizar_estado` | Convierte un texto de estado en HTML para el panel. | `actualizar_estado_desde_detector`. |
| `actualizar_estado_desde_detector` | Lee el ultimo Estado guardado por el detector. | `gr.Timer` en `crear_interfaz`. |
| `crear_interfaz` | Construye la interfaz Gradio con componente FastRTC WebRTC. | Bloque `if __name__ == "__main__"`. |

## config.py

No tiene funciones. Define constantes compartidas:

| Constante | Uso |
|---|---|
| `SEÑAS` | Clases visibles que la app reconoce. |
| `SEÑA_DESCONOCIDA` | Clase interna para ejemplos negativos. |
| `SEÑAS_ENTRENAMIENTO` | Clases usadas al capturar y entrenar. |
| `ETIQUETA_DESCONOCIDA` | Texto interno para una prediccion rechazada. |
| `RUTA_DATOS` | Carpeta del dataset `.npy`. |
| `RUTA_MODELO` | Ruta del modelo entrenado. |
| `LARGO_SECUENCIA` | Cantidad de frames por muestra. |
| `MUESTRAS_OBJETIVO_POR_SEÑA` | Objetivo visual de muestras por clase en el colector. |
| `CONFIANZA_MINIMA_DETECCION` | Umbral de deteccion de MediaPipe. |
| `CONFIANZA_MINIMA_SEGUIMIENTO` | Umbral de seguimiento de MediaPipe. |
| `CONFIANZA_MINIMA_PREDICCION` | Umbral minimo para aceptar una prediccion del modelo. |

## train_model.py

| Funcion | Que hace | Donde se usa |
|---|---|---|
| `cargar_dataset` | Lee secuencias `.npy`, valida largo y arma secuencias/etiquetas. | `entrenar_modelo`. |
| `entrenar_modelo` | Entrena Random Forest, muestra metricas y guarda `models/sign_model.joblib`. | Al ejecutar `python train_model.py`. |

## utils/landmarks.py

| Funcion | Que hace | Donde se usa |
|---|---|---|
| `create_holistic` | Crea MediaPipe Holistic con umbrales de `config.py`. | App y colector. |
| `extract_keypoints` | Une pose, mano izquierda y mano derecha en un vector de 258 valores. | App y colector. |

## utils/features.py

| Funcion | Que hace | Donde se usa |
|---|---|---|
| `temporal_normalize_sequence` | Recorta el gesto activo y lo lleva a 30 frames. | `build_feature_vector`. |
| `normalize_sequence` | Normaliza pose y manos respecto de cara/hombros. | `build_feature_vector`. |
| `hand_face_features` | Genera distancias mano-cara/hombros por frame. | `build_feature_vector`. |
| `build_feature_vector` | Convierte una muestra de 30 frames en vector final para el modelo. | Entrenamiento y prediccion. |
| `build_feature_matrix` | Aplica `build_feature_vector` a muchas muestras. | `train_model.py`. |

## utils/prediction.py

| Funcion | Que hace | Donde se usa |
|---|---|---|
| `ModelNotFoundError` | Error claro cuando falta el modelo entrenado. | App y prediccion. |
| `load_model` | Carga `models/sign_model.joblib`. | Prediccion. |
| `predict_sequence_with_scores` | Predice etiqueta, confianza ganadora y porcentajes por clase. | App. |
| `predict_sequence` | Devuelve solo etiqueta y confianza. | Utilidad compatible con codigo anterior. |

## scripts/collect_dataset.py

| Funcion/clase | Que hace | Donde se usa |
|---|---|---|
| `EstadoColector` | Guarda estado de UI del colector. | Todo el colector. |
| `leer_frame_procesado` | Lee webcam local con OpenCV, espeja frame y extrae landmarks. | Loop de captura. |
| `capturar_muestra` | Captura los 30 frames de una muestra. | `capturar_lote`. |
| `capturar_lote` | Captura 1 o varias muestras seguidas. | Comandos del panel. |
| `ejecutar_colector` | Abre webcam local y ejecuta el colector. | Al ejecutar `python scripts/collect_dataset.py`. |
