# Guia de funciones

Resumen rapido de las funciones propias del proyecto: que hacen y donde se usan.

## app.py

| Funcion | Que hace | Donde se usa |
|---|---|---|
| `obtener_detector_holistic` | Crea una unica instancia compartida de MediaPipe Holistic para procesar frames de la webcam. | `analizar_frame_webcam`. |
| `crear_estado_deteccion` | Crea el estado interno de lectura: secuencia de 30 frames, pausa, ultimo frame y ultimo estado mostrado. | `crear_interfaz`, `analizar_frame_webcam` y `reiniciar_lectura`. |
| `obtener_secuencia` | Recupera la cola de frames del estado y la repara si Gradio la devuelve como lista. | Lectura, reinicio y calculo de movimiento. |
| `reiniciar_lectura_estado` | Limpia la secuencia y deja la deteccion lista para leer una seña nueva. | Cuando termina la pausa de 3 segundos. |
| `procesar_frame_webcam` | Recibe un frame RGB del navegador, lo espeja, ejecuta MediaPipe y extrae landmarks. | `analizar_frame_webcam`. |
| `nombre_para_mostrar` | Convierte la etiqueta interna de rechazo en `NO RECONOCIDA`. | Estado de la app y texto sobre la camara. |
| `armar_lineas_confianza` | Arma una linea por clase con su porcentaje de confianza. | `armar_estado`. |
| `armar_estado` | Construye el texto del campo Estado, incluyendo umbral y porcentajes. | Resultado de prediccion. |
| `calcular_movimiento_manos` | Mide cuanto se mueven las manos entre frames recientes. | `hay_movimiento_de_seña`. |
| `hay_movimiento_de_seña` | Decide si hay movimiento suficiente para empezar a leer una seña. | `analizar_frame_webcam`. |
| `dibujar_resultado_centrado` | Dibuja solo la palabra detectada sobre la imagen de la camara, usando Pillow para soportar `Ñ`. | `analizar_frame_webcam`. |
| `analizar_frame_webcam` | Funcion principal de la app: recibe frames del navegador, espera movimiento, predice, congela el resultado 3 segundos y vuelve a esperar. | Evento `stream` del componente de webcam de Gradio. |
| `reiniciar_lectura` | Limpia la salida, el estado y la secuencia interna. | Boton `Reiniciar lectura`. |
| `crear_interfaz` | Construye la interfaz Gradio preparada para local y Hugging Face Spaces. | Bloque `if __name__ == "__main__"`. |

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
| `cargar_dataset` | Lee las secuencias `.npy`, valida largo y arma secuencias/etiquetas. | `entrenar_modelo`. |
| `entrenar_modelo` | Entrena Random Forest, muestra metricas y guarda `models/sign_model.joblib`. | Al ejecutar `python train_model.py`. |

## utils/landmarks.py

| Funcion | Que hace | Donde se usa |
|---|---|---|
| `_require_legacy_solutions` | Valida que MediaPipe tenga `mp.solutions`. | Al importar el modulo. |
| `_warn_windows_unicode_path` | Evita usar MediaPipe desde rutas Windows con caracteres no ASCII. | `create_holistic`. |
| `create_holistic` | Crea la instancia de MediaPipe Holistic con los umbrales de `config.py`. | App y colector. |
| `_extract_pose` | Convierte landmarks de pose en vector numerico fijo. | `extract_keypoints`. |
| `_extract_hand` | Convierte landmarks de una mano en vector numerico fijo. | `extract_keypoints`. |
| `extract_keypoints` | Une pose, mano izquierda y mano derecha en un vector de 258 valores. | App y colector. |

## utils/features.py

| Funcion | Que hace | Donde se usa |
|---|---|---|
| `_split_sequence` | Separa una secuencia en pose, mano izquierda y mano derecha. | Varias funciones del extractor. |
| `_smooth_scores` | Suaviza puntajes de movimiento entre frames. | `_motion_scores`. |
| `_motion_scores` | Calcula movimiento de manos y brazos para detectar el tramo activo. | `_active_segment_bounds`. |
| `_active_segment_bounds` | Encuentra inicio y fin del movimiento principal. | `temporal_normalize_sequence`. |
| `_resample_frames` | Reescala un tramo a 30 frames. | `temporal_normalize_sequence`. |
| `temporal_normalize_sequence` | Recorta el gesto activo y lo lleva a 30 frames. | `build_feature_vector`. |
| `_pose_anchor_and_scale` | Usa cara/hombros para normalizar posicion y escala. | Normalizacion y features relativas. |
| `_normalize_hand` | Normaliza una mano respecto del cuerpo. | `normalize_sequence`. |
| `normalize_sequence` | Normaliza pose y manos para reducir dependencia de distancia/camara. | `build_feature_vector`. |
| `_mean_point` | Calcula promedio de puntos visibles. | `hand_face_features`. |
| `_relative_point_features` | Calcula vector y distancia entre mano y objetivo corporal. | `hand_face_features`. |
| `hand_face_features` | Genera distancias mano-cara/hombros por frame. | `build_feature_vector`. |
| `build_feature_vector` | Convierte una muestra de 30 frames en vector final para el modelo. | Entrenamiento y prediccion. |
| `build_feature_matrix` | Aplica `build_feature_vector` a muchas muestras. | `train_model.py`. |

## utils/prediction.py

| Funcion | Que hace | Donde se usa |
|---|---|---|
| `ModelNotFoundError` | Error claro cuando falta el modelo entrenado. | App y prediccion. |
| `load_model` | Carga `models/sign_model.joblib`. | Prediccion. |
| `_validate_sequence` | Verifica forma esperada `(30, 258)`. | `_prepare_features`. |
| `_prepare_features` | Valida version de features y arma entrada del modelo. | `predict_sequence_with_scores`. |
| `predict_sequence_with_scores` | Predice etiqueta, confianza ganadora y porcentajes por clase. | App. |
| `predict_sequence` | Devuelve solo etiqueta y confianza. | Utilidad compatible con codigo anterior. |

## utils/drawing.py

| Funcion | Que hace | Donde se usa |
|---|---|---|
| `draw_holistic_landmarks` | Dibuja pose y manos sobre el frame. | Colector. |
| `medir_texto` | Calcula ancho y alto de texto usando Pillow. | App y dibujo de texto. |
| `draw_text` | Escribe texto Unicode con Pillow sobre frames OpenCV. | App y colector. |
| `draw_status_text` | Muestra clase, secuencia y frame durante captura. | Colector. |
| `draw_wait_text` | Muestra pantalla de espera. | Funcion auxiliar disponible. |
| `draw_countdown_text` | Muestra cuenta regresiva. | Funcion auxiliar disponible. |

## scripts/collect_dataset.py

| Funcion/clase | Que hace | Donde se usa |
|---|---|---|
| `EstadoColector` | Guarda estado de UI del colector: clase seleccionada, botones y mensajes. | Todo el colector. |
| `crear_carpetas_dataset` | Crea carpetas por clase si faltan. | `ejecutar_colector`. |
| `carpeta_de_seña` | Devuelve carpeta de una clase. | Conteo, guardado y limpieza. |
| `contar_muestras` | Cuenta muestras `.npy` de una clase. | Panel del colector. |
| `proximo_indice_muestra` | Calcula el proximo numero de archivo. | `guardar_muestra`. |
| `limpiar_muestras` | Borra muestras de una clase. | `manejar_comando_limpiar`. |
| `leer_frame_procesado` | Lee webcam, espeja frame y extrae landmarks. | Loop de captura. |
| `dibujar_boton` | Dibuja botones del panel OpenCV. | `dibujar_panel`. |
| `dibujar_panel` | Dibuja panel lateral del colector. | `mostrar_frame`. |
| `punto_en_rectangulo` | Detecta clicks dentro de un rectangulo. | `manejar_mouse`. |
| `manejar_mouse` | Procesa clicks del usuario en el panel. | Callback OpenCV. |
| `mostrar_frame` | Muestra frame + panel y lee teclado. | Loop de captura. |
| `guardar_muestra` | Guarda una muestra `.npy`. | `capturar_lote`. |
| `capturar_muestra` | Captura los 30 frames de una muestra. | `capturar_lote`. |
| `esperar_entre_capturas` | Espera entre capturas por lote. | `capturar_lote`. |
| `capturar_lote` | Captura 1 o varias muestras seguidas. | Comandos del panel. |
| `manejar_comando_limpiar` | Pide confirmacion y borra muestras. | `manejar_comando_pendiente`. |
| `manejar_comando_pendiente` | Ejecuta comando pendiente del panel. | Loop principal. |
| `ejecutar_colector` | Abre webcam y ejecuta el colector. | Al ejecutar `python scripts/collect_dataset.py`. |
