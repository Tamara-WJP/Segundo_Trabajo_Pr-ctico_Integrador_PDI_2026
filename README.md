# Seña a Texto (LSA)

Prototipo educativo de reconocimiento basico de señas usando Python,
MediaPipe Holistic, OpenCV, scikit-learn, Gradio y FastRTC.

La app reconoce HOLA, GRACIAS, SI, NO y AYUDA. Tambien usa una clase interna
`OTRA` para ejemplos negativos. Si la confianza queda por debajo de
`CONFIANZA_MINIMA_PREDICCION`, muestra `NO RECONOCIDA`.

## Integrante

- Tamara Peña

## Instalacion local

Usar Python 3.11. En Windows conviene crear el entorno virtual fuera de rutas
con acentos o caracteres especiales.

```powershell
py -3.11 -m venv C:\venvs\tp-senias-mediapipe
C:\venvs\tp-senias-mediapipe\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuracion

`config.py` centraliza:

- `SEÑAS`: señas reconocidas por la app.
- `SEÑA_DESCONOCIDA`: clase interna para ejemplos negativos.
- `LARGO_SECUENCIA`: cantidad de frames por muestra. Actualmente 30.
- `CONFIANZA_MINIMA_DETECCION`: umbral de MediaPipe para detectar.
- `CONFIANZA_MINIMA_SEGUIMIENTO`: umbral de MediaPipe para seguir landmarks.
- `CONFIANZA_MINIMA_PREDICCION`: umbral del modelo. Actualmente `0.3`.

Cambiar `CONFIANZA_MINIMA_PREDICCION` no requiere reentrenar. Solo hay que
reiniciar `python app.py`.

## Captura de dataset

```powershell
python scripts\collect_dataset.py
```

El colector usa webcam local con OpenCV y guarda muestras `.npy` en:

```text
data/sequences/NOMBRE_SEÑA/numero.npy
```

Cada muestra contiene 30 frames.

## Entrenamiento

```powershell
python train_model.py
```

El entrenamiento:

- lee secuencias `.npy`;
- recorta el tramo activo del movimiento y lo reescala a 30 frames;
- genera features relativas a cara, hombros, manos, brazos y movimiento;
- entrena un `RandomForestClassifier`;
- guarda el modelo en `models/sign_model.joblib`.

Hay que reentrenar cuando se agregan, borran o corrigen muestras del dataset.

## Ejecucion de la app

```powershell
python app.py
```

Abrir:

```text
http://localhost:7860
```

La app usa FastRTC/WebRTC para tomar la camara desde el navegador y procesar cada
frame en Python. Se usa `skip_frames=True` para evitar acumulacion de frames si
MediaPipe tarda en procesar.

Flujo de estados:

- `ESPERANDO SEÑA`: la app esta activa, pero aun no hay movimiento suficiente.
- `LEYENDO SEÑA`: hay movimiento y se esta juntando la secuencia de 30 frames.
- `SEÑA LEIDA`: se detecto una clase y se congelan resultado y porcentajes por
  3 segundos para poder leer el campo Estado.

Sobre la camara se muestra solo la palabra detectada. En el panel Estado se
muestran la palabra, el umbral y los porcentajes para todas las clases.

## Hugging Face Spaces

Para subir la demo, incluir como minimo:

- `app.py`
- `config.py`
- `requirements.txt`
- `README.md`
- carpeta `utils/`
- archivo `models/sign_model.joblib`

Como `.gitignore` ignora modelos `.joblib`, agregar el modelo con:

```powershell
git add -f models/sign_model.joblib
```

FastRTC permite usar camara del navegador en un deploy web. En Spaces puede ser
necesario configurar WebRTC/TURN si la conexion no inicia correctamente.

## Archivos auxiliares

`GUIA_FUNCIONES.md` resume las funciones principales, que hacen y donde se usan.
