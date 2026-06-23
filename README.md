---
title: Seña a Texto LSA
sdk: gradio
app_file: app.py
python_version: 3.11
---

# Seña a Texto (LSA)

Prototipo educativo de reconocimiento basico de señas usando Python,
MediaPipe Holistic, OpenCV, scikit-learn y Gradio.

El objetivo del proyecto es capturar landmarks desde webcam, generar un dataset
propio, entrenar un modelo liviano y probar predicciones desde una interfaz web.

## Alcance

La app reconoce un conjunto acotado de cinco señas:

- HOLA
- GRACIAS
- SI
- NO
- AYUDA

Tambien se usa una clase interna `OTRA` para ejemplos negativos. Si el modelo
predice `OTRA` o si la confianza queda por debajo de
`CONFIANZA_MINIMA_PREDICCION`, la app muestra `NO RECONOCIDA`.

Este proyecto no es un traductor completo de lengua de señas. Es un prototipo
academico acotado.

## Integrante

- Tamara Peña

## Tecnologias

- Python 3.11
- MediaPipe Holistic
- OpenCV
- NumPy
- scikit-learn
- joblib
- Gradio
- Pillow

## Estructura

```text
tp-senias-mediapipe/
|-- app.py
|-- config.py
|-- train_model.py
|-- requirements.txt
|-- README.md
|-- GUIA_FUNCIONES.md
|-- scripts/
|   `-- collect_dataset.py
|-- utils/
|   |-- __init__.py
|   |-- landmarks.py
|   |-- drawing.py
|   |-- features.py
|   `-- prediction.py
|-- models/
|   `-- sign_model.joblib
`-- data/
    `-- sequences/
        |-- HOLA/
        |-- GRACIAS/
        |-- SI/
        |-- NO/
        |-- AYUDA/
        `-- OTRA/
```

## Instalacion local

Usar Python 3.11. En Windows conviene crear el entorno virtual fuera de rutas
con acentos o caracteres especiales.

Ejemplo recomendado:

```powershell
py -3.11 -m venv C:\venvs\tp-senias-mediapipe
C:\venvs\tp-senias-mediapipe\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si el entorno ya existe:

```powershell
C:\venvs\tp-senias-mediapipe\Scripts\Activate.ps1
```

## Configuracion

El archivo `config.py` centraliza los valores principales:

- `SEÑAS`: señas reconocidas por la app.
- `SEÑA_DESCONOCIDA`: clase interna para ejemplos negativos.
- `SEÑAS_ENTRENAMIENTO`: clases usadas al capturar y entrenar.
- `LARGO_SECUENCIA`: cantidad de frames por muestra. Actualmente 30.
- `CONFIANZA_MINIMA_DETECCION`: confianza minima de MediaPipe para detectar.
- `CONFIANZA_MINIMA_SEGUIMIENTO`: confianza minima de MediaPipe para seguir landmarks.
- `CONFIANZA_MINIMA_PREDICCION`: confianza minima del modelo para aceptar una
  prediccion. Actualmente esta en `0.3`.

Cambiar `CONFIANZA_MINIMA_PREDICCION` no requiere reentrenar. Solo hay que
reiniciar `python app.py`.

## Captura de dataset

Ejecutar:

```powershell
python scripts\collect_dataset.py
```

El colector usa webcam local, MediaPipe Holistic y guarda cada muestra como
`.npy` en:

```text
data/sequences/NOMBRE_SEÑA/numero.npy
```

Cada muestra contiene 30 frames. El panel permite seleccionar la seña, capturar
una muestra, capturar 10 muestras seguidas y limpiar muestras de una clase.

Para mejorar la separacion entre gestos parecidos, conviene capturar muestras
con inicio y final claros. En HOLA y GRACIAS, marcar bien la diferencia entre
frente/sien y menton/boca.

## Entrenamiento

Ejecutar:

```powershell
python train_model.py
```

El entrenamiento:

- lee las secuencias `.npy`;
- valida cantidad y forma de datos;
- recorta el tramo activo del movimiento y lo reescala a 30 frames;
- genera features relativas a cara, hombros, manos, brazos y movimiento;
- entrena un `RandomForestClassifier`;
- muestra accuracy, classification report y matriz de confusion;
- guarda el modelo en `models/sign_model.joblib`.

Hay que reentrenar cuando se agregan, borran o corrigen muestras del dataset.

## Ejecucion de la app local

Ejecutar:

```powershell
python app.py
```

Abrir:

```text
http://localhost:7860
```

La app trabaja con la camara del navegador. Esto permite que tambien funcione
en Hugging Face Spaces, donde el servidor no puede abrir una webcam local con
`cv2.VideoCapture(0)`.

Flujo de estados:

- `ESPERANDO SEÑA`: la app esta activa, pero aun no hay movimiento suficiente.
- `LEYENDO SEÑA`: hay movimiento y la app esta intentando clasificarlo.
- `SEÑA LEIDA`: se detecto una clase y se congelan resultado y porcentajes por
  3 segundos para poder leer el campo Estado.

Sobre la camara se muestra solo la palabra detectada. En el campo Estado se
muestran la palabra, el umbral y los porcentajes para todas las clases.

## Deploy en Hugging Face Spaces

No hace falta Docker para este proyecto. Conviene crear un Space con SDK
`Gradio` y subir el repo con estos archivos:

- `app.py`
- `config.py`
- `requirements.txt`
- `README.md`
- `GUIA_FUNCIONES.md`
- carpeta `utils/`
- archivo `models/sign_model.joblib`

El dataset (`data/sequences/`) y `scripts/collect_dataset.py` pueden quedar
fuera del Space si solo se quiere usar la demo entrenada.

Como `.gitignore` ignora modelos `.joblib`, para incluir el modelo en el repo
hay que agregarlo de forma forzada:

```powershell
git add -f models/sign_model.joblib
```

Luego subir el repo al Space. Hugging Face instala `requirements.txt` y ejecuta
`app.py` automaticamente por el bloque YAML del README.

## Archivos auxiliares

`GUIA_FUNCIONES.md` resume las funciones principales, que hacen y donde se usan.

## Mejoras posibles

- Agregar mas muestras de señas que se confunden.
- Capturar algunas muestras a velocidades distintas.
- Ajustar el umbral de movimiento si la app empieza a leer demasiado pronto o
  demasiado tarde.
- Evaluar otros modelos si Random Forest deja de alcanzar.
