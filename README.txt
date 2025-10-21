# Clasificación en vivo con Keras + Streamlit

## Archivos
- `app_streamlit.py`: Aplicación Streamlit con video en tiempo real y sección de resultados.
- `requirements.txt`: Paquetes necesarios.

Coloca en la misma carpeta tu `keras_Model.h5` y `labels.txt`.

## Pasos de ejecución (Windows / Linux / macOS)

```bash
# 1) Crear entorno virtual (ejemplos)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 2) Instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Ejecutar la app
streamlit run app_streamlit.py
```

> Si no se muestra la cámara, revisa permisos del navegador y prueba con Chrome. Algunas redes corporativas bloquean WebRTC; usa el modo alternativo con foto si es necesario.
