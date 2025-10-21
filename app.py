import time
import numpy as np
import cv2
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.io as pio
import io, zipfile
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Reconocimiento en Vivo", page_icon="🎥", layout="wide")

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
DB_PATH = "database.db"

# ---------------------------
# CONFIGURACIÓN DE KALEIDO
# ---------------------------
pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 1000
pio.kaleido.scope.default_height = 600

# ---- CARGA DE MODELO Y LABELS ----
@st.cache_resource
def load_model_cached(path):
    return load_model(path, compile=False)

@st.cache_data
def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

try:
    model = load_model_cached(MODEL_PATH)
    labels = load_labels(LABELS_PATH)
except Exception as e:
    st.error(f"Error al cargar modelo: {e}")
    st.stop()

# ---- BASE DE DATOS ----
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    source TEXT,
    label TEXT,
    confidence REAL
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT,
    correo TEXT,
    rol TEXT,
    umbral REAL,
    notas TEXT
)
""")
conn.commit()

def insert_prediction(label, confidence, source):
    ts = datetime.now().isoformat()
    c.execute("INSERT INTO predictions (timestamp, source, label, confidence) VALUES (?, ?, ?, ?)", 
              (ts, source, label, confidence))
    conn.commit()

def get_predictions():
    return pd.read_sql("SELECT * FROM predictions ORDER BY timestamp DESC", conn)

def get_persons():
    return pd.read_sql("SELECT * FROM persons ORDER BY id DESC", conn)

# ---- SIDEBAR ----
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio("Ir a:", ["En vivo", "Administración", "Analítica"])

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ==========================================================
# EN VIVO
# ==========================================================
if page == "En vivo":
    st.title("🎥 Clasificación en vivo con Keras + Streamlit")
    st.caption("Reconocimiento facial o de imagen en tiempo real.")

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.latest = {"class": None, "confidence": 0.0}

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            x = resized.astype(np.float32).reshape(1, 224, 224, 3)
            x = (x / 127.5) - 1.0
            pred = model.predict(x, verbose=0)
            idx = int(np.argmax(pred))
            label = labels[idx] if idx < len(labels) else f"Clase {idx}"
            conf = float(pred[0][idx])
            self.latest = {"class": label, "confidence": conf}
            overlay = img.copy()
            text = f"{label} | {conf*100:.1f}%"
            cv2.rectangle(overlay, (5, 5), (5 + 8*len(text), 45), (0, 0, 0), -1)
            cv2.putText(overlay, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            return overlay

    webrtc_ctx = webrtc_streamer(
        key="keras-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=VideoTransformer,
        async_processing=True,
    )

    st.info("Si no se muestra la cámara, usa Chrome y concede permisos.")

    if webrtc_ctx and webrtc_ctx.state.playing:
        result = st.empty()
        while webrtc_ctx.state.playing:
            vt = webrtc_ctx.video_transformer
            if vt and vt.latest["class"]:
                label, conf = vt.latest["class"], vt.latest["confidence"]
                result.markdown(f"**Clase:** `{label}` - **Confianza:** `{conf*100:.2f}%`")
                insert_prediction(label, conf, "camera")
            time.sleep(1)

    st.markdown("---")
    st.subheader("📸 Modo Foto")
    snap = st.camera_input("Captura una imagen")
    if snap is not None:
        img = cv2.imdecode(np.frombuffer(snap.read(), np.uint8), 1)
        resized = cv2.resize(img, (224, 224))
        x = (resized.astype(np.float32).reshape(1, 224, 224, 3) / 127.5) - 1.0
        pred = model.predict(x, verbose=0)
        idx = np.argmax(pred)
        label, conf = labels[idx], float(pred[0][idx])
        st.image(img, caption=f"{label} | {conf*100:.2f}%")
        insert_prediction(label, conf, "photo")
        st.success(f"Predicción: **{label}** ({conf*100:.2f}%)")

# ==========================================================
# ADMINISTRACIÓN
# ==========================================================
elif page == "Administración":
    st.title("👤 Administración de Personas")

    with st.form("form_persona"):
        nombre = st.text_input("Nombre completo")
        correo = st.text_input("Correo electrónico")
        rol = st.selectbox("Rol", ["Estudiante", "Docente", "Invitado", "Otro"])
        umbral = st.slider("Umbral de confianza individual", 0.0, 1.0, 0.8, 0.01)
        notas = st.text_area("Notas adicionales")
        submitted = st.form_submit_button("Guardar")

        if submitted:
            c.execute("INSERT INTO persons (nombre, correo, rol, umbral, notas) VALUES (?, ?, ?, ?, ?)", 
                      (nombre, correo, rol, umbral, notas))
            conn.commit()
            st.success("Persona registrada correctamente ✅")

    st.divider()
    st.subheader("📋 Lista de personas")
    persons_df = get_persons()
    st.dataframe(persons_df)

# ==========================================================
# ANALÍTICA
# ==========================================================
elif page == "Analítica":
    st.title("📊 Panel Analítico de Predicciones")

    df = get_predictions()
    if df.empty:
        st.warning("No hay registros aún.")
        st.stop()

    st.write("### Total de predicciones:", len(df))
    df['confidence'] = df['confidence'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # --- GRÁFICAS ---
    fig1 = px.histogram(df, x="label", title="Frecuencia de clases detectadas")
    fig2 = px.line(df, x="timestamp", y="confidence", title="Nivel de confianza a lo largo del tiempo")
    fig3 = px.box(df, x="label", y="confidence", title="Distribución de confianza por clase")
    if df['source'].dropna().empty:
        fig4 = px.pie(names=["Sin datos"], values=[1], title="Origen de las predicciones")
    else:
        fig4 = px.pie(df, names="source", title="Origen de las predicciones")
    fig5 = px.scatter(df, x="timestamp", y="confidence", color="label", title="Confianza por predicción")

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)
    st.plotly_chart(fig5, use_container_width=True)

    # --- EXPORTACIONES ---
    st.divider()
    st.subheader("⬇️ Exportaciones")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV", csv_bytes, "predicciones.csv", "text/csv")

    # ✅ USO DE KALEIDO PARA PNG
    figuras = [fig1, fig2, fig3, fig4, fig5]
    nombres = ["grafica_1.png", "grafica_2.png", "grafica_3.png", "grafica_4.png", "grafica_5.png"]

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for fig, name in zip(figuras, nombres):
            try:
                # Exportando con kaleido seguro
                img_bytes = fig.to_image(format="png")
                zf.writestr(name, img_bytes)
            except Exception as e:
                st.warning(f"No se pudo generar PNG para {name}: {e}")

    zip_buffer.seek(0)
    st.download_button(
        "📦 Descargar ZIP con gráficas",
        data=zip_buffer,
        file_name="graficas.zip",
        mime="application/zip"
    )
