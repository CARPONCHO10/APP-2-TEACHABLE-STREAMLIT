from tensorflow.keras.models import load_model  # Keras dentro de TensorFlow
import cv2
import numpy as np

np.set_printoptions(suppress=True)

# Cargar modelo y etiquetas
MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"

model = load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# 0 suele ser la webcam integrada; cambia a 1 si usas una USB adicional
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW ayuda en Windows

if not camera.isOpened():
    raise RuntimeError("No se pudo abrir la cámara. Prueba con otro índice (0/1) o revisa permisos.")

try:
    while True:
        ret, image = camera.read()
        if not ret:
            print("No llega imagen de la cámara.")
            break

        # Redimensionar a 224x224 (lo que usa Teachable Machine por defecto)
        resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Mostrar la imagen original
        cv2.imshow("Webcam Image", image)

        # Preparar tensor
        x = np.asarray(resized, dtype=np.float32).reshape(1, 224, 224, 3)
        x = (x / 127.5) - 1.0

        # Predicción
        prediction = model.predict(x, verbose=0)
        index = int(np.argmax(prediction))
        class_name = class_names[index] if index < len(class_names) else f"Clase {index}"
        confidence_score = float(prediction[0][index])

        # Imprimir resultados
        print(f"Class: {class_name} | Confidence: {confidence_score * 100:.2f}%")

        # Esc para salir
        if cv2.waitKey(1) == 27:
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
