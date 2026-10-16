from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog

# ============================================================
# SELECCIONAR VIDEO
# ============================================================

root = tk.Tk()
root.withdraw()

ruta_video = filedialog.askopenfilename(
    title="Selecciona un video",
    filetypes=[
        ("Videos", "*.mp4 *.avi *.mov *.mkv"),
        ("Todos los archivos", "*.*")
    ]
)

if not ruta_video:
    print("No se seleccionó ningún video.")
    exit()

print("Video seleccionado:")
print(ruta_video)

# ============================================================
# CARGAR YOLO
# ============================================================

print("\nCargando YOLOv8...")

modelo = YOLO("yolov8n.pt")

print("Modelo cargado.")

# ============================================================
# ABRIR VIDEO
# ============================================================

video = cv2.VideoCapture(ruta_video)

if not video.isOpened():
    print("No se pudo abrir el video.")
    exit()

# ============================================================
# INFORMACIÓN DEL VIDEO
# ============================================================

fps = video.get(cv2.CAP_PROP_FPS)
ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\nInformación del video:")
print("Resolución:", ancho, "x", alto)
print("FPS:", fps)

# ============================================================
# CREAR VIDEO DE SALIDA
# ============================================================

nombre_salida = "video_detectado.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

salida = cv2.VideoWriter(
    nombre_salida,
    fourcc,
    fps,
    (ancho, alto)
)

# ============================================================
# PROCESAR VIDEO
# ============================================================

print("\nProcesando video...")
print("Presiona Q para detener.")

while True:

    ret, frame = video.read()

    if not ret:
        break

    # ========================================================
    # DETECCIÓN YOLO
    # ========================================================

    resultados = modelo(
        frame,
        conf=0.40,
        verbose=False
    )

    # Dibujar detecciones
    frame_resultado = resultados[0].plot()

    # ========================================================
    # GUARDAR FRAME
    # ========================================================

    salida.write(frame_resultado)

    # ========================================================
    # MOSTRAR FRAME
    # ========================================================

    cv2.imshow(
        "YOLO - Deteccion de objetos",
        frame_resultado
    )

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# CERRAR TODO
# ============================================================

video.release()
salida.release()
cv2.destroyAllWindows()

print("\nProcesamiento terminado.")
print("Video guardado como:")
print(nombre_salida)