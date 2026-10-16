import cv2
import mediapipe as mp

# ============================================================
# CONFIGURACIÓN DE MEDIAPIPE
# ============================================================

mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils
mp_estilos = mp.solutions.drawing_styles

manos = mp_manos.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# ABRIR CÁMARA INTEGRADA
# ============================================================

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Cámara iniciada.")
print("Presiona Q para salir.")

# ============================================================
# LOOP PRINCIPAL
# ============================================================

while True:

    ret, frame = camara.read()

    if not ret:
        print("No se pudo obtener el frame.")
        break

    # Voltear horizontalmente para que funcione como espejo
    frame = cv2.flip(frame, 1)

    # OpenCV trabaja en BGR
    # MediaPipe trabaja en RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Detectar manos
    resultado = manos.process(frame_rgb)

    # ========================================================
    # DIBUJAR ESQUELETO
    # ========================================================

    if resultado.multi_hand_landmarks:

        for mano in resultado.multi_hand_landmarks:

            # Dibujar los 21 puntos y las conexiones
            mp_dibujo.draw_landmarks(
                frame,
                mano,
                mp_manos.HAND_CONNECTIONS,
                mp_estilos.get_default_hand_landmarks_style(),
                mp_estilos.get_default_hand_connections_style()
            )

            # =================================================
            # MOSTRAR COORDENADAS
            # =================================================

            alto, ancho, _ = frame.shape

            for numero, punto in enumerate(mano.landmark):

                x = int(punto.x * ancho)
                y = int(punto.y * alto)

                # Mostrar número del punto
                cv2.putText(
                    frame,
                    str(numero),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

    # ========================================================
    # MOSTRAR VIDEO
    # ========================================================

    cv2.imshow(
        "Tracker de mano - Esqueleto",
        frame
    )

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# CERRAR
# ============================================================

camara.release()
cv2.destroyAllWindows()
manos.close()

print("Programa terminado.")