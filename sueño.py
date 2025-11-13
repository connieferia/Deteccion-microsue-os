import cv2
import mediapipe as mp
import numpy as np
from pygame import mixer  # Importar Pygame para el sonido
from datetime import datetime
import time
from datetime import datetime



#para contar parpadeos
parpadeo_detectado = False
contador_cierre = 0
contador_parpadeo = 0
parpadeos_totales = 0
frames_totales = 0
frames_dormido = 0




registro = open("registro_microsuenos.txt", "a")

# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Configuración de EAR
UMBRAL_EAR = 0.25
FRAMES_MICROSUEÑO = 15
contador_cierre = 0
alarma_sonando = False  # Para evitar repetir el sonido

# Inicializar Pygame Mixer
mixer.init()
mixer.music.load('alarma.mp3') 
# Función para calcular EAR
def calcular_ear(ojos):
    A = np.linalg.norm(ojos[1] - ojos[5])
    B = np.linalg.norm(ojos[2] - ojos[4])
    C = np.linalg.norm(ojos[0] - ojos[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Iniciar captura de video
cap = cv2.VideoCapture(0)
inicio = time.time()

ear_prom = 0.0




while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    frames_totales += 1
    duracion = int(time.time() - inicio)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            puntos_ojos_izq = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
            puntos_ojos_der = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]

            ojos_izq = np.array([(int(p.x * w), int(p.y * h)) for p in puntos_ojos_izq])
            ojos_der = np.array([(int(p.x * w), int(p.y * h)) for p in puntos_ojos_der])

            ear_izq = calcular_ear(ojos_izq)
            ear_der = calcular_ear(ojos_der)
            ear_prom = (ear_izq + ear_der) / 2.0

            for punto in ojos_izq:
                cv2.circle(frame, punto, 2, (0, 255, 0), -1)
            for punto in ojos_der:
                cv2.circle(frame, punto, 2, (0, 255, 0), -1)

            if ear_prom < UMBRAL_EAR:
                contador_cierre += 1
                frames_dormido += 1
            else:
                if 1 <= contador_cierre <= 3:
                    parpadeos_totales += 1
                    print("Parpadeo detectado")
                contador_cierre = 0
                alarma_sonando = False

            if contador_cierre >= FRAMES_MICROSUEÑO:
                if not alarma_sonando:
                    mixer.music.play()
                    alarma_sonando = True

                    timestamp = datetime.now().strftime("")
                    filename = f"microsueno_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f" Imagen guardada: {filename}")

                # Dibujo de alerta roja
                cv2.rectangle(frame, (20, 80), (620, 140), (0, 0, 255), -1)
                cv2.putText(frame, "MICROSUENO DETECTADO", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    # HUD / Indicadores
    porcentaje_dormido = (frames_dormido / frames_totales) * 100
    cv2.rectangle(frame, (0, 0), (640, 40), (50, 50, 50), -1)
    cv2.putText(frame, f"Tiempo: {duracion}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Parpadeos: {parpadeos_totales}", (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Dormido: {porcentaje_dormido:.1f}%", (330, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)
    cv2.putText(frame, f"EAR: {ear_prom:.2f}", (520, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow(" Detector de Microsueños", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
registro.close()

print(f"Total de parpadeos: {parpadeos_totales}")


cv2.destroyAllWindows()
