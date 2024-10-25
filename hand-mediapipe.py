import cv2
import mediapipe as mp

# Inicialización de MediaPipe y OpenCV
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Configuración de la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Inicialización de la detección de manos
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    # Contador de toques
    touch_count = 0
    # Coordenadas del círculo
    circle_radius = 70
    circle_center = (150, 100)  # Esquina superior derecha

    # Coordenadas del círculo
    circle_radius2 = 70
    circle_center2 = (500, 100)  # Esquina superior derecha


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Voltear el marco horizontalmente
        frame = cv2.flip(frame, 1)
        
        # Convertir el marco a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar el marco y detectar manos
        results = hands.process(frame_rgb)

        # Dibujar el círculo
        cv2.circle(frame, circle_center, circle_radius, (0, 0, 255), -1)  # Círculo rojo
        cv2.putText(frame, f'Tocar: {touch_count}', (circle_center[0] - 30, circle_center[1] + 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Dibujar el círculo2
        cv2.circle(frame, circle_center2, circle_radius2, (0, 0, 255), -1)  # Círculo rojo
        cv2.putText(frame, f'Resetear: {touch_count}', (circle_center2[0] - 30, circle_center2[1] + 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Verificar las manos detectadas
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                # Obtener la posición del dedo índice
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_x = int(index_finger_tip.x * frame.shape[1])
                index_y = int(index_finger_tip.y * frame.shape[0])

                # Dibujar los puntos de referencia de las manos
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Comprobar si el índice toca el círculo
                if (index_x - circle_center[0])**2 + (index_y - circle_center[1])**2 <= circle_radius**2:
                    touch_count += 1
                    cv2.putText(frame, "Tocado!", (circle_center[0] - 30, circle_center[1] + 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                # Comprobar si el índice toca el círculo2
                if (index_x - circle_center2[0])**2 + (index_y - circle_center2[1])**2 <= circle_radius**2:
                    touch_count = 0
                    cv2.putText(frame, "Recetear!", (circle_center2[0] - 30, circle_center2[1] + 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Mostrar el marco
        cv2.imshow('Frame', frame)
        
        # Salir si se presiona la tecla 'Esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()