import face_recognition
import numpy as np
import cv2
import os

# Cargar rostros conocidos
known_encodings = []
known_names = []

for filename in os.listdir("fotos"):
    image_path = os.path.join("fotos", filename)
    name = os.path.splitext(filename)[0]
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(name)

# Captura de la cámara
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Reducir tamaño para acelerar procesamiento
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detección de caras
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Desconocido"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        # Escalar coordenadas al tamaño original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # === DETECCIÓN DE LÁPIZ/LAPICERO por color (ajustable) ===
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Filtrar colores que podrían parecerse a un lápiz azul oscuro o negro
    lower = np.array([10, 80, 80])
    upper = np.array([50, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Encontrar contornos del objeto
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Encontrar contornos del objeto
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]

            if w == 0 or h == 0:
                continue

            aspect_ratio = max(w, h) / min(w, h)

            if aspect_ratio > 3:
                box = cv2.boxPoints(rect)
                box = box.astype(np.intp)


                # Dibujar contorno
                cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)

                # Centro y ángulo
                (x, y), (w, h), angle = rect
                center = (int(x), int(y))
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                # Dibujar orientación
                angle_rad = np.deg2rad(angle)
                length = 50
                dx = int(length * np.cos(angle_rad))
                dy = int(length * np.sin(angle_rad))
                cv2.line(frame, center, (center[0] + dx, center[1] + dy), (0, 0, 255), 2)

                cv2.putText(frame, f"{int(angle)}°", (center[0]+10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Reconocimiento Facial", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
