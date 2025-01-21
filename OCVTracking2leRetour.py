import cv2
from collections import deque

video_path = "Videos\P1_24h_01top.wmv"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()
# Check la première frame
ret, previous_frame = cap.read()
if not ret:
    print("Erreur : Impossible de lire la première frame.")
    exit()

# Convertir en niveaux de gris
previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# Paramètres du buffer
buffer_size = 5  # Nombre de frames à conserver dans le batch
frame_buffer = deque(maxlen=buffer_size)  # Initialiser le buffer circulaire

while cap.isOpened():
    # Lire la frame suivante
    ret, current_frame = cap.read()
    if not ret:
        break

    # Convertir en niveaux de gris
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Ajouter la frame actuelle au buffer
    frame_buffer.append(current_frame_gray)

    # Si le buffer est plein, comparer les frames
    if len(frame_buffer) == buffer_size:
        # Calculer les différences cumulées entre toutes les frames du buffer
        diff_sum = None
        for i in range(1, buffer_size):
            diff = cv2.absdiff(frame_buffer[i - 1], frame_buffer[i])
            if diff_sum is None:
                diff_sum = diff
            else:
                diff_sum = cv2.bitwise_or(diff_sum, diff)

        # Appliquer un seuillage sur la somme des différences
        _, thresh = cv2.threshold(diff_sum, 12, 255, cv2.THRESH_BINARY)

        # Trouver les contours des zones en mouvement
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dessiner les contours sur la frame originale
        display_frame = current_frame.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 6:  # Filtrer les petits objets
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Afficher la frame avec les mouvements détectés
        cv2.imshow("Tracking des organismes", display_frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
