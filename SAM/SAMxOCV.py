import cv2
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
from collections import deque, defaultdict

# Chemin vers le modèle SAM pré-entraîné
sam_checkpoint = "D:\\Scolaire\\Travail\\ENSC_3A\\AI4Industry\\SAM\\models\\sam_vit_b_01ec64.pth"
model_type = "vit_b" # vit_h, vit_l, vit_b

# Charger le modèle SAM
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

# Générateur de masques
mask_generator = SamAutomaticMaskGenerator(sam)

# Charger la vidéo
video_path = "D:\\Scolaire\\Travail\\ENSC_3A\\AI4Industry\\Videos\\P2_24h_01top.wmv"
cap = cv2.VideoCapture(video_path)

# Vérifier si la vidéo s'ouvre correctement
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

# Paramètres pour la détection des mouvements
buffer_size = 12  # Nombre de frames pour le batch
frame_buffer = deque(maxlen=buffer_size)

# Obtenez les dimensions de la vidéo d'entrée
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2  # Si redimensionné
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2  # Si redimensionné
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Définissez le codec et créez l'objet VideoWriter
output_path = "D:\\Scolaire\\Travail\\ENSC_3A\\AI4Industry\\OutVideos\\P2_24h_01top.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec pour AVI
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


while cap.isOpened():
    ret, current_frame = cap.read()
    if not ret:
        break

    # Redimensionner pour accélérer le traitement (facultatif)
    height, width, _ = current_frame.shape
    resized_frame = cv2.resize(current_frame, (width // 2, height // 2))

    # Convertir en niveaux de gris pour la détection des mouvements
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    frame_buffer.append(gray_frame)

    # Si le buffer est plein, détecter les mouvements
    if len(frame_buffer) == buffer_size:
        # Calculer les différences cumulées entre les frames du buffer
        diff_sum = None
        for i in range(1, buffer_size):
            diff = cv2.absdiff(frame_buffer[i - 1], frame_buffer[i])
            if diff_sum is None:
                diff_sum = diff
            else:
                diff_sum = cv2.bitwise_or(diff_sum, diff)

        # Appliquer un seuillage pour isoler les zones de mouvement
        _, motion_mask = cv2.threshold(diff_sum, 12, 255, cv2.THRESH_BINARY)

        # Passer le mouvement détecté dans SAM
        masks = mask_generator.generate(resized_frame)

        # Appliquer les masques aux zones en mouvement
        result_frame = resized_frame.copy()
        for mask in masks:
            # Extraire le masque généré par SAM
            sam_mask = mask['segmentation']
            
            # Filtrer les zones qui ne correspondent pas au mouvement
            sam_mask = cv2.bitwise_and(sam_mask.astype('uint8') * 255, motion_mask)

            # Trouver les contours des zones segmentées par SAM
            contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if 2 < cv2.contourArea(contour) < 50:  # Filtrer les petits objets
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Afficher la frame avec les détections
        out.write(result_frame)
        cv2.imshow("Détection des organismes avec SAM", result_frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()