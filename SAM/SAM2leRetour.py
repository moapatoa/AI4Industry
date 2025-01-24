import cv2
import torch
from segment_anything import SamPredictor, sam_model_registry
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
predictor = SamPredictor(sam)

# Charger la vidéo
video_path = "D:\\Scolaire\\Travail\\ENSC_3A\\AI4Industry\\Videos\\P1_48h_02mid.wmv"
cap = cv2.VideoCapture(video_path)

# Vérifier si la vidéo s'ouvre correctement
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

# Obtenez les dimensions de la vidéo d'entrée
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Définissez le codec et créez l'objet VideoWriter
output_path = "D:\\Scolaire\\Travail\\ENSC_3A\\AI4Industry\\OutVideos\\P1_48h_02mid.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec pour AVI
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


# Prétraitement des frames précédentes pour détecter les mouvements
prev_frame = None
persistent_masks = []  # Liste pour stocker les masques persistants

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir en niveaux de gris pour détecter les mouvements
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    if prev_frame is None:
        prev_frame = blurred_frame
        continue

    # Détection de mouvement (différence entre frames)
    diff_frame = cv2.absdiff(prev_frame, blurred_frame)
    _, thresh = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)

    # Trouver les contours des zones en mouvement
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points_of_interest = []
    
    for contour in contours:
        # Filtrer les contours par taille
        if 1 < cv2.contourArea(contour) < 100:
            (x, y, w, h) = cv2.boundingRect(contour)
            points_of_interest.append((x + w // 2, y + h // 2))  # Centre du contour

    # Convertir la frame en RGB pour SAM
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb_frame)

    if points_of_interest:
        point_coords = np.array(points_of_interest)
        point_labels = np.ones(len(point_coords))  # Tous les points sont positifs

        # SAM : segmentation guidée par points
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False
        )

        # Mettre à jour les masques persistants
        updated_masks = []
        for mask in masks:
            # Calculer le centre et la taille du nouveau masque
            mask_center = np.mean(np.argwhere(mask), axis=0)
            mask_size = np.sum(mask)  # Taille du masque en pixels actifs

            # Filtrer les masques trop grands
            if mask_size > 500:  # Ajustez ce seuil pour votre cas
                continue

            mask_persistent = False

            # Comparer avec les masques persistants
            for prev_mask, prev_center in persistent_masks:
                distance = np.linalg.norm(mask_center - prev_center)
                if distance < 20:  # Seuil de déplacement
                    updated_masks.append((mask, mask_center))
                    mask_persistent = True
                    break

            # Si le masque est nouveau, l'ajouter
            if not mask_persistent:
                updated_masks.append((mask, mask_center))

        persistent_masks = updated_masks

    # Combiner les masques persistants
    combined_mask = np.zeros_like(gray_frame)
    for mask, _ in persistent_masks:
        combined_mask = np.maximum(combined_mask, mask)

    # Superposer le masque sur la vidéo
    mask_bgr = cv2.cvtColor(combined_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    segmented_frame = cv2.addWeighted(frame, 0.7, mask_bgr, 0.3, 0)

    # Afficher la vidéo en temps réel
    cv2.imshow("Segmentation persistante", segmented_frame)

    # Sauvegarder la vidéo segmentée
    out.write(segmented_frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Mettre à jour la frame précédente
    prev_frame = blurred_frame

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()