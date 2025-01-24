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
video_path = "D:\\Scolaire\\Travail\\ENSC_3A\\AI4Industry\\Videos\\P1_24h_01top_r.mp4"
cap = cv2.VideoCapture(video_path)

# Vérifier si la vidéo s'ouvre correctement
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

# Obtenez les dimensions de la vidéo d'entrée
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2  # Si redimensionné
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2  # Si redimensionné
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Définissez le codec et créez l'objet VideoWriter
output_path = "D:\\Scolaire\\Travail\\ENSC_3A\\AI4Industry\\OutVideos\\P1_24h_01top.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec pour AVI
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


# Paramètres
buffer_size = 3  # Nombre de frames à comparer dans le batch
trajectory_length = 6  # Nombre de frames pendant lesquelles la trace persiste
max_disappearance = 10  # Nombre maximum de frames où un objet peut "disparaître"
max_area = 50000  # Aire maximale des masques pour les objets suivis
frame_buffer = deque(maxlen=buffer_size)  # Buffer circulaire pour les frames
objects = {}  # Dictionnaire des objets suivis : ID -> {position, disparition, trajectoire}
next_object_id = 0  # ID unique pour chaque objet détecté

# Fonction pour trouver le centre d'un rectangle
def get_center(x, y, w, h):
    return (x + w // 2, y + h // 2)

# Fonction pour calculer la distance euclidienne
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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
        _, thresh = cv2.threshold(diff_sum, 8, 255, cv2.THRESH_BINARY)

        # Trouver les contours des zones en mouvement
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Générer les masques avec SAM pour la frame actuelle
        input_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(input_frame)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=True,
        )

        # Filtrer les masques trop grands
        valid_masks = []
        for mask in masks:
            if np.sum(mask) < max_area:
                valid_masks.append(mask)

        # Combiner les masques pour créer une région d'exclusion
        exclusion_mask = np.zeros_like(current_frame_gray, dtype=np.uint8)
        for mask in valid_masks:
            exclusion_mask = cv2.bitwise_or(exclusion_mask, mask.astype(np.uint8) * 255)

        # Liste des centres détectés dans la frame actuelle
        detected_centers = []

        for contour in contours:
            if cv2.contourArea(contour) > 2:  # Filtrer les petits objets
                x, y, w, h = cv2.boundingRect(contour)
                center = get_center(x, y, w, h)

                # Vérifier si le centre est dans une zone exclue
                if exclusion_mask[center[1], center[0]] == 0:
                    detected_centers.append(center)

        # Mettre à jour les objets existants
        for obj_id, data in list(objects.items()):
            # Chercher le centre le plus proche
            if detected_centers:
                distances = [euclidean_distance(data["position"], center) for center in detected_centers]
                min_distance = min(distances)
                if min_distance < 60:  # Distance maximale pour associer un objet
                    # Associer l'objet au centre détecté
                    closest_center = detected_centers.pop(distances.index(min_distance))
                    objects[obj_id]["position"] = closest_center
                    objects[obj_id]["disappearance"] = 0
                    objects[obj_id]["trajectory"].append(closest_center)
                    if len(objects[obj_id]["trajectory"]) > trajectory_length:
                        objects[obj_id]["trajectory"].pop(0)
                    continue

            # Si aucun centre détecté proche, incrémenter le compteur de disparition
            objects[obj_id]["disappearance"] += 1
            if objects[obj_id]["disappearance"] > max_disappearance:
                del objects[obj_id]

        # Ajouter de nouveaux objets pour les centres restants
        for center in detected_centers:
            objects[next_object_id] = {
                "position": center,
                "disappearance": 0,
                "trajectory": [center],
            }
            next_object_id += 1

        # Dessiner les trajectoires et les objets
        display_frame = current_frame.copy()
        for obj_id, data in objects.items():
            # Dessiner la trajectoire
            for i in range(1, len(data["trajectory"])):
                cv2.line(display_frame, data["trajectory"][i - 1], data["trajectory"][i], (0, 0, 255), 2)

            # Dessiner la position actuelle
            cv2.circle(display_frame, data["position"], 5, (0, 255, 0), -1)

        # Afficher la frame avec les mouvements détectés
        cv2.imshow("Tracking avec SAM", display_frame)
        out.write(display_frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()