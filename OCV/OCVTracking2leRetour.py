import cv2
import numpy as np
from collections import deque
import time


def detect_contour_with_bubbles(image, threshold=12, max_iterations=1000, visualize=True):
    """
    Détecte les contours en simulant des bulles qui gonflent jusqu'à rencontrer des obstacles.
    Affiche visuellement la croissance des bulles si `visualize` est True.
    
    Arguments :
    - image : Image d'entrée en niveaux de gris (numpy array 2D).
    - threshold : Seuil pour détecter les changements de couleur (matière).
    - max_iterations : Nombre maximal d'itérations pour la croissance des bulles.
    - visualize : Booléen indiquant si la croissance des bulles doit être affichée.
    
    Retour :
    - mask : Masque binaire avec les contours détectés.
    """
    # Dimensions de l'image
    h, w = image.shape
    
    # Masque pour stocker les contours
    mask = np.zeros((h, w), dtype=np.uint8)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Image de visualisation (pour dessiner les bulles en couleur)
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Liste des bulles (initialisées au centre de l'image ou sur une grille)
    bubbles = [(430, 300), (430, 900)]  # Une seule bulle initiale au centre
    visited = set(bubbles)       # Ensemble pour éviter de revisiter les mêmes pixels
    
    # Directions pour expansion des bulles (haut, bas, gauche, droite, et diagonales)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for iteration in range(max_iterations):
        new_bubbles = []
        
        # Faire croître chaque bulle
        for y, x in bubbles:
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                
                # Vérifier les limites de l'image
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                
                # Vérifier si ce pixel a déjà été visité
                if (ny, nx) in visited:
                    continue
                
                # Calculer la différence de couleur (gradient local)
                diff = abs(int(image[ny, nx]) - int(image[y, x]))
                
                # Si la différence est au-dessus du seuil, on considère qu'on a touché un obstacle
                if diff > threshold:
                    mask[ny, nx] = 255  # Marquer ce point comme un contour
                    full_mask[ny, nx] = 255
                    vis_image[ny, nx] = (0, 0, 255)  # Dessiner en rouge
                else:
                    # Ajouter ce pixel comme nouvelle position de la bulle
                    new_bubbles.append((ny, nx))
                    full_mask[ny, nx] = 255
                    vis_image[ny, nx] = (0, 255, 0)  # Dessiner en vert
                
                visited.add((ny, nx))
        
        # Mettre à jour la liste des bulles
        bubbles = new_bubbles
        
        # Visualiser la croissance des bulles
        if visualize:
            cv2.imshow("Croissance des bulles", vis_image)
            cv2.waitKey(1)  # Petite pause pour rendre l'animation visible
        
        # Arrêter si plus aucune bulle ne peut croître
        if not bubbles:
            break

    if visualize:
        # Attendre avant de fermer
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return full_mask

def HOUGH(image):
    circles = cv2.HoughCircles(
        image,                 # Image d'entrée (les contours détectés)
        cv2.HOUGH_GRADIENT,           # Méthode
        dp=1,                         # Résolution de l'accumulateur inversement proportionnelle
        minDist=300,                   # Distance minimale entre les centres des cercles
        param1=50,                    # Gradient pour le canny edge
        param2=30,                    # Seuil pour le centre du cercle
        minRadius=100,                  # Rayon minimal du cercle
        maxRadius=1000                  # Rayon maximal du cercle
        )
    # Vérifier si des cercles ont été détectés
    if circles is not None:
        circles = np.uint16(np.around(circles))  # Arrondir les valeurs des cercles
        for i in circles[0, :]:
            # Dessiner le cercle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Cercle en vert
            # Dessiner le centre
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)     # Centre en rouge

    # Afficher l'image avec les cercles détectés
    cv2.imshow("Cercles détectés", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

# Appliquer la détection de contour avec bulles
contour_mask = detect_contour_with_bubbles(previous_frame, threshold=1, visualize=False)

cv2.imshow("Masque entier", contour_mask)

HOUGH(contour_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()


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
