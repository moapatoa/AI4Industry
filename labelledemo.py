import streamlit as st
import cv2
import numpy as np

# Fonction principale pour traiter la vidéo
def process_video(video_path, var_threshold, kernel_size, contour_area_threshold):
    # Charger la vidéo
    cap = cv2.VideoCapture(video_path)

    # Vérifier si la vidéo a été chargée correctement
    if not cap.isOpened():
        st.error("Impossible de charger la vidéo.")
        return

    # Création d'un soustracteur de fond
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=var_threshold, detectShadows=True)

    # Créer un conteneur pour afficher la vidéo
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Appliquer le soustracteur de fond
        fgmask = fgbg.apply(frame)

        # Appliquer des opérations morphologiques pour réduire le bruit
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # Trouver les contours dans le masque
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dessiner les rectangles autour des contours détectés
        for contour in contours:
            if cv2.contourArea(contour) > contour_area_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convertir l'image BGR en RGB pour l'affichage dans Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Afficher la vidéo dans l'application
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Pause pour simuler la vitesse réelle de la vidéo
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()

# Interface utilisateur avec Streamlit
st.title("Détection de Contours en Vidéo")

# Charger un fichier vidéo
uploaded_video = st.file_uploader("Choisissez une vidéo", type=["mp4", "wmv", "avi"])

if uploaded_video is not None:
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    # Réglages interactifs
    st.sidebar.header("Paramètres de Détection")
    var_threshold = st.sidebar.slider("Seuil de variation (varThreshold)", min_value=1, max_value=100, value=6)
    kernel_size = st.sidebar.slider("Taille du kernel (morphologie)", min_value=1, max_value=20, value=1)
    contour_area_threshold = st.sidebar.slider("Seuil de surface des contours", min_value=1, max_value=500, value=65)

    # Lancer le traitement vidéo
    process_video("uploaded_video.mp4", var_threshold, kernel_size, contour_area_threshold)
