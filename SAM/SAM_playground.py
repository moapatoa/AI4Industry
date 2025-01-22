import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

#image = cv2.imread('D:\\Scolaire\\Travail\\ENSC_3A\\AI4Industry\\SAM\\images\\Gandalf.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

video_path = "D:\\Scolaire\\Travail\\ENSC_3A\\AI4Industry\\Videos\\P1_24h_01top.wmv"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()
# Check la première frame
ret, frame = cap.read()
if not ret:
    print("Erreur : Impossible de lire la première frame.")
    exit()

# Convertir en niveaux de gris
image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#plt.figure(figsize=(20,20))
#plt.imshow(image)
#plt.axis('off')
#plt.show()


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "D:\\Scolaire\\Travail\\ENSC_3A\\AI4Industry\\SAM\\models\\sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()