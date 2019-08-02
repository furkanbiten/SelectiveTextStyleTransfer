import os
from PIL import Image
import numpy as np
import cv2

images_path = 'data/img/'
heatmaps_path = 'data/heatmaps/'
styled_images_path = 'data/styleTransfer/'
out_path = 'data/selectiveStyleTransfer/'
num_styles = 34
text_prob_threshold = 200  # Don't stylize pixels below this text probability (range is [0-255])

if not os.path.isdir(out_path):
    os.mkdir(out_path)

print("Doing weighted blending ...")

for file in os.listdir(images_path):
    img = np.array(Image.open(images_path + file))
    heatmap = np.array(Image.open(heatmaps_path + file.split('/')[-1].replace('jpg','png')))
    h, w, c = img.shape
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap[heatmap < text_prob_threshold] = 0  # Ignore pixels below this text prob.
    heatmap = heatmap / 255.0
    inv_heatmap = 1.0 - heatmap

    # For each style
    for s in range(34):
        try:
            styled_img = np.array(Image.open(styled_images_path + file.split('/')[-1].split('.')[0] + '_' + str(s) + '.png'))

            styled_img = cv2.resize(styled_img, (w, h))
            img_edited = img.copy()
            # For each RGB channel
            for c in range(3):
                img_edited[:,:,c] = img_edited[:,:,c] * inv_heatmap  # Info kept form original image
                styled_img[:,:,c] = styled_img[:,:,c] * heatmap  # Info used form styled image

            out_img = Image.fromarray(styled_img + img_edited)
            out_img.save(out_path + file.split('/')[-1].split('.')[0] + '_' + str(s) + '.jpg')
        except:
            continue

print("Done")

