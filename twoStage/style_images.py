from magenta.models.image_stylization import image_stylization_transform
from PIL import Image
import os

images_path = 'data/img/'
out_path = 'data/styleTransfer/'
checkpoint = "models/magenta_scene_text/"
num_styles = 34

if not os.path.isdir(out_path):
    os.mkdir(out_path)

which_styles = []
for i in range(num_styles): which_styles.append(i)

input_images = []
for file in os.listdir(images_path): input_images.append(file.split('/')[-1])


print("Styling images ...")
result_images = image_stylization_transform.multiple_input_images(checkpoint, num_styles, images_path, input_images, which_styles)

print("Saving results")
for k, v in result_images.iteritems():
    v = v[0,:,:,:]
    pil_image = Image.fromarray((v*255).astype('uint8'))
    pil_image.save(out_path + k + '.png')

print("Done")


