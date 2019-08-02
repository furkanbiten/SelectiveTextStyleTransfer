import numpy as np
from PIL import Image
import caffe
import os

images_path = 'data/img/'
out_path = 'data/heatmaps/'

resize = False  # Do resize if you have extreme image sizes

# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

# load net
net = caffe.Net('models/TextFCN_deploy.prototxt', 'models/TextFCN.caffemodel', caffe.TEST)

print('Computing heatmaps ...')

if not os.path.isdir(out_path):
    os.mkdir(out_path)

for file in os.listdir(images_path):

    # Load image
    im = Image.open(images_path + file)
    if resize:
        im = im.resize((512,512), Image.ANTIALIAS)

    # Turn grayscale images to 3 channels
    if (im.size.__len__() == 2):
        im_gray = im
        im = Image.new("RGB", im_gray.size)
        im.paste(im_gray)

    # Switch to BGR and substract mean
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    # Shape for input (data blob is N x C x H x W)
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    # Run net
    net.forward()

    # Compute Softmax Heatmap
    hmap_0 = net.blobs['score_conv'].data[0][0, :, :]   # Text score
    hmap_1 = net.blobs['score_conv'].data[0][1, :, :]   # Backgroung score
    hmap_0 = np.exp(hmap_0)
    hmap_1 = np.exp(hmap_1)
    hmap_softmax = hmap_1 / (hmap_0 + hmap_1)

    # Save PNG softmax heatmap
    hmap_softmax_2save = (255.0 * hmap_softmax).astype(np.uint8)
    hmap_softmax_2save = Image.fromarray(hmap_softmax_2save)
    hmap_softmax_2save.save(out_path + file.split('/')[-1].replace('jpg','png'))

print('Done')