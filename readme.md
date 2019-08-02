# Selective Style Transfer for Text
Accepted to ICDAR 2019 [PDF](https://arxiv.org/abs/1906.01466)

Authors: Raul Gomez, Ali Furkan Biten, Lluis Gomez, Jaume Gibert, Marçal Rusiñol, Dimosthenis Karatzas

![intro](https://gombru.github.io/assets/text_style_transfer/intro.png)

## End-To-End Model

To be released soon.

## Two Stage Model

Requirements:
```text
tensorflow
caffe
magenta (only to train the style transfer model)
```

### Models

Download the models and put them in ``data/models/``.

[Magenta Scene Text Style Transfer Model](https://mega.nz/#!Dc4HBQCJ!FHCFxA84JIGZujMHNLs1NabrJrEokLmCIY_qa4R9XQ4)

[TextFCN Model](https://github.com/gombru/TextFCN)

### Stylizing images

Images are assumed to be in ``data/img/``.

<img src="https://github.com/furkanbiten/SelectiveTextStyleTransfer/blob/master/twoStage/data/img/example.jpg" width="150">

**Style Transfer**

Stylize the entire images using magenta scene text model. To stylize images you don't need a complete magenta installation,
it's enough with the magenta code included in ``magenta/``. (Notice we have modified some code in ``image_stylization_transform.py``,
so a raw magenta won't work).
Results are saved in ``data/styleTransfer/``.
```
python style_images.py
```
<p align="left">
<img src="https://github.com/furkanbiten/SelectiveTextStyleTransfer/blob/master/twoStage/data/styleTransfer/example_1.png" width="150">
<img src="https://github.com/furkanbiten/SelectiveTextStyleTransfer/blob/master/twoStage/data/styleTransfer/example_3.png" width="150">
<img src="https://github.com/furkanbiten/SelectiveTextStyleTransfer/blob/master/twoStage/data/styleTransfer/example_14.png" width="150">
<img src="https://github.com/furkanbiten/SelectiveTextStyleTransfer/blob/master/twoStage/data/styleTransfer/example_23.png" width="150">
</p>

**Text Segmentation**

Get text segmentation heatmaps using TextFCN.
Results are saved in ``data/heatmaps/``.
```
python get_TextFCN_heatmaps.py
```

<img src="https://github.com/furkanbiten/SelectiveTextStyleTransfer/blob/master/twoStage/data/heatmaps/example.png" width="150">

**Selective Text Style Transfer**

Do weighted blending to get the final results of selective style transfer two stage model.
Results are saved in ``data/SelectivestyleTransfer/``.
```
python weighted_blending.py
```
<p align="left">
<img src="https://github.com/furkanbiten/SelectiveTextStyleTransfer/blob/master/twoStage/data/selectiveStyleTransfer/example_1.jpg" width="150">
<img src="https://github.com/furkanbiten/SelectiveTextStyleTransfer/blob/master/twoStage/data/selectiveStyleTransfer/example_3.jpg" width="150">
<img src="https://github.com/furkanbiten/SelectiveTextStyleTransfer/blob/master/twoStage/data/selectiveStyleTransfer/example_14.jpg" width="150">
<img src="https://github.com/furkanbiten/SelectiveTextStyleTransfer/blob/master/twoStage/data/selectiveStyleTransfer/example_23.jpg" width="150">
</p>

### Training

To train the magenta style transfer model follow [the original instructions](https://github.com/tensorflow/magenta/tree/master/magenta/models/image_stylization)
using the source style images found in ``src_styles/``.
