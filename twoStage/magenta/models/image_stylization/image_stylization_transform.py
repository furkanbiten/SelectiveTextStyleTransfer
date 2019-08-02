# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates a stylized image given an unstylized image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os

import numpy as np
import tensorflow as tf

from magenta.models.image_stylization import image_utils
from magenta.models.image_stylization import model
from magenta.models.image_stylization import ops

import cv2
import time


flags = tf.flags
flags.DEFINE_integer('num_styles', 1,
                     'Number of styles the model was trained on.')
flags.DEFINE_string('checkpoint', None, 'Checkpoint to load the model from')
flags.DEFINE_string('input_image', None, 'Input image file')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_string('output_basename', None, 'Output base name.')
flags.DEFINE_string('which_styles', '[0]',
                    'Which styles to use. This is either a Python list or a '
                    'dictionary. If it is a list then a separate image will be '
                    'generated for each style index in the list. If it is a '
                    'dictionary which maps from style index to weight then a '
                    'single image with the linear combination of style weights '
                    'will be created. [0] is equivalent to {0: 1.0}.')
FLAGS = flags.FLAGS


def _load_checkpoint(sess, checkpoint):
    """Loads a checkpoint file into the session."""
    model_saver = tf.train.Saver(tf.global_variables())
    checkpoint = os.path.expanduser(checkpoint)
    if tf.gfile.IsDirectory(checkpoint):
        checkpoint = tf.train.latest_checkpoint(checkpoint)
        tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))
    model_saver.restore(sess, checkpoint)
    print("Checkpoint loaded: " + str(checkpoint))


def _describe_style(which_styles):
    """Returns a string describing a linear combination of styles."""
    def _format(v):
        formatted = str(int(round(v * 1000.0)))
    while len(formatted) < 3:
        formatted = '0' + formatted
    return formatted

    values = []
    for k in sorted(which_styles.keys()):
        values.append('%s_%s' % (k, _format(which_styles[k])))
    return '_'.join(values)


def _style_mixture(which_styles, num_styles):
    """Returns a 1-D array mapping style indexes to weights."""
    if not isinstance(which_styles, dict):
        raise ValueError('Style mixture must be a dictionary.')
    mixture = np.zeros([num_styles], dtype=np.float32)
    for index in which_styles:
        mixture[index] = which_styles[index]
    return mixture


def _multiple_images(input_image, which_styles, output_dir):
    """Stylizes an image into a set of styles and writes them to disk."""
    with tf.Graph().as_default(), tf.Session() as sess:
        stylized_images = model.transform(
            tf.concat([input_image for _ in range(len(which_styles))], 0),
            normalizer_params={
                'labels': tf.constant(which_styles),
                'num_categories': FLAGS.num_styles,
                'center': True,
                'scale': True})
        _load_checkpoint(sess, FLAGS.checkpoint)

        stylized_images = stylized_images.eval()
        for which, stylized_image in zip(which_styles, stylized_images):
          image_utils.save_np_image(
              stylized_image[None, ...],
              '{}/{}_{}.png'.format(output_dir, FLAGS.output_basename, which))


def _multiple_styles(input_image, which_styles, output_dir):
    """Stylizes image into a linear combination of styles and writes to disk."""
    with tf.Graph().as_default(), tf.Session() as sess:
        mixture = _style_mixture(which_styles, FLAGS.num_styles)
        stylized_images = model.transform(
            input_image,
            normalizer_fn=ops.weighted_instance_norm,
            normalizer_params={
                'weights': tf.constant(mixture),
                'num_categories': FLAGS.num_styles,
                'center': True,
                'scale': True})
        _load_checkpoint(sess, FLAGS.checkpoint)

        stylized_image = stylized_images.eval()
        # image_utils.save_np_image(
        #     stylized_image,
        #     os.path.join(output_dir, '%s_%s.png' % (
        #         FLAGS.output_basename, _describe_style(which_styles))))
        # Raul, changed it since describe_style function was bugged
        image_utils.save_np_image(
            stylized_image,
            os.path.join(output_dir, '%s.png' % (
                FLAGS.output_basename)))

def multiple_input_images(checkpoint, num_styles, input_images_dir, input_images, which_styles):
    """Added by Raul Gombru. Computes style transfer for a list of images"""

    result_images = {}

    with tf.Graph().as_default(), tf.Session() as sess:

        image_path = input_images_dir + input_images[0]
        image = np.expand_dims(image_utils.load_np_image(os.path.expanduser(image_path)), 0)
        stylized_images = model.transform(
            tf.concat([image for _ in range(len(which_styles))], 0),
            normalizer_params={
                'labels': tf.constant(which_styles),
                'num_categories': num_styles,
                'center': True,
                'scale': True}, reuse=tf.AUTO_REUSE)

        _load_checkpoint(sess, checkpoint)
        # stylized_images = stylized_images.eval()
        # for which, stylized_image in zip(which_styles, stylized_images):
        #   image_utils.save_np_image(
        #       stylized_image[None, ...],
        #       '{}/{}_{}.png'.format(output_dir, input_images[0].split('.')[0], which))

        for image_name in input_images:
            image_path = input_images_dir + image_name
            image = np.expand_dims(image_utils.load_np_image(os.path.expanduser(image_path)), 0)
            stylized_images = model.transform(
                tf.concat([image for _ in range(len(which_styles))], 0),
                normalizer_params={
                    'labels': tf.constant(which_styles),
                    'num_categories': num_styles,
                    'center': True,
                    'scale': True}, reuse=tf.AUTO_REUSE)
            stylized_images = stylized_images.eval()
            for which, stylized_image in zip(which_styles, stylized_images):
              # image_utils.save_np_image(
              #     stylized_image[None, ...],
              #     '{}/{}_{}.png'.format(output_dir, image_name.split('.')[0], which))
              result_images[image_name.split('.')[0] + '_' + str(which)] = stylized_image[None, ...]

    return result_images

def style_from_camera(checkpoint, num_styles, which_style, SaveVideo=False):
    """Added by Raul Gombru. Computes style transfer frame by frame"""

    # initialize video camera input
    cap = cv2.VideoCapture(0)

    if (SaveVideo):
        video = cv2.VideoWriter('output.avi', 0, 12.0, (640, 480))

    with tf.Graph().as_default(), tf.Session() as sess:

        frame_2_init = np.expand_dims(np.float32(np.zeros((640, 480, 3))), 0)
        stylized_images = model.transform(
            tf.concat([frame_2_init for _ in range(1)], 0),
            normalizer_params={
                'labels': tf.constant(which_style),
                'num_categories': num_styles,
                'center': True,
                'scale': True}, reuse=tf.AUTO_REUSE)

        _load_checkpoint(sess, checkpoint)

        while (True):
            start_time = time.time()
            # Read frame-by-frame
            ret, frame = cap.read()
            original_frame = frame
            frame = np.expand_dims(np.float32(frame/255.0), 0)
            stylized_images = model.transform(
                tf.concat([frame for _ in range(1)], 0),
                normalizer_params={
                    'labels': tf.constant(which_style),
                    'num_categories': num_styles,
                    'center': True,
                    'scale': True}, reuse=tf.AUTO_REUSE)
            stylized_images = stylized_images.eval()
            for which, stylized_image in zip(which_style, stylized_images):
                out_frame = stylized_image[None, ...]

            out_frame = (out_frame[0,:,:,:]*255).astype('uint8')

            elapsed_time = time.time() - start_time
            print("Running at --> " + str(1 / elapsed_time) + " fps")
            # Show frames
            cv2.namedWindow("input")
            cv2.imshow('input', original_frame)

            cv2.namedWindow("output")
            cv2.imshow('output', out_frame)

            if (SaveVideo):
                video.write(out_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        if (SaveVideo):
            video.release()
        cv2.destroyAllWindows()



def main(unused_argv=None):
    # Load image
    image = np.expand_dims(image_utils.load_np_image(
      os.path.expanduser(FLAGS.input_image)), 0)

    output_dir = os.path.expanduser(FLAGS.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    which_styles = ast.literal_eval(FLAGS.which_styles)
    if isinstance(which_styles, list):
        print("-->  List of styles found, computing several outputs")
        _multiple_images(image, which_styles, output_dir)
    elif isinstance(which_styles, dict):
        print("-->  Dict of styles found, averaging styles")
        _multiple_styles(image, which_styles, output_dir)
    else:
        raise ValueError('--which_styles must be either a list of style indexes '
                         'or a dictionary mapping style indexes to weights.')


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
