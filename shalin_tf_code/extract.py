from __future__ import print_function

import argparse
import logging
import os
import subprocess
import sys
import time
from os.path import join as oj

import h5py
import numpy as np
import tensorflow as tf

sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path


def get_args():
    parser = argparse.ArgumentParser(description='Run feature extraction')
    parser.add_argument('--device', type=str, default='/cpu:0',
                        help='an integer for the accumulator')
    return parser.parse_args()


# params
args = get_args()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
np.random.seed(13)
num_ims_per_clip = 16
device = args.device  # '/cpu:0', '/gpu:0'
ims_dir = os.path.join('data', 'stim')
out_file = oj(ims_dir, 'features.h5')
if os.path.exists(out_file):  # delete the features if they already exist
    subprocess.call('rm ' + out_file, shell=True)

with h5py.File(oj(ims_dir, 'ims_small.h5')) as f:
    ims = f['ims']
    N, H, W, C = ims.shape[0], ims.shape[1], ims.shape[2], ims.shape[3]
    print('ims.shape', ims.shape)
    from models.c3d.c3d_model import build_model

    # loop over clips here
    t0 = time.clock()
    frame_num = 0
    clip = ims[frame_num: frame_num + num_ims_per_clip]
    clip = np.reshape(clip, (1, num_ims_per_clip, H, W, C))
    placeholder, layer_names, layer_graphs = build_model(clip.shape)

    for frame_num in range(N - num_ims_per_clip):
        print('frame_num', frame_num, str(time.clock() - t0))
        clip = ims[frame_num: frame_num + num_ims_per_clip]
        clip = np.reshape(clip, (1, num_ims_per_clip, H, W, C))

        # extract the features
        with tf.device(device):
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            t = time.time()
            layer_features = sess.run(layer_graphs, feed_dict={placeholder: clip})

        # save the features
        if frame_num == 0:  # create datasets
            with h5py.File(out_file) as out:
                for l in range(len(layer_names)):
                    feature_dim = layer_features[l].shape
                    out.create_dataset(layer_names[l], (N - (16 - 1), layer_features[l].shape[-4],
                                                        layer_features[l].shape[-3], layer_features[l].shape[-2],
                                                        layer_features[l].shape[-1]), dtype='float32')
        with h5py.File(out_file) as out:
            for l in range(len(layer_names)):
                # print(l, layer_names[l], N, layer_features[l].shape)
                out[layer_names[l]][frame_num] = layer_features[l][0]  # batch size of 1
