import os
import glob
import argparse
import scipy.io
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():

    parser = argparse.ArgumentParser(description='Converts SBD dataset label(*.mat to *.png)')
    parser.add_argument('--root', help='root directory of SBD dataset (contains cls and inst directory')
    args = parser.parse_args()
    
    return args


def make_palette(num_classes):

    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    labels = np.arange(num_classes, dtype=np.uint8)

    for i in range(8):
        labels, color = np.divmod(labels, 2)
        palette[:, i%3] += 2**(7-i//3) * color

    return palette


if __name__ == "__main__":
    
    args = parse_args()
    palette = make_palette(256).reshape(-1)

    root_dir = args.root

    for kind in ('cls', 'inst'):

        img_paths = glob.glob('{}/{}/*.mat'.format(root_dir, kind))

        for img_path in tqdm(img_paths):

            img_name = os.path.basename(img_path).split('.')[0]

            mat = scipy.io.loadmat(img_path)
            arr = mat['GT{}'.format(kind)][0, 0]['Segmentation']

            img = Image.fromarray(arr)
            img.putpalette(palette)
            img.save(img_path.replace('.mat', '.png'))