from fast import KCF
import os
from skimage.io import imread
import numpy as np


def correct_bounding_boxes(X, Y, w, h, max_patch_size=100*100, resize=False):
    X_, Y_, w_, h_ = np.array(X), np.array(Y), w, h
    if resize and w * h > max_patch_size:
        X_ *= 0.5
        Y_ *= 0.5
        w_ *= 0.5
        h_ *= 0.5

    X_ -= w_ / 2.
    Y_ -= h_ / 2.
    w_ *= 2
    h_ *= 2

    return list(X_), list(Y_), w_, h_

if __name__ == '__main__':
    PATH = '/Users/phanquochuy/Projects/rotoscopy/data/ROTO'
    # PATH = '/home/phan/Projects/rotoscopy/data/ROTO'
    ROTO_PATH = os.path.join(PATH, 'processed')
    IMAGE_PATH = os.path.join(PATH, 'jpegs')

    SHOT = 'SH0080'

    resize_image = False
    max_patch_size = 100*100

    image = imread(os.path.join(IMAGE_PATH, SHOT, 'MODE_' + SHOT + '.1001.jpg')) # .astype(np.float32)
    kcf = KCF.KCFTracker(5.2, resize=resize_image, compress_feature=True, compressed_size=4, detect_thresh=0.2, max_patch_size=max_patch_size)

    x, y, w, h = 677., 37., 200., 100.
    X, Y = [x, x+50], [y, y + 50]

    kcf.init(image, X, Y, w, h)

    for i in range(2, 9):
        next_image = imread(os.path.join(IMAGE_PATH, SHOT, 'MODE_' + SHOT + '.100%d.jpg' % i))
        # X_, Y_, w_, h_ = correct_bounding_boxes(X, Y, w, h, max_patch_size=max_patch_size, resize=resize_image)
        # roi = kcf.update(next_image, X_, Y_, w_, h_)
        roi = kcf.update(next_image, X, Y, w, h)

        print(roi)