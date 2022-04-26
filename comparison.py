from sky_mask_cv import create_sky_mask as create_sky_mask_cv
from sky_mask_cnn import create_sky_mask as create_sky_mask_cnn
import numpy as np
import os
import cv2
import argparse
import time


def compute_score(mask, gt):
    """
    Compute the error score of the mask

    :param mask: computed sky mask
    :param gt: ground truth
    :return: percentage of wrong pixels (between 0 and 1)
    """
    return np.sum(np.abs(mask - gt)) / mask.size / 255


def main():
    # args
    parser = argparse.ArgumentParser("comparison")
    parser.add_argument("-d", "--datapath", type=str, default="./data/sky/", help="path to dataset")
    args = parser.parse_args()

    # get images list
    root_dir = args.datapath
    fpath = root_dir + '/data'
    image_list = [f[:4] for f in os.listdir(fpath) if os.path.isfile(os.path.join(fpath, f))][40:]

    scores_cv = []
    scores_cnn = []
    times_cv = []
    times_cnn = []
    for img_id in image_list:

        # open files
        img_path = f"data/{img_id}.jpg"
        gt_path = f"groundtruth/{img_id}_gt.pgm"
        img = cv2.imread(root_dir + img_path)
        gt = cv2.imread(root_dir + gt_path, cv2.IMREAD_GRAYSCALE)
        gt[gt > 0] = 255  # some are 0-255, some 0-1

        # resize both to 256x256
        h, w, _ = img.shape
        img = img[h // 2 - 256 // 2:h // 2 + 256 // 2,
                  w // 2 - 256 // 2:w // 2 + 256 // 2]
        gt = gt[h // 2 - 256 // 2:h // 2 + 256 // 2,
                w // 2 - 256 // 2:w // 2 + 256 // 2]

        # process with the two pipelines and add to score
        start_time = time.time()
        sky_mask_cv = create_sky_mask_cv(img)
        times_cv.append(time.time() - start_time)
        scores_cv.append(compute_score(sky_mask_cv, gt))

        start_time = time.time()
        sky_mask_cnn = create_sky_mask_cnn(img)
        times_cnn.append(time.time() - start_time)
        scores_cnn.append(compute_score(sky_mask_cnn, gt))

    # convert to percentage accuracy
    scores_cv = (1 - np.array(scores_cv)) * 100
    scores_cnn = (1 - np.array(scores_cnn)) * 100

    # print results
    print(f" CV avg accuracy: {np.mean(scores_cv):.3}% (std={np.std(scores_cv):.3})")
    print(f"CNN avg accuracy: {np.mean(scores_cnn):.3}% (std={np.std(scores_cnn):.3})")
    print(f" CV avg execution time: {np.mean(times_cv):.3}s (std={np.std(times_cv):.3})")
    print(f"CNN avg execution time: {np.mean(times_cnn):.3}s (std={np.std(times_cnn):.3})")


if __name__ == '__main__':
    main()