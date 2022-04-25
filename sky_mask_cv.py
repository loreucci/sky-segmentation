import cv2
import numpy as np
import argparse
from commons import display_with_mask


def calculate_border(t, gradient):
    """
    Calculate border in gradient image, given a certain threshold

    :param t: threshold
    :param gradient: gradient image
    :return: border
    """
    h, w = gradient.shape
    b = h * np.ones(w, dtype=int)
    for x in range(w):
        y = np.argmax(gradient[:, x] > t)
        if y > 0:
            b[x] = y
    return b


def get_mask(img, b):
    """
    Compute binary mask given a border

    :param img: image
    :param b: border
    :return: binary mask
    """
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y in enumerate(b):
        mask[:y, x] = 255

    return mask


def get_sky_ground_pixels(img, b):
    """
    Divide image in sky and ground regions, given a border

    Regions have the same size as the original image.

    :param img: image
    :param b: border
    :return: (sky, ground) regions
    """
    _, _, c = img.shape
    mask = get_mask(img, b)
    ground = np.ma.array(img, mask=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)).compressed().reshape(-1, c)
    sky = np.ma.array(img, mask=cv2.cvtColor(cv2.bitwise_not(mask), cv2.COLOR_GRAY2BGR)).compressed().reshape(-1, c)
    return sky, ground


def energy(img, b):
    """
    Compute inverse of energy of the image, given a border

    Energy is J = 1 / (gamma * |sigma_sky| + |sigma_ground| + gamma * |lamda_sky_1| + |lamda_ground_1|

    :param img: image
    :param b: border
    :return: inverse of energy
    """
    # compute regions
    sky, ground = get_sky_ground_pixels(img, b)

    # covariance matrices
    sigma_g, _ = cv2.calcCovarMatrix(ground, None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
    sigma_s, _ = cv2.calcCovarMatrix(sky, None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)

    # eigenvalues
    lambda_g = max(np.linalg.eig(sigma_g)[0])
    lambda_s = max(np.linalg.eig(sigma_s)[0])

    return 2 * np.linalg.det(sigma_s) + np.linalg.det(sigma_g) + 2 * np.abs(lambda_s) + np.abs(lambda_g)


def energy_optimization(img, gradient, thresh_min=5, thresh_max=600, n=120):
    """
    Perform energy optimization search, finding the optimal border

    Searches for the theshold value that maximise the energy.

    :param img: image
    :param gradient: gradient image
    :param thresh_min: minimum value of the threshold
    :param thresh_max: maximum value of the threshold
    :param n: number of thresholds to test
    :return: optimal border
    """
    j_max = np.inf
    b_opt = None
    for k in range(n):
        t = thresh_min + (thresh_max - thresh_min) / (n-1) * k
        b = calculate_border(t, gradient)
        j = energy(img, b)
        if j < j_max:
            j_max = j
            b_opt = b
    return b_opt


def check_no_sky(img, b, thresh1=None, thresh2=None, thresh3=5):
    """
    Heuristics to check if there is actually a sky

    :param img: image
    :param b: border
    :param thresh1: border average threshold 1
    :param thresh2: border average threshold 2
    :param thresh3: ASADSBP threshold
    :return: true if the there is no sky, false otherwise
    """
    h, w, _ = img.shape

    if thresh1 is None:
        thresh1 = h / 30
    if thresh2 is None:
        thresh2 = h / 5

    # border position function
    border_ave = np.sum(b) / w

    # ASADSBP
    asadsbp = np.sum(np.abs(np.diff(b))) / (w-1)

    return border_ave < thresh1 or (border_ave < thresh2 and asadsbp > thresh3)


def check_fake_sky(img, b, thresh4=None):
    """
    Heuristics to check if the border really separates the sky from the ground

    :param img: image
    :param b: border
    :param thresh4: heuristics threshold
    :return: true if the border is not the real sky border, false otherwise
    """
    h, w, _ = img.shape

    if thresh4 is None:
        thresh4 = h / 3

    return np.any(np.abs(np.diff(b)) > thresh4)


def fix_fake_sky(img, b):
    """
    Try to fix a border that does not separate the sky from the ground

    :param img: image
    :param b: border
    :return: adjusted border
    """
    # get regions
    sky, ground = get_sky_ground_pixels(img, b)

    # cluster sky with K-Means and get the two regions
    _, best_labels, _ = cv2.kmeans(np.float32(sky), 2, None,
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1), 20,
                                   cv2.KMEANS_RANDOM_CENTERS)

    # compute covariances and means
    sigma_s1, mu_s1 = cv2.calcCovarMatrix(sky[best_labels.ravel() == 0], None,
                                          cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
    _, sigma_inv_s1 = cv2.invert(sigma_s1, cv2.DECOMP_LU)
    sigma_s2, mu_s2 = cv2.calcCovarMatrix(sky[best_labels.ravel() == 1], None,
                                          cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
    _, sigma_inv_s2 = cv2.invert(sigma_s2, cv2.DECOMP_LU)
    sigma_g, mu_g = cv2.calcCovarMatrix(ground, None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
    _, sigma_inv_g = cv2.invert(sigma_g, cv2.DECOMP_LU)

    # select the cluster more distant from the ground as sky
    if cv2.Mahalanobis(mu_s1, mu_g, sigma_inv_s1) > cv2.Mahalanobis(mu_s2, mu_g, sigma_inv_s2):
        mu_s = mu_s1
        sigma_inv_s = sigma_inv_s1
    else:
        mu_s = mu_s2
        sigma_inv_s = sigma_inv_s2

    # recalculate border
    _, w, c = img.shape
    for x in range(w):
        cnt = 0
        for y in range(b[x]):
            pixel = np.array(img[y, x].reshape(1, c), dtype=float)
            dist_s = cv2.Mahalanobis(pixel, mu_s, sigma_inv_s)
            dist_g = cv2.Mahalanobis(pixel, mu_g, sigma_inv_g)
            if dist_s < dist_g:
                cnt += 1
        if cnt < b[x] / 2:
            b[x] = 0

    return b


def create_sky_mask(img):
    """
    Compute sky mask by using the algorithm from (Shen and Wang, 2013)

    :param img: image
    :return: binary sky mask or None if there is no sky
    """
    # Image Pre-processing and Gradient Image Calculation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gradient = np.hypot(cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    # optimize sky border
    b_opt = energy_optimization(img, gradient, thresh_max=np.max(gradient) - 5)
    mask = get_mask(img, b_opt)

    # check no sky condition
    if check_no_sky(img, b_opt):
        print("No sky detected")
        return None

    # check for fake sky
    if check_fake_sky(img, b_opt):
        print("Partially occluded sky detected, correcting")
        b_new = fix_fake_sky(img, b_opt)
        mask = get_mask(img, b_new)

    return mask


def main():

    # args
    parser = argparse.ArgumentParser("sky_mask_cv")
    parser.add_argument("image", type=str, help="image to process")
    args = parser.parse_args()
    filepath = args.image

    # open image
    img = cv2.imread(filepath)

    # compute sky mask
    sky_mask = create_sky_mask(img)

    # display
    cv2.imshow("Original", img)
    if sky_mask is not None:
        display_with_mask(img, sky_mask, label="Masked")
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
