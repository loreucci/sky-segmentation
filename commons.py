import numpy as np
import cv2


def display_with_mask(img, mask, mask_color=None, label="Masked image"):
    """
    Display an image masking the sky according to a binary mask

    :param img: image
    :param mask: binary mask
    :param mask_color: color of the sky mask
    :param label: label for the display window
    """
    if mask_color is None:
        mask_color = [0, 0, 255]
    sky = np.full(img.shape, mask_color, dtype=np.uint8)
    sky = cv2.bitwise_and(sky, sky, mask=mask)

    ground = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

    cv2.imshow(label, sky + ground)
