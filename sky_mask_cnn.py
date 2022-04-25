import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import argparse
from commons import display_with_mask


def create_sky_mask(img, networkpath="saved_unet.pth"):
    """
    Compute sky mask by using a pretrained Unet CNN

    :param img: image (must be 256x256)
    :param networkpath: path to network weights
    :return: binary sky mask
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = smp.Unet("resnet18", encoder_weights=None, classes=1, activation=None)
    model.to(device)
    model.eval()
    state = torch.load(networkpath, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["params"])

    # convert image to tensor
    img = img.copy()
    img = np.moveaxis(img, -1, 0)

    # normalize
    img = np.array(img, dtype=np.float32) / 255
    for c in range(3):
        img[c] = (img[c] - np.mean(img[c])) / np.std(img[c])
    img = img.reshape(1, *img.shape)  # add initial axis for torch

    # run network
    img_t = torch.from_numpy(img)
    img_t = img_t.to(device)
    out = model(img_t)

    # convert output to binary mask
    mask_np = out.detach().cpu().numpy().reshape((256, 256))
    mask_np[mask_np < 0.5] = 0
    mask_np[mask_np >= 0.5] = 255
    return np.array(mask_np, dtype=np.uint8)


def main():

    # args
    parser = argparse.ArgumentParser("sky_mask_cnn")
    parser.add_argument("image", type=str, help="image to process")
    parser.add_argument("-n", "--network", type=str, default="saved_unet.pth", help="path to saved network")
    args = parser.parse_args()
    filepath = args.image

    # load image and crop to 256x256
    img = cv2.imread(filepath)
    h, w, _ = img.shape
    img = img[h//2 - 256//2:h//2 + 256//2,
              w//2 - 256//2:w//2 + 256//2]

    # compute sky mask
    sky_mask = create_sky_mask(img, networkpath=args.network)

    # display
    cv2.imshow("Original", img)
    display_with_mask(img, sky_mask, label="Masked")
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
