import torch
import cv2
import numpy as np
import os
import argparse
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


class SkyDataset(Dataset):
    """
    Dataset of sky images
    """

    def __init__(self, root_dir, start=0, end=60, transform=None):
        """
        Create a dataset of sky images

        :param root_dir: path to dataset
        :param start: index of first image to use
        :param end: index of last image to use
        :param transform: transformation to apply to images
        """
        self._root_dir = root_dir
        self._transform = transform

        fpath = root_dir + '/data'
        self._image_list = [f[:4] for f in os.listdir(fpath) if os.path.isfile(os.path.join(fpath, f))][start:end]

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, idx):

        img = self._image_list[idx]
        f_path = f"data/{img}.jpg"
        gt_path = f"groundtruth/{img}_gt.pgm"

        # input
        f = cv2.imread(self._root_dir + f_path)
        f = np.moveaxis(f, -1, 0)
        # normalize
        f = np.array(f, dtype=np.float32) / 255
        for c in range(3):
            f[c] = (f[c] - np.mean(f[c])) / np.std(f[c])

        # ground truth
        gt = cv2.imread(self._root_dir + gt_path, cv2.IMREAD_GRAYSCALE)
        gt[gt > 0] = 1  # some are 0-255, some 0-1
        gt = gt.reshape((1, gt.shape[0], gt.shape[1]))
        gt = np.array(gt, dtype=np.float32)

        if self._transform:
            f, gt = self._transform((f, gt))

        return torch.from_numpy(f), torch.from_numpy(gt)


class RandomCrop(object):
    """
    Random crop transformation
    """

    def __init__(self, output_size, skip_y=20):
        """
        Create a random crop transformation

        :param output_size: size of the output image (tuple or int)
        :param skip_y: exclude pixels from the tpo and bottom of the image
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self._output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self._output_size = output_size
        self._skip_y = skip_y

    def __call__(self, sample):
        f, gt = sample

        _, h, w = f.shape
        new_h, new_w = self._output_size

        top = np.random.randint(self._skip_y, h - new_h - self._skip_y)
        left = np.random.randint(0, w - new_w)

        f = f[:,
              top: top + new_h,
              left: left + new_w]

        gt = gt[:,
                top: top + new_h,
                left: left + new_w]

        return f, gt


class CenterCrop(object):
    """
    Random crop transformation
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self._output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self._output_size = output_size

    def __call__(self, sample):
        f, gt = sample

        _, h, w = f.shape
        new_h, new_w = self._output_size

        top = h // 2 - new_h // 2
        left = w // 2 - new_w // 2

        f = f[:,
              top: top + new_h,
              left: left + new_w]

        gt = gt[:,
                top: top + new_h,
                left: left + new_w]

        return f, gt


def run_tests(model, test_loader, criterion, device):
    model.eval()
    loss = 0.0
    for f, gt in test_loader:
        f = f.to(device)
        gt = gt.to(device)
        out = model(f)
        batch_loss = criterion(out, gt)
        loss += batch_loss.item()
    return loss


def train_network(data_root_dir="./data/sky/",
                  epochs=100,
                  save_to=None):

    train_size = 40
    batch_size = 16
    train_set = SkyDataset(root_dir=data_root_dir, transform=RandomCrop(256), end=train_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_set = SkyDataset(root_dir=data_root_dir, transform=CenterCrop(256), start=train_size)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

    # load network
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = smp.Unet("resnet18", encoder_weights="imagenet", classes=1, activation=None)
    model = model.to(device)

    # loss and training alg
    lr = 5e-4
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training
    for e in range(epochs):
        model.train()
        optimizer.zero_grad()
        tr_loss = 0.0
        for f, gt in train_loader:
            f = f.to(device)
            gt = gt.to(device)
            out = model(f)
            loss = criterion(out, gt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tr_loss += loss.item()
        tr_loss /= len(train_set)
        te_loss = run_tests(model, test_loader, criterion, device) / len(test_set)
        print(f"epoch {e}: tr_loss={tr_loss:.3}, te_loss={te_loss:.3}")

    if save_to is not None:
        state = {
            "params": model.state_dict()
        }
        torch.save(state, save_to)


def main():
    # args
    parser = argparse.ArgumentParser("train_cnn")
    parser.add_argument("-d", "--datapath", type=str, default="./data/sky/", help="path to dataset")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-s", "--saveto", type=str, default=None, help="save network to path")
    args = parser.parse_args()

    train_network(data_root_dir=args.datapath,
                  epochs=args.epochs,
                  save_to=args.saveto)


if __name__ == '__main__':
    main()
