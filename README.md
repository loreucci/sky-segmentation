## Usage

The two models can be launched with

`python sky_mask_cv.py [image]`

`python sky_mask_cnn.py [image]`

They will both display a visualization of the sky map computed.

For more options please see `python sky_mask_[cv|cnn].py --help`

### CNN training

To train the CNN first download the [dataset](https://www.ime.usp.br/~eduardob/datasets/sky/) and then launch

`python train_cnn.py -d /path/to/data`