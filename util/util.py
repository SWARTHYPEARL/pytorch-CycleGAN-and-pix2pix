"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os

import pydicom
from typing import Tuple
from torchvision.transforms import functional as F


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def resize_normalize(image):
    image = np.array(image, dtype=np.float64)
    #image -= np.min(image)
    image -= -1024
    #image /= np.max(image)
    image /= 4095
    return image


def open_dicom(path):
    """
    load dicom pixel data and return HU value
    Args:
        path: dicom file path

    Returns: dicom HU pixel array
    """
    image_medical = pydicom.dcmread(path)
    image_data = image_medical.pixel_array

    hu_image = image_data * image_medical.RescaleSlope + image_medical.RescaleIntercept
    hu_image[hu_image < -1024] = -1024
    hu_image[hu_image > 3071] = 3071

    #image_window = window_image(image_hu.copy(), window_level, window_width)

    hu_image = np.expand_dims(hu_image, axis=2)  # (512, 512, 1)
    #image_norm = resize_normalize(hu_image)

    #return image_norm  # use single-channel
    return hu_image  # use single-channel


def tensor2dicom(input_image, original_path, save_path):
    image_numpy = input_image.squeeze(0).cpu().numpy()

    original_dicom = pydicom.dcmread(original_path)
    original_numpy = original_dicom.pixel_array
    #print(original_dicom.file_meta.TransferSyntaxUID)

    #original_dicom.PixelData = np.round((image_numpy + 1) / 2.0 * (np.max(original_numpy * original_dicom.RescaleSlope))).astype(np.uint16).tobytes()
    original_dicom.PixelData = np.round((image_numpy + 1) / 2.0 * 4095).astype(np.uint16).tobytes()
    original_dicom.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    original_dicom.save_as(save_path)


def get_RandomCrop(img: torch.Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Get parameters for ``crop`` for a random crop.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (tuple): Expected output size of the crop.

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    w, h = F._get_image_size(img)
    th, tw = output_size

    if h + 1 < th or w + 1 < tw:
        raise ValueError(
            "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
        )

    if w == tw and h == th:
        return 0, 0, h, w

    i = torch.randint(0, h - th + 1, size=(1, )).item()
    j = torch.randint(0, w - tw + 1, size=(1, )).item()
    return i, j, th, tw
