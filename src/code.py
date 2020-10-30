import os
import cv2
import numpy as np
import glob
from skimage import color
import skimage
import operator
import scipy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from skimage.color import rgb2gray
import matplotlib.patches as patches
import skimage.morphology
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage import data, transform, exposure
from skimage.util import compare_images

import matplotlib.pyplot as plt

from pathlib import Path



def get_patch_ranges():
    """

    Defines different regions where you extract cropped regions
    in the original image for good visualization of the focal stack

    You should visualize at least 4 different patches
     - 3 for the different objects at different depths
     - 1 for the floor

    Returns:
        my_paches(my_patches): 2D array of shape (Num_patches, 4)
    """

    raise NotImplementedError


def calculate_magnification_factor(diopters,focal_length):
    """
    Calculates the magnification factor for each focus distance
    according to the formulas provided in the jupyter notebook

    Args:
        diopters(1d-array): numpy array with the diopter
        focal_length: the focal length of the camera
    Returns:
        magnifications(1d-array): the magnification factor for each focus distance
    """

    raise NotImplementedError


def crop_image(img, bounding):
    """
    Crops an a small image with dimension defined by bounding
    out of the original img exactly in the center !

    HINT:
    1. Think about where is the center of the image
    2. Calculate how far you have to go the left and right to crop in the center

    Args:
        img(np.array): RGB-images that should be cropped
        bounding(tuple): The new size of the image e.g. (128,128)
    Returns:
        out(np.array): The cropped image
    """

    raise NotImplementedError


def scale_image_and_crop(img,magnification):
    """

    Scales the image with the magnification factor and crops it back to the same size

    Args:
        img(np.array): RGB image that needs to be scaled and cropped
        magnfication(float): magnfication factor of how much the image should be scaled
    Returns:
        im_cropped(np.array): The corrected image (scaled and cropped). Should have the simze
        shape as the input img
    """

    raise NotImplementedError




def correct_focal_stack_scaling(imgs,diopters,focal_length):
    """

    Corrects the complet focal stack for magnification issue

    Hint:
     1. call the calculate_magnification_factor in here first
     2. Call the scale_image_and_crop function for each image


    Args:
        imgs(np.array): the focal stack (should be 4-dimensional)
        diopters(1d-array): diopters
        focal_length(float): the focal length
    Returns:
        result(np.array): the corrected focal stack with same dimension as imgs
    """

    raise NotImplementedError




def get_laplacian():
    """

    Returns the 3x3 laplacian matrix

    Returns:
        L(np.array): The 3x3 laplacian matrix

    """

    raise NotImplementedError


def norm_image(img):
    """
    Normalize an image to be between 0 and 1.
    """
    raise NotImplementedError


def filter_laplacian(img):
    """

    Filter the image with a kernel that is the laplacian matrix

    Args:
        img(np.array): Gray-scaled image ==> img.shape = (numX,numY)
    Returns:
        im_filtered(np.array): filtered image with same shape as input
    """

    raise NotImplementedError


def filter_element(img,diameter,element='disk'):
    """
    HINT: For speed reasons you might want to use: scipy.signal.fftconvolve

    To avoid boundary problems we will implement the MIRROR boundary conditions

    In order to do this, we will have to pad the image to be convolved

    For padding you can use np.pad with the option mode='reflect'
    The width of the pad must be the half the diameter of the filter-element

    In order to maintain the correct image size, we need to choose the correct
    mode for the fftconvolve method. Think carefully which on of the 3 it is
    and apply it!

    Args:
        img(np.array): Gray-scaled image ==> img.shape = (numX,numY)
        diameter(float): Diameter of the filter mask
        element(str): structure element that you filter with (either "disk" or "square")
    Returns:
        im_filtered(np.array): filtered image with same shape as input

    """

    raise NotImplementedError


def filter_maximum(img,diameter):
    """
    Applys a maximum filter to the image.

    A maximum filter is local-filter which looks at the environment of a point
    and calculates the maximum in this neighborhood. This value is then assigned to
    the maximum image.

    Look e.g. into ndimage and the maximum_filter method

    Args:
        img(ndarray): The grayscale image where the maximum filter should be apploied
        diameter(int): The diameter of the filter size used for maximum filter

    """

    raise NotImplementedError



def filter_images(img,diameter_element,diameter_max,element='disk'):
    """
    Hint: To the following

    1. Transform the input image (RGB) into a grayscale image first
    2. Apply the laplacian
    3. Apply the structural element (disk or rectangle kernel)
    4. Apply the maximum filter

    Args:
        img(np.array): Gray-scaled image ==> img.shape = (numX,numY)
        diameter(float): Diameter of the filter mask
        element(str): structure element that you filter with (either "disk" or "square")
    Returns:
        im_filtered(np.array): filtered image with same shape as input


    """

    raise NotImplementedError




def compute_depth_map(focus_metric_maps,focus_distances):
    """

    Calculates the depth map from a focus metric
    according to the following formula

    D(x,y) = \argmax_k M(x,y,k)

    Args:
        focus_metric_maps(np.array): the focus-metrics with shape (N_img,Num_x,Num_y)
        focus_distances(1d-array): list of focus_distances in m

    Returns:
        depth_map(np.array): Returns the depth map in m
        depth_map_index(np.array): Returns an image which holds the
        index-position in the focal stack with the most focused point
    """

    raise NotImplementedError


def combine_to_compute_all_in_focus(imgs,depth_map_idx):
    """

    Combines the indexed depth-map and the focalstack images
    into one image where everything is in focus

    HINT: One way to solve this might be:

    1. Make a for-loop through all index-positions
    2. Find all the values in deptH_map_idx that corresponds to
    the current index in the loop
    3. Mask those pixels out an put the color-pixels into the all-in-focus image
    4. After the loop-each pixel must have been assigned and you should see the
    all-in-focus image

    Args:
        imgs(np.array):
        depth_map_index(np.array): Returns an image which holds the

    Returns:
        all_in_focus(np.array): Returns the all-in-focus-image

    """

    raise NotImplementedError
