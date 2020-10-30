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


savedir = 'output'

def save_fig_as_png(figtitle):
    '''
    Saves the current figure into the output folder
    The figtitle should not contain the ".png".
    This helper function shoudl be easy to use and should help you create the figures 
    needed for the report
    
    The directory where images are saved are taken from savedir in "Code.py" 
    and should be included in this function.
    
    Hint: The plt.gcf() might come in handy
    Hint 2: read about this to crop white borders
    https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content
    
    '''
    fig = plt.gcf()
    path ='.//' + savedir + '//' + figtitle + str('.png')
    print(path)

    fig.savefig(path,bbox_inches='tight', pad_inches=0)


def plot_with_colorbar(img,vmax=0):
    """
    args:
        vmax: The maximal value to be plotted
    """
    ax = plt.gca()
    if(vmax == 0):
        im = ax.imshow(img, cmap= 'gray')
    else:
        im = ax.imshow(img, cmap= 'gray',vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    

def load_focal_stack_data(path,scale = 0.5):
    """
    
    Loads the imagesa and focal values defined by the path variable.
    
    Note, images should be:
        1. Have float value
        2. Be normalized between 0 and 1.
        3. You might need to flip some images because they look weird depending
        on the image loading method your are using
        4. The filenames might not be sorted. Make sure to sort your values
        according to the filename
        
    HINT: Sorting filenames
        1. You can use np.argsort
        2. You can use array indexing. E.g. if X are you images, you can simply do: X[I]
        if I is the permutation matrix from the argsort function
    
    Args:
        path(string): Path to the folder
        scale(float): scaling factor for the image when loaded
    
    Returns:
    """
    #percent by which the image is resized

    files = glob.glob(path + '*.jpg')

    imgs = []
    vals = []
    for k in range(len(files)):

        #print(k)
        f = files[k]
        
        filename = Path(f).stem
        val = float(filename.split('_')[2])
        vals.append(val)
        
        img = cv2.imread(f)
        img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        #calculate the 50 percent of original dimensions
        width = int(img.shape[1] * scale )
        height = int(img.shape[0] * scale )

        # dsize
        dsize = (width, height)

        # resize image
        img = cv2.resize(img, dsize)

        img = np.fliplr(img)

        imgs.append(img)
        
    imgs = np.array(imgs)
    vals = np.array(vals)
    
    # Maybe imgs and vals are not sorted yet
    # We need to sort them
    I = np.argsort(vals)
    
    imgs = imgs[I]
    vals = vals[I]
    
    return imgs,vals


def display_patches(img,my_patches):
    """
    Visualizes the regions where the you crop from the image from the image
    
    Args:
        img(np.array): RGB image that is plotted in the background
        my_patches(np.arra)
    """
    plt.imshow(img)
    ax = plt.gca()
    # Create a Rectangle patch
    for k in range(my_patches.shape[1]):
        xmin = my_patches[k,0]
        ymin = my_patches[k,2]
        xmax = my_patches[k,1]
        ymax = my_patches[k,3]
        rect = patches.Rectangle((ymin,xmin),ymax-ymin,xmax-xmin,linewidth=3,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    
def plot_images_grid(imgs,focus_distances,m_plots=3,n_plots=3,x0=None,x1=None,y0=None,y1=None):
    """
    
    Plots the images in a grid.
    
    HINT: Make sure that parameters actually work!
    
    Args:
        imgs(np.array): Focal stack images
        focus_distances(1d-array): contians the focus distances in meter
        m_plots,n_plots(int): number of subfigures you want to have in this function
        x0,x1,y0,y1: The crop regions where you crop the images for better visualization
    """
    
    N_imgs = imgs.shape[0]
    
    my_range = np.linspace(0,N_imgs-1,m_plots*n_plots,dtype=np.int)

    
    for idx in range(len(my_range)):
        k = my_range[idx]
        plt.subplot(m_plots,n_plots,idx+1)
        idx += 1
        plt.imshow(imgs[k,x0:x1,y0:y1,:].squeeze())
        plt.axis('off')
        plt.title(str(focus_distances[k]) + " m")
    plt.tight_layout()
    
    


def plot_comparison(img1,img2,method = 'checkerboard'):
    """
    Adapt the 3 methods from: 
    https://scikit-image.org/docs/dev/auto_examples/applications/plot_image_comparison.html
    
    Tip: To use this with RGB images you can implement a for-loop
    to process each color channel differently
    
    Plot only the difference images. Don't plot the overview image on the top
    You can plot the overview in a different function and include it in your report
    
    """
    img_compare = np.zeros(img1.shape)
    for k in range(3):
        img_compare[:,:,k] = compare_images(img1[:,:,k], img2[:,:,k], method)

    plt.figure(figsize=(10,10))
    plt.imshow(img_compare)
    plt.title(method + ' comparison',fontsize=(20))
    plt.tight_layout()
    

def plot_filter_maps(maps,focus_distances,m_plots = 5, n_plots = 5):
    """
    
    Visualize the filtered focal stack. Each subfigure should have
    the focus-distance on the title.
    
    Args:
        maps(np.array): The filtered image where edge information is stored in
        focus_distances(1d-array): contians the focus distances in meter
        m_plots,n_plots(int): number of rows and columns for subplots
    
    """
    
    N_imgs = maps.shape[0]       
    my_range = np.linspace(0,N_imgs-1,m_plots*n_plots,dtype=np.int)
    
    
    for idx in range(len(my_range)):
        k = my_range[idx]
        plt.subplot(m_plots,n_plots,idx+1)

        plt.imshow(maps[k,:,:].squeeze())
        plt.axis('off')
        plt.title(str(focus_distances[k])+"m")

        
        
def get_points_for_focus(dimX,dimY,num_samples_x = 3,num_samples_y = 3):
    """
    
    Samples points on the image in a uniform grid-like manner.
    The sampled points shouldn't be at the borders, but the first point and last
    point should start definetely a bit away from the image borders
    
    
    Good value e.g. to start are dim/6 and 6/7*dim (where dim is the dimension in x and y of the image)
    
    There's much leeway how you choose the points. You can look at the example
    image to get some inspiration. You don't have to choose exactly the same points.
    
    Simply analyse the image and try to find images that you find meaningful
    
    Hint:
     1. You can use the np.meshgrid function.
     2. If you use meshgrid, you can reshape a 2D-matrix into a 1D-mastrix using the .ravel() method
    
    Args:
        dimX(int): the number of pixels of the image in X
        dimY(int): the number of pixels of the image in Y
        num_samples_x(int): how many points should be sampled in X
        num_sampled_y(int): how many points should be sampled in Y
    
    Returns:
        x,y(1D-arrays): the coordinates of the points sampled on the image
    """
    
    x = np.linspace(dimX/6,6/7.0 * dimX,3).astype(int)
    y = np.linspace(dimY/6,6/7.0 * dimY,3).astype(int)

    [X,Y] = np.meshgrid(x,y)
    
    return X.ravel(),Y.ravel()


def plot_focus_metric_distances(focus_maps,focus_distances, x,y):
    
    colors = cm.gist_rainbow(np.linspace(0, 1, len(x)))

    for idx in range(colors.shape[0]):
        plt.plot(focus_distances,focus_maps[:,x[idx],y[idx]],color=colors[idx,:], label=str(idx))
        plt.plot(focus_distances,focus_maps[:,x[idx],y[idx]],'x',color=colors[idx,:])
        plt.ylabel("Normalized Focus Measure")
        plt.xlabel("Focus-Distance")
    plt.grid()
    plt.legend()
        
def plot_chosen_points(img,x,y):
    """
    
    Visualizes the points that are chosen in the scene that where will analyze the focus metric on later

    1. Use the matlplotlib colomap function (e.g. gist_rainbow) to create different colormps for the points
        You can then choose the color with the argument color=colors[idx,:] during plotting
    2. Use plt.scatter to plot both the dots as well as the option marker = '$'+str("TEST")+'$' to add text to the figure
    
    
    Args:
        img: RGB image
        x,y(1D-array): x,ycoordinates of the chosen points   
    """
    plt.imshow(img)
    colors = cm.gist_rainbow(np.linspace(0, 1, len(x)))

    print(colors.shape)

    for idx in range(colors.shape[0]):
        plt.scatter(y[idx], x[idx], s=50,color=colors[idx,:])
        plt.scatter(y[idx]+30, x[idx]-30, s=100, marker='$'+str(idx)+'$',color=colors[idx,:])