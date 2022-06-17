import os

import random
import numpy as np

import skimage
import cv2

import matplotlib.pyplot as plt

KERNEL = np.ones((3, 3), dtype = np.uint16)

def analyse(path: str, out_dir: str) -> None:
    if os.path.isfile(path):
        analyse_single(path, out_dir)

    if os.path.isdir(path):
        for img in os.listdir(path):
            if os.path.isfile(img):
                analyse_single(os.path.join(path, img), out_dir)

def analyse_single(img_path: str, out_dir: str) -> None:
    img_name = os.path.split(img_path)[1].split(".")[0]
    print(f"IMAGE PATH: {img_path}")
    img = skimage.io.imread(img_path)
    processed_img = pre_process(img)
    contour_data = detect_contours(processed_img)
    fig = plot_results(processed_img, contour_data)
    write_results(contour_data, img_name, out_dir, fig)

def pre_process(image: np.ndarray) -> np.ndarray:
    grayscale_img = skimage.color.rgb2gray(image)
 
    # Applying Gaussian filter.
    blurred_img = skimage.filters.gaussian(grayscale_img, 1)

    # Apply open filter. 
    opened_img = cv2.morphologyEx(blurred_img, cv2.MORPH_OPEN, KERNEL, iterations = 2)

    # Apply close filter.
    closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, KERNEL, iterations = 2)

    # Erode the image.
    eroded_img = cv2.erode(closed_img, KERNEL, iterations = 1)
    
    return eroded_img

def detect_contours(image: np.ndarray) -> list[np.ndarray]:
    contour_data = skimage.measure.find_contours(image, level = 0.2)
    return contour_data

def plot_results(image: np.ndarray, 
                 contour_data: list[np.ndarray],
                 figsize: tuple = (16, 9)
                 ) -> None:
    
    fig, ax = plt.subplots(nrows = 1, 
                        figsize = figsize,
                        gridspec_kw = dict(left = 0.01, right = 0.9,
                                            bottom = 0.0001, top = 0.9)) 
    
    ax.imshow(image, cmap = "gray")
    
    for i, contour in enumerate(contour_data):
        centroid_x, centroid_y = np.mean(contour, axis = 0)
        ax.scatter(centroid_y, centroid_x, s = 10, c = "red")
        ax.text(centroid_y, centroid_x, f"{i}", fontsize = 10)
        
    return fig

def write_results(contour_data: list[np.ndarray], 
                  img_name: str,
                  out_dir: str,
                  fig: plt.figure) -> None:
    
    
    write_dir = os.path.join(out_dir, "_".join(["out", img_name]))
    
    
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
        
        
    # Text data.
    contour_count = len(contour_data)
    
    with open(os.path.join(write_dir, "data.txt"), "w") as f:
        f.write(f"Contour count: {contour_count}\n")
    
    # Image data.
    fig.savefig(os.path.join(write_dir, ".".join(["contours", "png"])), 
                             dpi = 80,
                             bbox_inches = "tight")