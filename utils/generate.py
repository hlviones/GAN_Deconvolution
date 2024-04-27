import tifffile
import os
import numpy as np
import os
from shutil import copyfile
import numpy as np
from scipy.ndimage import gaussian_filter
from tifffile import imread, imsave

import tifffile
import os
import numpy as np
from scipy.ndimage import gaussian_filter

def apply_random_blur_to_tif(input_path, output_path):
    """
    Applies random Gaussian blur and noise to each frame in a TIF image and saves the
    processed image as a new TIF file.

    Args:
        input_path (str): The path to the input TIF image.
        output_path (str): The path where the processed TIF image will be saved.
    """
    # Open the TIF image
    with tifffile.TiffFile(input_path) as tif:
        # Get the number of frames in the TIF image
        num_frames = len(tif.pages)

        # Create a list to store the processed frames
        processed_frames = []

        # Iterate through each frame
        for i, page in enumerate(tif.pages):
            # Get the image data for the current frame
            image = page.asarray()

            # Apply the random blur function
            processed_frame = apply_random_blur(image)
            processed_frames.append(processed_frame)

        # Stack the processed frames into a single 3D array
        processed_image = np.stack(processed_frames, axis=0)

        # Save the processed image as a new TIF file
        tifffile.imwrite(output_path, processed_image)

def apply_random_blur(image):
    """
    Applies a random level of Gaussian blur and Gaussian noise to an image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The processed image.
    """
    # Define the kernel sizes for the blur and the standard deviations for the noise
    kernel_sizes = [1.0, 1.2, 1.5]
    noise_std_devs = [0, 1, 2]

    # Randomly select a kernel size and a noise standard deviation
    kernel_size = np.random.choice(kernel_sizes)
    noise_std_dev = np.random.choice(noise_std_devs)

    # Apply Gaussian blur with the selected kernel size
    blurred_image = gaussian_filter(image, sigma=kernel_size)

    # Create Gaussian noise with the selected standard deviation
    noise = np.random.normal(0, noise_std_dev, blurred_image.shape)

    # Add the noise to the blurred image
    noisy_blurred_image = blurred_image + noise

    return noisy_blurred_image

apply_random_blur_to_tif("0_12.tif", "12.tif")