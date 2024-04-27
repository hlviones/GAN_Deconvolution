import os
from shutil import copyfile
import argparse
import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter
from tifffile import imread, imsave

def reorganize_files(dir_in, test_ratio=0.2):
    """
    This function reorganizes files in a given directory. It creates two subdirectories, 'sharp' and 'blur', 
    and moves the original files to 'sharp'. It then applies a random blur to each image and saves the blurred 
    images in 'blur'. A small percentage of the pairs are moved to a test folder.
    
    Args:
        dir_in (str): The path to the input directory.
        test_ratio (float): The ratio of data to be used for testing.
        
    Returns:
        tuple: The paths to the 'sharp', 'blur', 'test_sharp', and 'test_blur' directories.
    """
    output_directory_A = os.path.join(dir_in, 'train/gt')
    output_directory_B = os.path.join(dir_in, 'train/blur')
    test_directory_A = os.path.join(dir_in, 'test/gt')
    test_directory_B = os.path.join(dir_in, 'test/blur')
    if not os.path.exists(output_directory_A):
        os.makedirs(output_directory_A)
    if not os.path.exists(output_directory_B):
        os.makedirs(output_directory_B)
    if not os.path.exists(test_directory_A):
        os.makedirs(test_directory_A)
    if not os.path.exists(test_directory_B):
        os.makedirs(test_directory_B)

    # Get a list of all files
    all_files = os.listdir(os.path.join(dir_in))
    num_test_files = int(len(all_files) * test_ratio)
    test_files = np.random.choice(all_files, size=num_test_files, replace=False)

    # Iterate over files in the directory
    for filename in tqdm.tqdm(all_files):
        # Check if it's a file
        if os.path.isfile(os.path.join(dir_in, filename)):
            # Determine whether this file is for testing or training
            if filename in test_files:
                sharp_dir = test_directory_A
                blur_dir = test_directory_B
            else:
                sharp_dir = output_directory_A
                blur_dir = output_directory_B

            # Move file to 'sharp' folder
            sharp_path = os.path.join(sharp_dir, filename)
            copyfile(os.path.join(dir_in, filename), sharp_path)

            # Apply blur function and save in 'blur' folder
            blur_path = os.path.join(blur_dir, filename)
            apply_random_blur(sharp_path, blur_path)  

    # Return the paths to the created folders
    return output_directory_A, output_directory_B, test_directory_A, test_directory_B


def apply_random_blur(image_path, output_path):
    """
    This function applies a random level of Gaussian blur and Gaussian noise to an image. 
    The kernel sizes for the blur are [1.0, 1.2, 1.5] and the standard deviations for the noise are [0, 15, 30]. 
    The blurred and noisy image is saved in TIFF format.
    
    Args:
        image_path (str): The path to the input image.
        output_path (str): The path where the blurred and noisy image will be saved.
    """
    # Open the TIFF image file with tifffile
    image = imread(image_path)

    # Define the kernel sizes for the blur and the standard deviations for the noise
    kernel_sizes = [1.0, 1.2, 1.5]
    noise_std_devs = [0, 15, 30]

    # Randomly select a kernel size and a noise standard deviation
    kernel_size = np.random.choice(kernel_sizes)
    noise_std_dev = np.random.choice(noise_std_devs)

    # Apply Gaussian blur with the selected kernel size
    blurred_image = gaussian_filter(image, sigma=kernel_size)

    # Create Gaussian noise with the selected standard deviation
    noise = np.random.normal(0, noise_std_dev, blurred_image.shape)

    # Add the noise to the blurred image
    noisy_blurred_image = blurred_image + noise

    # Save the noisy and blurred image in TIFF format
    imsave(f"{output_path}_blur_{kernel_size}_noise_{noise_std_dev}.tif", noisy_blurred_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', default='data/training/nuc')
    args = parser.parse_args()
    sharp_dir, blur_dir = reorganize_files(args.dir_in)