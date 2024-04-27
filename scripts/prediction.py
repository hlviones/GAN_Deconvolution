from    model.lucyd import *
import os
model = LUCYD(num_res=1).to(device)


WEIGHTS='model/weights/lucyd-mixture.pth'
# WEIGHTS='lucyd-act.pth'
# WEIGHTS='lucyd-nuc.pth'

model.load_state_dict(torch.load(os.path.join('', WEIGHTS)))


#@title # Deconvolution
#@markdown The paths assume the starting directory is your Google Drive.<br>
#@markdown Do not include `/` at the end of `output_dir`.<br>

input = "input/Nuc_5.tif" #@param {type:"string"}
output_dir = "input" #@param {type:"string"}



# Output
if output_dir == '':
    dir_out = 'lucyd-output'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
dir_out = output_dir

# -- DO THINGS --
import tifffile
import numpy as np
# Load the image
yy = tifffile.imread(input, maxworkers=6)
yy = (yy - np.min(yy))/(np.max(yy) - np.min(yy))
print('Image loaded.')

# Initialize the canvas and output arrays
canvas = np.zeros((((yy.shape[0]//512)+1)*512,((yy.shape[1]//512)+1)*512,((yy.shape[2]//512)+1)*512))
output = np.zeros(canvas.shape)
canvas[:yy.shape[0],:yy.shape[1],:yy.shape[2]] = yy

# Define the overlap size and adjust the crop and step sizes
overlap_size = 16
crop_size_z = 32 + 2 * overlap_size
crop_size_y = 64 + 2 * overlap_size
crop_size_x = 64 + 2 * overlap_size
step_size_z = 32
step_size_y = 64
step_size_x = 64

# Initialize the count array
count = np.zeros(canvas.shape)

# Running deconvolution.
print('Running deconvolution.')
model.eval()
with torch.no_grad():
    for z_i in range(0, canvas.shape[0] - crop_size_z + 1, step_size_z):
        for y_i in range(0, canvas.shape[1] - crop_size_y + 1, step_size_y):
            for x_i in range(0, canvas.shape[2] - crop_size_x + 1, step_size_x):
                # Extract the crop from the canvas
                crop = canvas[z_i:z_i+crop_size_z, y_i:y_i+crop_size_y, x_i:x_i+crop_size_x]
                crop_torch = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).float().to(device)

                # Pass the crop through the model
                out = model(crop_torch)[0].detach().cpu().numpy()

                # Create a blending mask
                blend_mask = np.ones((crop_size_z, crop_size_y, crop_size_x))
                overlap_mask_z = np.linspace(0, 1, overlap_size)
                overlap_mask_y = np.linspace(0, 1, overlap_size)
                overlap_mask_x = np.linspace(0, 1, overlap_size)
                blend_mask[:overlap_size, :, :] *= overlap_mask_z[:, None, None]
                blend_mask[-overlap_size:, :, :] *= overlap_mask_z[::-1, None, None]
                blend_mask[:, :overlap_size, :] *= overlap_mask_y[None, :, None]
                blend_mask[:, -overlap_size:, :] *= overlap_mask_y[None, ::-1, None]
                blend_mask[:, :, :overlap_size] *= overlap_mask_x[None, None, :]
                blend_mask[:, :, -overlap_size:] *= overlap_mask_x[None, None, ::-1]

                # Multiply the output by the blending mask
                out *= blend_mask

                # Add the output to the corresponding location in the output array
                output[z_i:z_i+crop_size_z, y_i:y_i+crop_size_y, x_i:x_i+crop_size_x] += out[0,0]

                # Increment the count array
                count[z_i:z_i+crop_size_z, y_i:y_i+crop_size_y, x_i:x_i+crop_size_x] += blend_mask
                
print('Deconvolution done.')

# Divide the output array by the count array to get the average
epsilon = 1e-7
output /= (count + epsilon)

# Normalize the output
output = (output - np.min(output)) / (np.max(output) - np.min(output))

# Crop the output to the original shape of yy
output = output[:yy.shape[0], :yy.shape[1], :yy.shape[2]]

print('Saving results as:', dir_out+'/Act_5_output.tif')
tifffile.imwrite(
    dir_out+'/test_output.tif',
    (output*255).astype(np.uint16),
    metadata={'axes': 'ZYX'},
    imagej=True,
)

print('Deconvolution OK.')