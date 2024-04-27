
from model.lucyd import LUCYD
from model.train import *
from utils.loader import *



"Load data"


# Read the data
blur_data, gt_data = read_data(folder_name='training/nuc/')

# Define the depth of slices for forward pass
depth = 10  # replace with your value

# Create the ImageLoader
dataset = ImageLoader(gt=gt_data, blur=blur_data, depth=depth)

# Define batch size
batch_size = 16  # replace with your value

# Create the DataLoader
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Read the data
blur_data, gt_data = read_data(folder_name='training/test/')
# Create the ImageLoader
test_dataset = ImageLoader(gt=gt_data, blur=blur_data, depth=depth)

# Create the DataLoader
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Now you can use train_dataloader in your training function

model = LUCYD(num_res=1)
model = train(model, train_dataloader, test_dataloader)

torch.save(model.state_dict(), 'model_weights.pth')