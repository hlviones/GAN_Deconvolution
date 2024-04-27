import torch.nn as nn
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: (batch_size, input_channels, 10, 64, 64)
            nn.Conv3d(input_channels, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (batch_size, 64, 10, 32, 32)
            nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (batch_size, 128, 10, 16, 16)
            nn.Conv3d(128, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (batch_size, 256, 10, 8, 8)
            nn.Conv3d(256, 512, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (batch_size, 512, 10, 4, 4)
            nn.Conv3d(512, 1, kernel_size=(10, 4, 4), stride=(1, 1, 1), padding=0, bias=False),
            nn.Sigmoid()
            # Output: (batch_size, 1, 1, 1, 1)
        )

    def forward(self, gt_image, generated_image):
        # Convert input tensors to float data type
        gt_image = gt_image.to(dtype=torch.float)
        generated_image = generated_image.to(dtype=torch.float)

        # Concatenate the ground truth image and the generated image along the channel dimension
        input_images = torch.cat([gt_image, generated_image], dim=2)
        return self.main(input_images).view(-1, 1)