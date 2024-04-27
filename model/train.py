import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio
from utils.ssim import get_SSIM
from tifffile import imread, imsave
from PIL import Image
import os
from torchvision import datasets, transforms, models
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(generator_model, train_dataloader, val_dataloader):
    epochs = 50

    # Initialize the main model and the discriminator
    generator_model = generator_model.to(device)
    discriminator_model = models.resnet18().to(device)

    # Freeze all but the final layer of the discriminator
    for name, param in discriminator_model.named_parameters():
        if "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Define the optimizers and loss functions
    gen_optimizer = torch.optim.Adam(generator_model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    discriminator_optimizer = torch.optim.SGD(discriminator_model.fc.parameters(), lr=0.001, momentum=0.9)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    psnr = PeakSignalNoiseRatio().to(device)
    deconvolved_images = {}
    ground_truth_images = {}
    psnrs = {}
    psnrs_list = {}
    gen_losses = {}
    disc_losses = {}
    for epoch in range(epochs):
        print(' -- Starting training epoch {} --'.format(epoch + 1))

        # Train the generator
        generator_model.train()
        discriminator_model.eval()
        train_gen_loss = []
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            gen_optimizer.zero_grad()

            # Forward pass through the generator
            y_hat, y_k, update = generator_model(x.float())
            
            # Compute the generator loss
            gen_loss = mse_loss(y_hat.float(), y.float()) - torch.log((1 + get_SSIM(y, y_hat)) / 2)

            # Add the discriminator loss to the generator loss if epoch is not 1
            if epoch != 0:  # epochs are 0-indexed
                # Use the precalculated discriminator loss
                precalculated_discriminator_loss = np.mean(np.array(train_discriminator_loss))
                # Define a weighting factor
                weight_factor = 0.1
                # gen_loss += weight_factor * precalculated_discriminator_loss

            gen_loss.backward(retain_graph=True)
            gen_optimizer.step()
            train_gen_loss.append(gen_loss.item())

        print('Generator train loss: {}'.format(np.mean(np.array(train_gen_loss))))
        gen_losses[epoch] = (np.mean(np.array(train_gen_loss)))
        psnrs_list[epoch] = psnr(y_hat.float(), y.float()).item()

        if epoch % 50 == 0:
            deconvolved_image = y_hat[0].cpu().detach().numpy().squeeze()
            ground_truth_image = y[0].cpu().detach().numpy().squeeze()
            deconvolved_images[epoch] = deconvolved_image[0]
            ground_truth_images[epoch] = ground_truth_image[0]
            # Calculate and store PSNR
            psnrs[epoch] = psnr(y_hat.float(), y.float()).item()

        # Train the discriminator
        generator_model.eval()
        discriminator_model.train()
        train_discriminator_loss = []


        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        ground_truth_images_transformed = []
        deconvolved_images_transformed = []

        for i in range(y.shape[0]):
            ground_truth_image = y[i].cpu().detach().numpy().squeeze()
            deconvolved_image = y_hat[i].cpu().detach().numpy().squeeze()

            for j in range(ground_truth_image.shape[0]):
                gt_img = Image.fromarray(ground_truth_image[j]).convert("RGB")
                gt_img_transformed = transform(gt_img)
                ground_truth_images_transformed.append(gt_img_transformed)

                dc_img = Image.fromarray(deconvolved_image[j]).convert("RGB")
                dc_img_transformed = transform(dc_img)
                deconvolved_images_transformed.append(dc_img_transformed)

        # Convert the lists of tensors to tensors
        ground_truth_images_transformed = torch.stack(ground_truth_images_transformed)
        deconvolved_images_transformed = torch.stack(deconvolved_images_transformed)

        # Concatenate the real and fake images and their corresponding labels
        discriminator_inputs = torch.cat((ground_truth_images_transformed, deconvolved_images_transformed), dim=0)
        discriminator_labels = torch.cat((torch.ones(len(ground_truth_images_transformed), dtype=torch.long), 
                                    torch.zeros(len(deconvolved_images_transformed), dtype=torch.long)), dim=0)

        # Create the discriminator dataset and dataloader
        discriminator_dataset = torch.utils.data.TensorDataset(discriminator_inputs, discriminator_labels)
        discriminator_dataloader = torch.utils.data.DataLoader(discriminator_dataset, batch_size=64, shuffle=True)

        train_discriminator_loss = []

        for inputs, labels in discriminator_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass through the discriminator
            discriminator_optimizer.zero_grad()
            discriminator_output = discriminator_model(inputs)
            discriminator_loss = ce_loss(discriminator_output, labels)
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()
            train_discriminator_loss.append(discriminator_loss.item())

        print('Discriminator train loss: {}'.format(np.mean(np.array(train_discriminator_loss))))
        disc_losses[epoch] = (np.mean(np.array(train_discriminator_loss)))
        if (epoch % 5 == 0) or (epoch + 1 == epochs):
            generator_model.eval()
            val_loss = []
            val_ssim = []
            val_psnr = []
            for x, y in val_dataloader:
                x, y = x.to(device), y.to(device)

                y_hat, y_k, update = generator_model(x.float())

                loss = mse_loss(y_hat.float(), y.float()) - torch.log((1+get_SSIM(y, y_hat))/2)

                val_loss.append(loss.item())
                val_ssim.append(get_SSIM(y, y_hat).item())
                val_psnr.append(psnr(y, y_hat).item())
            print('testing loss: {}'.format(np.mean(np.array(val_loss))))
            print('testing ssim: {} +- {}'.format(np.round(np.mean(np.array(val_ssim)), 5), np.round(np.std(np.array(val_ssim)), 5)))
            print('testing psnr: {} +- {}'.format(np.round(np.mean(np.array(val_psnr)), 5), np.round(np.std(np.array(val_psnr)), 5)))
        # Display all images in a single figure at the end of the loop
    
    num_epochs = len(sorted(deconvolved_images.keys()))
    fig, axs = plt.subplots(1, 2 * num_epochs, figsize=(12, 4))

    for i, epoch in enumerate(sorted(deconvolved_images.keys())):
        axs[2 * i].imshow(deconvolved_images[epoch], cmap='gray')
        axs[2 * i].set_title(f'Deconvolved Image - Epoch {epoch + 1}')
        axs[2 * i].text(0.95, 0.01, f'PSNR: {psnrs[epoch]:.2f}', verticalalignment='bottom', horizontalalignment='right', transform=axs[2 * i].transAxes, color='white', fontsize=15)
        axs[2 * i].axis('off')

        axs[2 * i + 1].imshow(ground_truth_images[epoch], cmap='gray')
        axs[2 * i + 1].set_title(f'Ground Truth Image - Epoch {epoch + 1}')
        axs[2 * i + 1].text(0.95, 0.01, f'PSNR: {psnrs[epoch]:.2f}', verticalalignment='bottom', horizontalalignment='right', transform=axs[2 * i + 1].transAxes, color='white', fontsize=15)
        axs[2 * i + 1].axis('off')

    plt.suptitle('Training Progress')
    plt.show()

    # Plot PSNR over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(*zip(*sorted(psnrs_list.items())))
    plt.title('PSNR over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.show()

    # Plot Generator and Discriminator loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(*zip(*sorted(gen_losses.items())), label='Generator Loss')
    plt.plot(*zip(*sorted(disc_losses.items())), label='Discriminator Loss')
    plt.title('Generator and Discriminator Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return generator_model