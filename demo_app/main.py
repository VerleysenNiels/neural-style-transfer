from io import BytesIO
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from src.nst_model import NST_VGG
from src.style_transfer import transfer_style


def denormalize_image(img, mean, std):
    # Denormalize using the content image's mean and std
    std = std.view(3, 1, 1)
    mean = mean.view(3, 1, 1)
    img = img * std + mean
    img = img.clamp(0, 1)
    return img


def post_process_image(generated_image, content_image_mean, content_image_std):
    """
    Post-processes the generated image by denormalizing it and converting it to PIL format.

    Args:
        generated_image (torch.Tensor): The generated image tensor.
        content_image_mean (List[float]): The mean values for normalizing the content image.
        content_image_std (List[float]): The standard deviation values for normalizing the content image.

    Returns:
        PIL.Image.Image: The post-processed image in PIL format.
    """

    # Denormalize the final generated image
    final_image = denormalize_image(generated_image.squeeze(0).cpu(), content_image_mean, content_image_std)

    # Convert to PIL for visualization
    final_image = transforms.ToPILImage()(final_image)

    return final_image

def plot_losses(content_losses, style_losses, total_losses):
    """
    Plots the content loss, style loss, and total loss over the steps of the neural style transfer algorithm.

    Args:
        content_losses (list): List of content losses at each step.
        style_losses (list): List of style losses at each step.
        total_losses (list): List of total losses at each step.

    Returns:
        PIL.Image.Image: The plot of the losses as an image.
    """

    plt.figure(figsize=(18, 6))
    plt.style.use('dark_background')
    
    # Content loss plot
    plt.subplot(1, 3, 1)
    plt.plot(content_losses, color='deepskyblue')
    plt.xlabel('Steps', color='white')
    plt.ylabel('Loss', color='white')
    plt.title('Content Loss', color='white')
    
    # Style loss plot
    plt.subplot(1, 3, 2)
    plt.plot(style_losses, color='deepskyblue')
    plt.xlabel('Steps', color='white')
    plt.title('Style Loss', color='white')
    
    # Total loss plot
    plt.subplot(1, 3, 3)
    plt.plot(total_losses, color='deepskyblue')
    plt.xlabel('Steps', color='white')
    plt.title('Total Loss', color='white')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', facecolor='black')
    buf.seek(0)

    return Image.open(buf)

def style_transfer_app(nst_model, content_image, style_image, device, alpha, beta, learning_rate, num_steps):    
    # Perform style transfer
    output_tensor, content_losses, style_losses, total_losses = transfer_style(
        nst_model, content_image, style_image, device, alpha, beta, learning_rate, num_steps, content_image_mean, content_image_std
    )

    loss_plot = plot_losses(content_losses, style_losses, total_losses)
    
    return output_tensor, loss_plot


if __name__ == "__main__":
    # Define the content and style images
    content_image_path = "examples/content_images/crete.JPG"
    style_image_path = "examples/style_images/bob_ross.jpg"
    
    # Load the content and style images
    content_image = Image.open(content_image_path)
    style_image = Image.open(style_image_path)

    # Convert images to RGB space as a safety measure depending on the input images
    content_image = content_image.convert("RGB")
    style_image = style_image.convert("RGB")

    # Define the mean and std for normalizing the images
    content_image_mean = torch.tensor([transforms.ToTensor()(content_image)[:, :, c].mean() for c in range(3)])
    content_image_std = torch.tensor([transforms.ToTensor()(content_image)[:, :, c].std() for c in range(3)])
    
    # Hyperparameters
    num_steps = 100  # Number of optimization steps
    learning_rate = 0.001 # Step size when optimizing
    alpha = 1        # Weight for content loss
    beta = 1000       # Weight for style loss
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the NST model
    nst_model = NST_VGG().to(device)
    
    # Run the style transfer app
    output_tensor, loss_plot = style_transfer_app(nst_model, content_image, style_image, device, alpha, beta, learning_rate, num_steps)

    output_image = post_process_image(output_tensor, content_image_mean, content_image_std)
    
    # Display the output image and loss plot
    output_image.show()
    loss_plot.show()