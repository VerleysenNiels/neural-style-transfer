import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from src.loss_functions import calculate_gram_matrix, calculate_loss


def transfer_style(nst_model, content_image, style_image, device, learning_rate, alpha, beta, num_steps, content_image_mean, content_image_std):
    """
    Transfers the style of a given style image onto a content image using neural style transfer.

    Args:
        nst_model (torch.nn.Module): The neural style transfer model.
        content_image (PIL.Image.Image): The content image.
        style_image (PIL.Image.Image): The style image.
        device (torch.device): The device to perform the computation on.
        learning_rate (float): The learning rate for the optimizer.
        alpha (float): The weight for the content loss.
        beta (float): The weight for the style loss.
        num_steps (int): The number of optimization steps.
        content_image_mean (List[float]): The mean values for normalizing the content image.
        content_image_std (List[float]): The standard deviation values for normalizing the content image.

    Returns:
        Tuple[torch.Tensor, List[float], List[float], List[float]]: A tuple containing the generated image,
        content losses, style losses, and total losses at each optimization step.
    """
    # Define preprocessing transformation using the content image's statistics
    preprocess = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        # Use content image's mean and std
        transforms.Normalize(content_image_mean, content_image_std)
    ])

    # Apply preprocessing
    content_tensor = preprocess(content_image).unsqueeze(0)
    content_tensor = content_tensor.to(device)
    style_tensor = preprocess(style_image).unsqueeze(0).to(device)

    # Pre-compute feature maps
    content_features, _ = nst_model(content_tensor)
    _, style_features = nst_model(style_tensor)

    # Pre-compute Gram matrices
    style_grams = [calculate_gram_matrix(
        feature_map) for feature_map in style_features]

    # Initialize the generated image (starting with the content image)
    generated_image = content_tensor.clone().requires_grad_(True).to(device)

    # Set up the optimizer
    optimizer = torch.optim.Adam([generated_image], lr=learning_rate)

    content_losses = []
    style_losses = []
    total_losses = []

    # Optimization loop
    for _ in tqdm(range(num_steps)):
        # Extract features from the generated image
        generated_content_features, generated_style_features = nst_model(
            generated_image)

        # Compute the total loss
        total_loss, content_loss, style_loss = calculate_loss(
            generated_content_feature_maps=generated_content_features,
            content_feature_maps=content_features,
            generated_style_feature_maps=generated_style_features,
            style_grams=style_grams,
            alpha=alpha,
            beta=beta
        )

        # Backpropagation and optimization step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        content_losses.append(content_loss.item())
        style_losses.append(style_loss.item())
        total_losses.append(total_loss.item())

    return generated_image, content_losses, style_losses, total_losses
