import torch


def calculate_gram_matrix(feature_map):
    # Extract dimensions
    _, n_channels, height, width = feature_map.shape

    # Flatten the feature map by reshaping it into (n_channels, height * width)
    features = feature_map.view(n_channels, height * width)

    # Calculate the Gram matrix as the product of the features with its transpose
    gram = torch.mm(features, features.t())

    return gram


def calculate_loss(generated_content_feature_maps, content_feature_maps, generated_style_feature_maps, style_grams, alpha, beta):
    # Calculate content loss as the MSE between the feature maps of the given layer
    # Looking at other implementations conv4_2 seems to be sufficient for the content loss
    content_loss = 0
    for content_features, generated_content_features in zip(content_feature_maps, generated_content_feature_maps):
        content_loss += torch.nn.functional.mse_loss(
            content_features, generated_content_features, reduction='mean')

    # Average content loss across layers
    # content_loss /= len(content_features)

    # Calculate the style loss using the Gram matrices
    style_loss = 0
    for style_gram, gen_feature_style in zip(style_grams, generated_style_feature_maps):
        generated_gram = calculate_gram_matrix(gen_feature_style)
        layer_style_loss = torch.nn.functional.mse_loss(
            generated_gram, style_gram)
        style_loss += layer_style_loss

    # Average style loss across layers
    # style_loss /= len(style_grams)

    # Combine the content and style losses with their respective weights
    total_loss = alpha * content_loss + beta * style_loss

    return total_loss, content_loss, style_loss
