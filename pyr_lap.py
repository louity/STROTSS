import torch.nn.functional as F

def create_laplacian_pyramid(image, pyramid_depth):
    laplacian_pyramid = []
    current_image = image
    for i in range(pyramid_depth):
        current_image_H = current_image.size(2)
        current_image_W = current_image.size(3)

        current_image_downsampled = F.interpolate(current_image, (max(current_image_H//2,1), max(current_image_W//2,1)), mode='bilinear')
        current_image_reconstructed  = F.interpolate(current_image_downsampled, (current_image_H, current_image_W), mode='bilinear')
        lap = current_image - current_image_reconstructed
        laplacian_pyramid.append(lap)
        current_image = current_image_downsampled

    laplacian_pyramid.append(current_image)

    return laplacian_pyramid

def synthetize_image_from_laplacian_pyramid(laplacian_pyramid):

    current_image = laplacian_pyramid[-1]
    pyramid_depth = len(laplacian_pyramid)
    for i in range(pyramid_depth-2, -1, -1):
        up_x = laplacian_pyramid[i].size(2)
        up_y = laplacian_pyramid[i].size(3)
        current_image = laplacian_pyramid[i] + F.interpolate(current_image, (up_x,up_y), mode='bilinear')

    return current_image
