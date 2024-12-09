#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, binary_closing


# In[ ]:


def otsu_thresholding_gray(img, num_iter=1, inverse=False):
    # Initialize the mask with all pixels as part of the image for the first iteration
    mask = np.ones_like(img, dtype=bool)
    
    for _ in range(num_iter):
        # Apply Otsu's thresholding to the current region defined by the mask
        hist, bins = np.histogram(img[mask].ravel(), bins=256, range=(0, 256))
        
        total = mask.sum()  # Only consider pixels in the current mask
        current_max = 0
        threshold = 0
        sum_total = np.sum(bins[:-1] * hist)
        sum_bg = 0
        weight_bg = 0
        weight_fg = 0

        for i in range(0, 256):
            weight_bg += hist[i]
            if weight_bg == 0:
                continue

            weight_fg = total - weight_bg
            if weight_fg == 0:
                break

            sum_bg += i * hist[i]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg

            variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

            if variance_between > current_max:
                current_max = variance_between
                threshold = i

        # Create a mask for the next iteration (foreground becomes the region to refine)
        if inverse:
            mask = img > threshold
        else:
            mask = img < threshold
        
    # Perform morphological opening to remove small noise (erosion followed by dilation)
    # mask = binary_opening(mask, structure=np.ones((3, 3)))
    
    # # Perform morphological closing to fill small gaps (dilation followed by erosion)
    # mask = binary_closing(mask, structure=np.ones((5, 5)))

    return mask.astype(np.uint8)


# In[ ]:


def otsu_thresholding(img, iter : list = [1, 1, 1], inverse=False):
    channels = img.shape[2]
    
    # Initialize an empty array for the channel-specific masks
    channel_masks = np.zeros(img.shape, dtype=np.int32)

    # Apply thresholding to each channel separately
    for ch in range(channels):
        print(f"Processing channel {ch + 1} of {channels}...")
        
        # Extract single color channel
        current_channel = img[:, :, ch]
        
        # Apply iterative Otsu thresholding for the current channel
        channel_masks[:, :, ch] = otsu_thresholding_gray(current_channel, iter[ch], inverse=inverse)

    # Combine masks across channels, only keeping foreground where all channels agree
    combined_mask = np.all(channel_masks == 1, axis=2).astype(np.int32)
    print('Completed processing!')

    return channel_masks, combined_mask


# In[ ]:


def plot_channels_segment(channel_masks, filename, channel_labels= ['B', 'G', 'R']):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    # Labels for each channel
    channel_labels = channel_labels
    
    for i in range(channel_masks.shape[2]):
        ax[i].imshow(channel_masks[:, :, i], cmap='gray')
        ax[i].set_title(f'{channel_labels[i]}')
        ax[i].axis('off')  # Remove axes
    
    plt.tight_layout()  # Ensure proper layout of subplots
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved at {filename}!")
    plt.show()


# In[ ]:


def compute_texture_features(img, window_size=3):
    """Compute texture features using a sliding window approach."""
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    half_window = window_size // 2
    height, width = img.shape
    texture_feature = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            # Define the window boundaries
            y_min = max(y - half_window, 0)
            y_max = min(y + half_window + 1, height)
            x_min = max(x - half_window, 0)
            x_max = min(x + half_window + 1, width)

            # Extract the window
            window = img[y_min:y_max, x_min:x_max]

            # Calculate the mean and variance for each channel
            mean_val = np.mean(window)  # Mean across the window
            variance_val = np.var(window)  # Variance across the window

            # Set the variance at the center pixel for each channel
            texture_feature[y, x] = variance_val

    return texture_feature


# In[ ]:


def extract_contours(binary_mask, filename='contour.png', plot=True, save=True):
    # Create an empty contour image
    contour_image = np.zeros(binary_mask.shape, dtype=np.uint8)

    # Iterate through each pixel in the binary mask
    for row in range(1, binary_mask.shape[0] - 1):
        for col in range(1, binary_mask.shape[1] - 1):
            # Skip background pixels
            if binary_mask[row, col] == 0:
                continue

            # Extract a 3x3 window around the current pixel
            local_window = binary_mask[row - 1:row + 2, col - 1:col + 2]
            # Count the number of foreground pixels in the window
            if np.sum(local_window) < 9:
                contour_image[row, col] = 1  # Mark as contour
    if plot:
        plt.imshow(contour_image, cmap='gray')
        plt.axis('off')
    if save:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved at {filename}!")
    return contour_image


# In[ ]:


def plot_channels_contour(channel_masks, filename='contour_channels.png', channel_labels=['B', 'G', 'R']):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    # Labels for each channel
    channel_labels = channel_labels
    for i in range(channel_masks.shape[2]):
        channel_mask= channel_masks[:,:,i]
        # for each mask
        contour_image= extract_contours(channel_mask, plot=False, save=False)
        ax[i].imshow(contour_image, cmap='gray')
        ax[i].set_title(f'{channel_labels[i]}')
        ax[i].axis('off')  # Remove axes
    
    plt.tight_layout()  # Ensure proper layout of subplots
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved at {filename}!")
    plt.show()


# In[ ]:


dog= cv2.imread('dog_small.jpg')
flower= cv2.imread('flower_small.jpg')
tree= cv2.imread('tree.jpeg')
sunflower= cv2.imread('sunflower.jpeg')


# In[ ]:


choose= 'flower'


# In[ ]:


if choose=='dog':
    img= dog
    segment_iter= [2,2,2]
    window_sizes= [3,5,7]
    texture_iter= [1,1,1] 
    inverse= False  
elif choose=='flower':
    img= flower
    segment_iter= [2,2,2]
    window_sizes= [3,5,7]
    texture_iter= [1,1,1]    
    inverse= True  
elif choose=='tree':
    img= tree
    segment_iter= [2,2,2]
    window_sizes= [3,5,7]
    texture_iter= [1,1,1]    
    inverse= False
elif choose=='sunflower':
    img= sunflower
    segment_iter= [1,0,0]
    window_sizes= [5,7,9]
    texture_iter= [1,1,1]   
    inverse= False
else:
    raise RuntimeError("Not implemented yet")


# In[ ]:


mask, combined_mask= otsu_thresholding(img, segment_iter, inverse=inverse)
plt.imshow(combined_mask, cmap='gray')
plt.axis('off')
plt.savefig(f"combined_seg_rgb_{choose}.png")
plt.show()
plot_channels_segment(mask, f"segmented_channels_rgb_{choose}.png")


# In[ ]:


contours= extract_contours(combined_mask, f"combined_contour_rgb_{choose}.png")
plot_channels_contour(mask, f"channels_contour_rgb_{choose}.png")


# In[ ]:


texture_features = np.zeros_like(img)
for i, window_size in enumerate(window_sizes):
    texture_feature = compute_texture_features(img, window_size)
    texture_features[:,:,i]= texture_feature


# In[ ]:


mask, combined_mask= otsu_thresholding(texture_features, texture_iter, inverse=inverse)
plot_channels_segment(mask, f"output/segmented_channels_text_{choose}.png", channel_labels=window_sizes)
plt.imshow(combined_mask, cmap='gray')
plt.axis('off')
plt.savefig(f"output/segmented_combined_text_{choose}.png")
plt.show()
contours= extract_contours(combined_mask, f"output/combined_contour_text_{choose}.png")
plot_channels_contour(mask, f"output/channels_contour_text_{choose}.png", channel_labels=window_sizes)

