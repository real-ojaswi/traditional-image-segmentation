# **Image Segmentation and Texture Analysis**

This repository contains Python implementations for multi-channel image segmentation and texture analysis using Otsu's thresholding, contour extraction, and texture feature computation. The implementation emphasizes a custom-built approach to processing RGB and grayscale images in order to understand the principles of Otsu based thresholding.

---

## **Features**

1. **Iterative Otsu's Thresholding**:
   - Implements Otsu's thresholding from scratch for RGB and grayscale images.
   - Allows iterative refinement of segmentation masks for improved accuracy.
   - Supports inverse thresholding for flexible segmentation.

2. **Channel-Specific Segmentation**:
   - Processes each color channel (R, G, B) independently.
   - Combines channel-specific masks into a unified segmentation result.

3. **Texture Feature Computation**:
   - Computes local texture features (mean and variance) using sliding window operations.
   - Generates multi-channel texture feature maps for further analysis.

4. **Contour Extraction**:
   - Identifies and visualizes contours of segmented regions using a custom implementation.
   - Creates per-channel contour maps and combined contour images.

5. **Visualization and Debugging**:
   - Saves intermediate segmentation masks, texture maps, and contour visualizations.
   - Outputs both channel-specific and combined results for analysis.

---


### **Dependencies**
Ensure the following Python libraries are installed:
- `numpy`
- `opencv-python`
- `matplotlib`
- `scipy`


