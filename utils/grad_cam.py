# utils/grad_cam.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval() # Set model to evaluation mode
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        # Register hooks to capture gradients and activations [23, 24, 25, 26]
        # Iterate through named modules to find the target layer
        found_layer = False
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(self._save_activations_hook)
                module.register_backward_hook(self._save_gradients_hook)
                found_layer = True
                break
        if not found_layer:
            raise ValueError(f"Target layer '{target_layer_name}' not found in model.")

    def _save_activations_hook(self, module, input, output):
        self.activations = output

    def _save_gradients_hook(self, module, grad_input, grad_output):
        # grad_output contains the gradients with respect to the output of the layer
        self.gradients = grad_output[0] # Take the first element as grad_output is a tuple [23]

    def generate_heatmap(self, input_image_tensor, target_class_idx):
        """
        Generates a Grad-CAM heatmap for a specific target class.
        Args:
            input_image_tensor (torch.Tensor): Preprocessed input image tensor (batch_size=1).
            target_class_idx (int): Index of the class for which to generate the heatmap.
        Returns:
            np.ndarray: Normalized heatmap (0-1).
        """
        # Ensure input image requires gradients
        input_image_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_image_tensor)
        
        # Zero gradients
        self.model.zero_grad()

        # Compute loss for the target class logit
        # For multi-label, we take the gradient with respect to the specific class logit
        target_output = output[:, target_class_idx]
        
        # Backward pass to compute gradients [23, 24]
        # Use torch.ones_like for scalar output to ensure proper gradient computation
        target_output.backward(torch.ones_like(target_output), retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy() # Shape: (C, H, W)
        activations = self.activations.cpu().data.numpy() # Shape: (C, H, W)

        # Pool the gradients across spatial dimensions (average pooling) [23, 24]
        pooled_gradients = np.mean(gradients, axis=(1, 2)) # Shape: (C,)

        # Weight the channels by their corresponding pooled gradients
        for i in range(activations.shape[0]): # Iterate over channels
            activations[i, :, :] *= pooled_gradients[i]

        # Sum the weighted activations across the channel dimension to get the heatmap [23, 24]
        heatmap = np.sum(activations, axis=0)

        # Apply ReLU to remove negative values [23, 24]
        heatmap = np.maximum(heatmap, 0)

        # Normalize the heatmap to 0-1 [23, 24]
        if np.max(heatmap) == 0: # Handle case where heatmap is all zeros
            return heatmap
        heatmap /= np.max(heatmap)

        return heatmap

    @staticmethod
    def overlay_heatmap(original_image_path, heatmap, alpha=0.4):
        """
        Overlays a heatmap onto the original image.
        Args:
            original_image_path (str): Path to the original image.
            heatmap (np.ndarray): Normalized heatmap (0-1).
            alpha (float): Transparency factor for the heatmap.
        Returns:
            np.ndarray: Image with heatmap overlay, in RGB format.
        """
        # Read the original image using OpenCV [23]
        img = cv2.imread(original_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB

        # Resize heatmap to original image size [23]
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_scaled = np.uint8(255 * heatmap_resized) # Scale to 0-255
        
        # Apply colormap (JET is common for heatmaps) [23]
        heatmap_colored = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # Convert to RGB

        # Overlay heatmap on original image [23]
        superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
        return superimposed_img