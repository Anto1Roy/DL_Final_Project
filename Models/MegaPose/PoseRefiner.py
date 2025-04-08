import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.helpers import quaternion_to_matrix

class PoseRefiner(nn.Module):
    """
    A render-and-compare network that takes an image crop and a CAD render,
    and predicts an incremental pose correction (rotation and translation delta).
    """
    def __init__(self, in_channels=6, feature_dim=64):
        """
        Args:
            in_channels (int): Number of input channels after concatenating real_crop and cad_render.
                               Typically, if both inputs are RGB images, in_channels=6.
            feature_dim (int): Number of output channels for the final convolutional feature map.
        """
        super(PoseRefiner, self).__init__()
        # Convolutional layers to process the concatenated inputs.
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, feature_dim, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(feature_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Global pooling to (B, feature_dim, 1, 1)
        
        # Fully-connected layers to predict rotation (as a quaternion) and translation.
        self.fc_rot = nn.Linear(feature_dim, 4)  # Outputs quaternion (w, x, y, z)
        self.fc_trans = nn.Linear(feature_dim, 3)  # Outputs translation delta

    def forward(self, real_crop, cad_render):
        """
        Args:
            real_crop: Cropped image region or feature map from the detector,
                       Tensor of shape (B, C, H, W).
            cad_render: Rendered CAD image/template from the render bank,
                        Tensor of shape (B, C, H, W).
        
        Returns:
            delta_R: Predicted rotation correction as a rotation matrix (B, 3, 3).
            delta_t: Predicted translation correction (B, 3).
        """
        # Concatenate along the channel dimension.
        x = torch.cat([real_crop, cad_render], dim=1)  # Expected shape: (B, in_channels, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling to get a feature vector.
        x = self.avgpool(x)  # Shape: (B, feature_dim, 1, 1)
        x = x.view(x.size(0), -1)  # Shape: (B, feature_dim)
        
        # Predict quaternion for rotation correction.
        delta_q = self.fc_rot(x)  # Shape: (B, 4)
        # Normalize to ensure unit quaternion.
        delta_q = delta_q / (torch.norm(delta_q, dim=1, keepdim=True) + 1e-8)
        
        # Predict translation correction.
        delta_t = self.fc_trans(x)  # Shape: (B, 3)
        
        # Convert the quaternion into a rotation matrix.
        delta_R = quaternion_to_matrix(delta_q)  # Shape: (B, 3, 3)
        
        return delta_R, delta_t

# Quick testing of the PoseRefiner.
if __name__ == "__main__":
    # Create dummy real_crop and cad_render tensors.
    # Here we assume both are RGB images of size 64x64.
    B = 2  # Batch size
    real_crop = torch.rand(B, 3, 64, 64)
    cad_render = torch.rand(B, 3, 64, 64)
    
    # Instantiate the PoseRefiner.
    refiner = PoseRefiner(in_channels=6, feature_dim=64)
    
    # Forward pass.
    delta_R, delta_t = refiner(real_crop, cad_render)
    
    print("Delta Rotation Matrices (batch):")
    print(delta_R)
    print("\nDelta Translation Vectors (batch):")
    print(delta_t)