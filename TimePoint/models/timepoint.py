import torch
import torch.nn as nn
import torch.nn.functional as F

from TimePoint.utils.kp_utils import non_maximum_suppression, get_topk_in_original_order
from TimePoint.models.layers import ConvBlock1D, WTConvBlock1D

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class Encoder1D(nn.Module):
    """
    Encoder that downsamples the input signal by a factor of 8.
    """
    def __init__(self, input_channels=1, dims=[64,64,128,128]):
        super(Encoder1D, self).__init__()
        self.layer1 = ConvBlock1D(input_channels, dims[0], stride=1)
        self.layer2 = ConvBlock1D(dims[0], dims[1], stride=2)
        self.layer3 = ConvBlock1D(dims[1], dims[2], stride=2)
        self.layer4 = ConvBlock1D(dims[2], dims[3], stride=2)

    def forward(self, x):
        # Input x: [N, C, L]
        x = self.layer1(x)   # [N, base_channels, L]
        x = self.layer2(x)   # [N, base_channels, L/2]
        x = self.layer3(x)   # [N, base_channels*2, L/4]
        x = self.layer4(x)   # [N, base_channels*2, L/8]
        return x  # Feature map of size L/8


class WTConvEncoder1D(nn.Module):
    """
    Encoder that downsamples the input signal by a factor of 8.
    """
    def __init__(self, input_channels=1, dims=[64,64,128,128], stride=2, wt_levels=[3,3,3]):
        super(WTConvEncoder1D, self).__init__()
        self.stride = stride
        self.layer1 = ConvBlock1D(input_channels, dims[0], stride=1, padding="same")
        self.layer2 = WTConvBlock1D(dims[0], dims[1], stride=self.stride, wt_levels=wt_levels[0]) # stride=2 to downsample
        self.layer3 = WTConvBlock1D(dims[1], dims[2], stride=self.stride, wt_levels=wt_levels[1])
        self.layer4 = WTConvBlock1D(dims[2], dims[3], stride=self.stride, wt_levels=wt_levels[2])

    def forward(self, x):
        # Input x: [N, C, L]
        x = self.layer1(x)   # [N, base_channels, L]
        x = self.layer2(x)   # [N, base_channels, L/2]
        x = self.layer3(x)   # [N, base_channels*2, L/4]
        x = self.layer4(x)   # [N, base_channels*2, L/8]
        return x  # Feature map of size L/8

class DetectorHead1D(nn.Module):
    """
    Detector Head for predicting keypoint probability map.
    """
    def __init__(self, input_channels, cell_size=8):
        super(DetectorHead1D, self).__init__()
        self.cell_size = cell_size
        self.conv = nn.Conv1d(input_channels, cell_size + 1, kernel_size=1)  # Output channels: cell_size + 1 (dustbin)

    def forward(self, x):
        # x: [N, C, L/8]
        N, C, Lc = x.shape  # Lc = L/8
        x = self.conv(x)     # [N, cell_size + 1, Lc]

        # Reshape to [N, cell_size + 1, Lc]
        # Softmax over the cell_size + 1 channels (including dustbin)
        x = F.sigmoid(x)
        # Remove dustbin (last channel)
        x = x[:, :-1, :]  # [N, cell_size, Lc]

        # Reshape to [N, 1, L]
        x = x.permute(0, 2, 1).reshape(N, 1, Lc * self.cell_size)

        return x  # Keypoint probability map of size [N, 1, L]

class DescriptorHead1D(nn.Module):
    """
    Descriptor Head for generating feature descriptors.
    """
    def __init__(self, input_channels, descriptor_dim=256):
        super(DescriptorHead1D, self).__init__()
        self.conv = nn.Conv1d(input_channels, descriptor_dim, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=8, mode='linear', align_corners=False)

    def forward(self, x):
        # x: [N, C, L/8]
        x = self.conv(x)  # [N, descriptor_dim, L/8]
        x = self.upsample(x)  # [N, descriptor_dim, L]
        #x = F.normalize(x, p=2, dim=1)  # L2 norm along channel dimension, now performed at loss.
        return x     


class TimePoint(nn.Module):
    """
    TimePoint
    """
    def __init__(self, input_channels=1, encoder_dims=[64,64,128,128], descriptor_dim=256,
                  encoder_type='wtconv', stride=2):
        super().__init__()
        
        self.valid_encoders = ['dense', 'wtconv']
        if encoder_type == 'dense':
            self.encoder = Encoder1D(input_channels, encoder_dims)
        elif encoder_type == 'wtconv':
            self.encoder = WTConvEncoder1D(input_channels, encoder_dims, stride=stride)
        else: 
            raise ValueError(f"Invalid encoder type: {encoder_type}. Encoder should be in {self.valid_encoders}")
        
        encoder_output_channels = encoder_dims[-1]
        self.detector_head = DetectorHead1D(encoder_output_channels, cell_size=8)
        self.descriptor_head = DescriptorHead1D(encoder_output_channels, descriptor_dim)

        # Compute parameters for each component
        encoder_params = count_parameters(self.encoder)
        detector_params = count_parameters(self.detector_head)
        descriptor_params = count_parameters(self.descriptor_head)
        total_params = encoder_params + detector_params + descriptor_params
        # Print the results
        print(f"Total number of trainable parameters: {total_params}")
        print(f"Encoder parameters: {encoder_params}")
        print(f"Detector head parameters: {detector_params}")
        print(f"Descriptor head parameters: {descriptor_params}")

    def forward(self, x):
        # Input x: [N, C, L]
        N, C, L = x.shape
        features = self.encoder(x)                   # Extract features
        detection_proba = self.detector_head(features)     # Keypoint detection map [N, 1, L]
        descriptors = self.descriptor_head(features) # Descriptors [N, descriptor_dim, L]
        # upscale might pad to multiples of 8
        detection_proba = detection_proba[: ,:, :L]
        descriptors = descriptors[: ,: , :L]
        return detection_proba, descriptors
    
    def get_topk_points(self, x, kp_percent=1, nms_window=5):
        """
        Extract descriptors and keypoints from input signal.
        Args:
            x [N, C=1, L]: input batch of univariate time series
            kp_percent = percentage of the series length (L) to keep. if kp_percent>=1, returns the entire series. 
        
        
        Returns:
        
        detection: Keypoint detection map [N, L] - topK keypoints, the rest are zero out
        descriptors: Descriptors [N, descriptor_dim, L]
        sorted_topk_indices: a list of len num_kp (L*kp_percent) of keypoint's timestep in their original order.
            for instace: full_timesteps: [0, 1, 2, 3, 4, 5]
                         detection_proba: [0.1, 0.3, 0, 0, 0.9, 0]
                         if num_kp = 3, then:
                         sorted_topk_indices = [0, 1, 4]
                         
        """
        N, C, L = x.shape
        features = self.encoder(x)                   # Extract features
        detection_proba = self.detector_head(features)[: ,: , :L]     # Keypoint detection map [N, 1, L]
        descriptors = self.descriptor_head(features)[: ,: , :L] # 
        # Non-maximum suppression (input is N,L, squeeze channel dim)
        detection_proba = detection_proba.squeeze(1)
        detection_proba = non_maximum_suppression(detection_proba, window_size=nms_window)
        # get top k points
        if kp_percent < 1:
            num_kp = int(kp_percent * L)
            sorted_topk_indices, detection_proba = get_topk_in_original_order(descriptors, detection_proba, K=num_kp)
        else:
            sorted_topk_indices = torch.arange(L)
        return sorted_topk_indices, detection_proba, descriptors