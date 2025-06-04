import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from TimePoint.utils.kp_utils import non_maximum_suppression

def plot_signal_with_poi(signal, pois, title=""):
    """
    Plot signals with their Points of Interest overlaid.
    """
    # if torch cuda
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()
    plt.figure(figsize=(12, 2))
    plt.plot(signal, label='Signal')
    poi_indices = np.where(pois == 1)[0]
    plt.scatter(poi_indices, signal[poi_indices], color='red', label='POIs', s=20)
    plt.legend()
    plt.ylabel("Amplitude")
    plt.grid()
    plt.title(title)


def visualize_keypoints(signals, timepoint, threshold=0.5, max_plots=5, window_size=7, savepath="./figures"):
    """
    Visualizes a batch of signals with estimated keypoints and their descriptors from the model.
    
    Parameters:
        signals (torch.Tensor): Input signals of shape (batch_size, 1, length).
        model (nn.Module): Trained SuperPoint1D model.
        threshold (float): Threshold for keypoint detection probabilities.
        max_plots (int): Maximum number of signals to plot from the batch.
    """
    # add batch dim
    if len(signals.shape) == 2:
        signals = signals.unsqueeze(0)

    timepoint.eval()  # Set model to evaluation mode
    N, C, L = signals.shape
    N = min(N, max_plots)
    signals = signals[:N]
        
    with torch.no_grad():
        # Forward pass through the model
        detection, descriptors = timepoint(signals)
        
        # Ensure detection has shape (batch_size, length)
        detection = detection.squeeze(1)  # Shape: (batch_size, length)
        detection_prob = detection.cpu().numpy()
        
        # Process descriptors
        # Assuming descriptors have shape (batch_size, descriptor_dim, length)
        if descriptors.dim() == 4:
            # If descriptors have an extra dimension, squeeze it
            # Shape becomes (batch_size, descriptor_dim, length)
            descriptors = descriptors.squeeze(1)
        elif descriptors.dim() == 3:
            # Descriptors already have shape (batch_size, descriptor_dim, length)
            pass
        else:
            raise ValueError(f"Unexpected descriptors shape: {descriptors.shape}")
        
        descriptors_np = descriptors.cpu().numpy()  # Convert to NumPy array
    
    signals_np = signals.squeeze(1).cpu().numpy()  # Shape: (batch_size, length)
    
    batch_size, length = detection_prob.shape
    num_plots = min(batch_size, max_plots)

    # Perform NMS
    # (N, L)
    detection_prob = non_maximum_suppression(detection_prob, window_size)
    # (N,C, L) -> (N, L, C)
    descriptors_np = descriptors_np.transpose(0,2,1).reshape(-1, descriptors_np.shape[1])
    # PCA on descriptors
    pca = PCA(n_components=5)
    descriptors_np = pca.fit_transform(descriptors_np)
    descriptors_np = descriptors_np.reshape(N, L, -1).transpose(0,2,1)


    #if N == 1:
        #detection_prob = detection_prob.unsqueeze(0)
    for i in range(num_plots):
        signal = signals_np[i]  # Shape: (length,)
        probs = detection_prob[i]
        descriptor = descriptors_np[i]  # Shape: (descriptor_dim, length)
        
        # Find keypoints based on the threshold
        keypoint_indices = np.where(probs >= threshold)[0]       
        keypoint_values = signal[keypoint_indices]
        
        # Create a figure with two subplots vertically stacked
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # Plot the signal and keypoints on the first subplot
        axes[0].plot(signal, label='Signal')
        axes[0].scatter(keypoint_indices, keypoint_values, color='red', s=20, label='Predicted Keypoints', zorder=5)
        axes[0].set_title(f'Time series with Predicted Keypoints', fontsize=16)
        axes[0].set_ylabel('Amplitude', fontsize=16)
        axes[0].legend(loc='upper right', fontsize=14)
        axes[0].grid(True)
        # Plot the keypoint proba on the second subplot
        axes[1].hlines(threshold, 0, L, color='red', linestyles="dashed", label="Threshold")
        x_axis = np.arange(L)
        im = axes[1].bar(x_axis, probs, width=2, label='Keypoint Probability')
        axes[1].set_title('Keypoint Probability', fontsize=14)
        #axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Keypoint probability', fontsize=14)
        axes[1].legend(fontsize=14)
        axes[1].grid()
        # Plot the descriptors on the second subplot
        im = axes[2].imshow(descriptor, aspect='auto', cmap='viridis', interpolation='None')
        axes[2].set_title('Descriptors', fontsize=14)
        axes[2].set_xlabel('Time', fontsize=16)
        axes[2].set_ylabel('Descriptors PCA', fontsize=14)


        #fig.colorbar(im, ax=axes[1], orientation='vertical')
        
        plt.tight_layout()
        plt.show()
        # fpath = savepath + f"_idx_{i}"
        # fig.savefig(fpath + ".png", dpi=200)
        #fig.savefig(savepath + ".pdf", dpi=200)