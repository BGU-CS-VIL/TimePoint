import torch
import torch.nn.functional as F 
import numpy as np 

def get_topk_in_original_order(X_desc, X_probas, K):
    """
    Get the descriptors of the top K keypoints from X_keypoints without changing their original order.

    Args:
        X_desc (torch.Tensor): The descriptors associated with keypoints, shape [N, C, L].
        X_keypoints (torch.Tensor): Tensor of keypoint probabilities, shape [N, L].
        K (int): Number of top elements to select per sample.

    Returns:
        X_topk (torch.Tensor): Tensor containing the descriptors of the top K keypoints per sample, shape [N, C, K].
    """
    N, C, L = X_desc.shape
    assert X_probas.shape == (N, L), "X_keypoints must have shape (N, L)"
    
    device = X_probas.device
    if K >= L:
        return X_probas, X_desc

    # Get the indices of the top K values per sample
    topk_values, topk_indices = torch.topk(X_probas, K, dim=1)
    # topk_indices: shape [N, K]
    
    # Sort the indices per sample to maintain original order
    sorted_topk_indices, _ = torch.sort(topk_indices, dim=1)
    # sorted_topk_indices: shape [N, K]
    
    # Expand indices for gathering
    indices_expanded = sorted_topk_indices.unsqueeze(1).expand(-1, C, -1)  # Shape: [N, C, K]
    
    # Gather descriptors along the L dimension (time steps)
    X_topk = torch.gather(X_desc, dim=2, index=indices_expanded)  # Shape: [N, C, K]
    
    return sorted_topk_indices, X_topk




def non_maximum_suppression(detection_prob, window_size=7):
    """
    Apply non-maximum suppression to the detection map.

    Args:
        detection_map: Tensor of shape [N, L].
        window_size: Size of the window for NMS.
        threshold: Detection threshold.

    Returns:
        keypoints: Tensor of shape [N, L], boolean mask of keypoints after NMS.
    """
    # NMS
    if isinstance(detection_prob, np.ndarray):
        detection_prob = torch.from_numpy(detection_prob)
    # prepare input
    N, L = detection_prob.shape
    # (1, L' < L)
    pooled, pooled_idx = F.max_pool1d(detection_prob, kernel_size=window_size,
                                      stride=window_size, padding=window_size // 2,
                                      return_indices=True)

    # Squeeze dim=1 from proba, make our life easier if only one sample
    if len(pooled.shape) == 3:
        detection_prob = detection_prob.squeeze()
        pooled_idx = pooled_idx.squeeze()
    # pooled_idx array of ints, turn to bool
    zero_out = torch.ones_like(detection_prob)
    for i in range(N):
        zero_out[i, pooled_idx[i]] = 0
    # zero out everything but max pooled
    detection_prob[zero_out.type(torch.bool)] = 0
    return detection_prob
