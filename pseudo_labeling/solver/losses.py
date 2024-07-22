import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#def AffinityLoss(pred_mask_logits, tau=0.5, reduction="mean"):
#    """
#    Custom loss implementation for mask affinity.
#    """
#    # Get number, height, and width of predictions
#    N, H, W = pred_mask_logits.shape
#
#    # Calculate the refined pixel probabilities
#    pred_probs = torch.sigmoid(pred_mask_logits)
#    refined_probs = torch.zeros_like(pred_probs)
#
#    # Define a 3x3 averaging kernel for a single channel
#    kernel = torch.ones((1, 1, 3, 3), device=pred_mask_logits.device) / 9.0
#
#    for i in range(N):
#        gi = pred_probs[i, :, :]  # Shape: [H, W]
#        gi = gi.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
#
#        # Apply convolution with the kernel
#        refined_gi = F.conv2d(gi, kernel, padding=1)
#        refined_gi = refined_gi.squeeze(0).squeeze(0)  # Shape: [H, W]
#
#        # Calculate refined probabilities
#        refined_probs[i] = 0.5 * (gi.squeeze(0) + refined_gi)
#
#    # Use refined g_i and transposed g_j to get affinities
#    refined_probs = refined_probs.unsqueeze(1)  # Now refined_probs is [N, 1, H, W]
#    permuted_refined_probs = refined_probs.permute(0, 2, 3, 1)  # Shape: [N, H, W, 1]
#
#    affinities = refined_probs * permuted_refined_probs + (1 - refined_probs) * (1 - permuted_refined_probs)
#    
#    # Mask for affinities greater than the threshold tau
#    affinity_mask = (affinities > tau).float()
#
#    # Calculate log probabilities
#    pred_probs_unsqueezed = pred_probs.unsqueeze(1)  # Shape: [N, 1, H, W]
#    permuted_pred_probs = pred_probs_unsqueezed.permute(0, 2, 3, 1)  # Shape: [N, H, W, 1]
#
#    log_probs = torch.log(pred_probs_unsqueezed * permuted_pred_probs + 1e-6)
#    log_inv_probs = torch.log((1 - pred_probs_unsqueezed) * (1 - permuted_pred_probs) + 1e-6)
#
#    # Compute the loss
#    affinity_matrix = -affinity_mask * (log_probs + log_inv_probs)
#
#    print(affinity_matrix)
#
#    affinity_loss = torch.sum(affinity_matrix, dim=(1, 2, 3)) / torch.sum(affinity_mask, dim=(1, 2, 3))
#
#    if reduction == "mean":
#        affinity_loss = affinity_loss.mean()
#
#    return affinity_loss

def calculate_affinity_mask(gt_logits_tensor, tau=0.5):
    """
    Custom loss implementation for mask affinity.
    """
    # Get number, height, and width of predictions
    N, H, W = gt_logits_tensor.shape

    # Calculate the refined pixel probabilities using pseudo masks
    gt_probs = torch.sigmoid(gt_logits_tensor)
    refined_probs = torch.zeros_like(gt_probs)

    # Define a 3x3 averaging kernel for a single channel
    kernel = torch.ones((1, 1, 3, 3), device=gt_logits_tensor.device) / 9.0

    for i in range(N):
        gi = gt_probs[i, :, :]  # Shape: [H, W]
        gi = gi.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

        # Apply convolution with the kernel
        refined_gi = F.conv2d(gi, kernel, padding=1)
        refined_gi = refined_gi.squeeze(0).squeeze(0)  # Shape: [H, W]

        # Calculate refined probabilities
        refined_probs[i] = 0.5 * (gi.squeeze(0) + refined_gi)

        # refined_probs shape: [N, H, W]

    # Compute affinity between neighboring pixels using refined pseudo mask probabilities
    # Using broadcasting to avoid explicit transposition
    refined_probs_expanded_1 = refined_probs.unsqueeze(3)  # Shape: [N, H, W, 1]
    refined_probs_expanded_2 = refined_probs.unsqueeze(2)  # Shape: [N, H, 1, W]
    affinities = refined_probs_expanded_1 * refined_probs_expanded_2 + \
                 (1 - refined_probs_expanded_1) * (1 - refined_probs_expanded_2)

    # Mask for affinities greater than the threshold tau
    affinity_mask = (affinities > tau).float()
    print(f"Affinity mask shape before reduction: {affinity_mask.shape}")
    print(torch.unique(affinity_mask))

    # Reduce dimensions by taking the max value along one of the axes
    affinity_mask = affinity_mask.max(dim=3)[0]
    print(f"Affinity mask shape after reduction: {affinity_mask.shape}")
    print(torch.unique(affinity_mask))

    return affinity_mask
    
def affinity_loss(pred_mask_logits, gt_masks, tau=0.5, reduction="mean"):
    """
    something
    """
    N, H, W = pred_mask_logits.shape
    gt_probs = torch.sigmoid(gt_masks)
    refined_probs = torch.zeros_like(gt_probs)
    kernel = torch.ones((1, 1, 3, 3), device=gt_masks.device) / 9.0

    for i in range(N):
        gi = gt_probs[i, :, :].unsqueeze(0).unsqueeze(0)
        refined_gi = F.conv2d(gi, kernel, padding=1).squeeze(0).squeeze(0)
        refined_probs[i] = 0.5 * (gi.squeeze(0) + refined_gi)

    refined_probs_expanded_1 = refined_probs.unsqueeze(3)
    refined_probs_expanded_2 = refined_probs.unsqueeze(2)
    affinities = refined_probs_expanded_1 * refined_probs_expanded_2 + \
                 (1 - refined_probs_expanded_1) * (1 - refined_probs_expanded_2)

    affinity_mask = (affinities > tau).float()

    pred_probs = torch.sigmoid(pred_mask_logits)
    log_probs = torch.log(pred_probs + 1e-6)
    log_inv_probs = torch.log(1 - pred_probs + 1e-6)
    log_affinities = log_probs.unsqueeze(3) + log_probs.unsqueeze(2) + \
                     log_inv_probs.unsqueeze(3) + log_inv_probs.unsqueeze(2)

    affinity_loss = -affinity_mask * log_affinities

    if reduction == "mean":
        return affinity_loss.mean()
    elif reduction == "sum":
        return affinity_loss.sum()
    else:
        return affinity_loss
    
def dice_loss(pred_mask_logits, gt_masks, smooth=1):
    """
    Detials
    """
    pred = torch.sigmoid(pred_mask_logits)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = gt_masks.view(gt_masks.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()