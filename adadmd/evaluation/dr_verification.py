"""Inference-time density-ratio verification.

Generates N candidate images per prompt and selects the best one
according to the density-ratio verifier r_psi.
"""

import torch


@torch.no_grad()
def verified_generation(generator, dr_network, z_shape, n_candidates=4,
                        labels=None, device='cuda'):
    """Generate multiple candidates and select best via density ratio.

    Args:
        generator: One-step generator model
        dr_network: Trained density-ratio network
        z_shape: Shape of noise input (C, H, W)
        n_candidates: Number of candidates to generate
        labels: Optional class labels
        device: Device

    Returns:
        Best image according to density ratio (highest r = most real-looking)
    """
    # Generate N candidates
    z = torch.randn(n_candidates, *z_shape, device=device)
    if labels is not None:
        y = labels.expand(n_candidates)
    else:
        y = None

    candidates = generator(z, y)

    # Score each candidate with density ratio (at t=0)
    t_zero = torch.zeros(n_candidates, dtype=torch.long, device=device)
    scores = dr_network(candidates, t_zero)

    # Select highest scoring (most "real-looking")
    best_idx = scores.argmax()
    return candidates[best_idx:best_idx+1], scores


@torch.no_grad()
def batch_verified_generation(generator, dr_network, batch_size, z_shape,
                               n_candidates=4, labels=None, device='cuda'):
    """Generate batch of images with verification.

    Args:
        batch_size: Number of final images to produce
        n_candidates: Candidates per image
        Returns: (batch_size, C, H, W) tensor of best images
    """
    all_images = []
    all_scores = []

    for i in range(batch_size):
        y = labels[i:i+1] if labels is not None else None
        img, scores = verified_generation(
            generator, dr_network, z_shape, n_candidates, y, device
        )
        all_images.append(img)
        all_scores.append(scores.max().item())

    return torch.cat(all_images), all_scores
