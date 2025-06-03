import torch
import torch.func
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm  
from autoattack import AutoAttack

def flat(x):            # x : (B,C,H,W)  contiguous
    return x.reshape(x.size(0), -1)           # (B, C*H*W)

def unflat(v, C, H, W):  # v : (B, C*H*W)
    return v.reshape(v.size(0), C, H, W)      # (B,C,H,W)

def batchnorm(tensor, p=2):
    # Assumes tensor shape is [B, C, H, W]. Returns tensor of shape [B] containing p-norm of each [C, H, W] tensor
    flat_tensor = flat(tensor)
    return torch.linalg.norm(flat_tensor, dim=1, ord=p)

def comparable_norm(tensor, lower=0, upper=1, p=2):
    # Assumes tensor shape is [B, C, H, W]. Returns tensor of shape [B] containing a comparable p-norm of each [C, H, W] tensor
    # Comparable norm: assumes that images belong to [lower, upper]^CxHxW, and normalizes the p-norm so that it belongs to the range [0, 1]
    flat_tensor = flat(tensor)
    coeff = np.power(flat_tensor.shape[1], 1/p) * np.max(np.abs(upper), np.abs(lower))
    norm = torch.linalg.norm(flat_tensor, dim=1, ord=p)
    return norm / coeff
def jacobian_batch(model, x_batch):
    # Function that takes one single image as input (no batch dim)
    f_single = lambda x: model(x.unsqueeze(0))[0]          # logits shape [C_out]

    # jacobian over single input function
    jac_single = torch.func.jacrev(f_single)               # callable

    # Batch that function
    J = torch.vmap(jac_single)(x_batch)                    # [B, C_out, C, H, W]

    return J.reshape(J.size(0), J.size(1), -1)             # [B, C_out, C*H*W]


def flipping_vector(model, x_batch, attacked_class=2, eps=1): 
    model_output = model(x_batch) # [B, c]
    B, c = model_output.shape

    assert 2 <= attacked_class <= c, "attacked_class must be in 2,...,c"

    largest_logits = torch.topk(model_output, k=attacked_class, dim=1).indices

    # i: index of max logit per sample (argmax) -> [B]
    i = largest_logits[:, 0]

    # j: index of attacked_class-th largest logit -> [B]
    j = largest_logits[:, -1] 
    

    flipping = torch.zeros_like(model_output)  # [B, c]

    batch_indices = torch.arange(B) #vector [0, 1, ... , B-1]

    flipping[batch_indices, i] = -eps
    flipping[batch_indices, j] = eps

    return flipping 

def step_estimator(model, batch, attacked_class=2):
    topk = torch.topk(model(batch), k=attacked_class)
    log_diff = topk.values[:, 0] - topk.values[:,-1] 
    return log_diff #torch.ceil(log_diff)

def ocf_attack_pure(model, batch, attacked_class=2, nb_steps=1):
    B, C, H, W = batch.shape # Batch size, number of channels, height and width respectively
    adversarial_batch = batch.clone()
    for i in range(nb_steps):
        jacobian_b = jacobian_batch(model, adversarial_batch) # jacobian_b has shape [B, nb_classes, CxHxW] # jacobian needs B, this could be streamlined.
        J_pinv = torch.linalg.pinv(jacobian_b) # [B, CxHxW, nb_classes]
        flips = flipping_vector(model, adversarial_batch, attacked_class=attacked_class) # [B, c] (B flipping vectors) #this also needs B
        attack = torch.bmm(J_pinv, flips.unsqueeze(2)) # Add unsqueeze to flips; dim [B,c] -> [B, c, 1]. Attack has shape [B, CxHxW, 1]
        attack_matrix_batch = unflat(attack.squeeze(2), C, H, W) # Now attack has shape [B, C, H, W]
        adversarial_batch += attack_matrix_batch # Add perturbation to the image batch
    return adversarial_batch
    
def ocf_attack_until_flip(model, batch, attacked_class: int = 2, max_steps: int = 50):
    device = batch.device
    B, C, H, W = batch.shape

    with torch.no_grad():
        orig_pred = model(batch).argmax(dim=1)          # [B]

    adv = batch.clone()

    # Boolean masks that track progress
    done  = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_steps):
        active = ~done                                   # images still in play
        if not active.any():
            break

        cur    = adv[active]                             # (B_active,C,H,W)
        jac    = jacobian_batch(model, cur)              # [B_act, c, n]
        J_pinv = torch.linalg.pinv(jac)                  # [B_act, n, c]
        flips  = flipping_vector(model, cur, attacked_class)  # [B_act, c]

        delta  = torch.bmm(J_pinv, flips.unsqueeze(2)).squeeze(2)  # [B_act, n]
        adv[active] += unflat(delta, C, H, W)


        # Check if active images flipped class
        with torch.no_grad():
            new_pred = model(adv[active]).argmax(dim=1)
        just_flipped      = new_pred != orig_pred[active]
        done[active]      |= just_flipped                # mark as finished

    return adv

def ocf_attack_until_flip_budget(model, batch, attacked_class = 2, max_budget = 5.0, max_steps = 50):
    device = batch.device
    B, C, H, W = batch.shape

    with torch.no_grad():
        orig_pred = model(batch).argmax(1)                # [B]

    adv   = batch.clone()
    pert  = torch.zeros_like(batch)
    active = torch.ones(B, dtype=torch.bool, device=device)

    for _ in range(max_steps):

        # Each loop step operates solely on th remaining (active) images of the batch
        cur_adv = adv[active]                           
        cur_pert = pert[active]

        jac = jacobian_batch(model, cur_adv)      # [B_active, c, n]
        Jpinv = torch.linalg.pinv(jac)                    # [B_active, n, c]
        flips = flipping_vector(model, cur_adv, attacked_class)

        delta = torch.bmm(Jpinv, flips.unsqueeze(2)).squeeze(2)  # [B_active,n]
        delta = unflat(delta, C, H, W)                       # [B_active,C,H,W]

        fut_pert = cur_pert + delta
        over = batchnorm(fut_pert) > max_budget          # bool [B_active]

        keep_mask = ~over
        if keep_mask.any():
            idx = active.nonzero(as_tuple=True)[0][keep_mask]      # global indexes of the active images in the active tensor
            adv[idx]  += delta[keep_mask]
            pert[idx] += delta[keep_mask]

        with torch.no_grad():
            new_pred = model(adv[active]).argmax(1)
        flipped = new_pred != orig_pred[active]

        # a sample stays alive only if it neither flipped nor overshot
        still_alive = ~(flipped | over)
        active[active.clone()] = still_alive

        if not active.any():
            break
    return adv

def ocf_attack(model, batch, eps= 1, attacked_class = 2, max_budget = 5.0, max_steps = 100):
    device = batch.device
    B, C, H, W = batch.shape

    with torch.no_grad():
        orig_pred = model(batch).argmax(1)                # [B]

    adv   = batch.clone()
    #pert  = torch.zeros_like(batch)
    active = torch.ones(B, dtype=torch.bool, device=device)

    for _ in range(max_steps):

        # Each loop step operates solely on th remaining (active) images of the batch
        cur_adv = adv[active]                           
        #cur_pert = pert[active]

        jac = jacobian_batch(model, cur_adv)      # [B_active, c, n]
        Jpinv = torch.linalg.pinv(jac)                    # [B_active, n, c]
        flips = flipping_vector(model, cur_adv, attacked_class, eps)

        delta = torch.bmm(Jpinv, flips.unsqueeze(2)).squeeze(2)  # [B_active,n]
        delta = unflat(delta, C, H, W)                       # [B_active,C,H,W]

        fut_pert = torch.clamp(cur_adv + delta,min=0, max=1) - batch[active]
        over = batchnorm(fut_pert) > max_budget          # bool [B_active]

        keep_mask = ~over
        if keep_mask.any():
            idx = active.nonzero(as_tuple=True)[0][keep_mask]      # global indexes of the active images in the active tensor
            adv[idx]  = torch.clamp(adv[idx] + delta[keep_mask], min=0, max=1)
            #pert[idx] += delta[keep_mask]

        with torch.no_grad():
            new_pred = model(adv[active]).argmax(1)
        flipped = new_pred != orig_pred[active]

        # a sample stays alive only if it neither flipped nor overshot
        still_alive = ~(flipped | over)
        active[active.clone()] = still_alive

        if not active.any():
            break
    return adv

# This function can be streamlined: I am calculating all the info for all the batch, which can be significantly longer than nb_examples
def attack_examples(model, images, labels, attack, nb_examples):
    img_adv = attack(model, images)
    pert = (img_adv - images)
    pert_norm = batchnorm(pert)
    pred_adv = model(img_adv).argmax(dim=1).to('cpu')
    
    pert = pert.to('cpu')
    img_adv = img_adv.to('cpu')
    images = images.to('cpu')
    labels = labels.to('cpu')

    height_per_example = 4
    fig_width = 10
    fig_height = nb_examples * height_per_example

    fig, axs = plt.subplots(nb_examples, 3, figsize=(fig_width, fig_height))


    if nb_examples == 1:
        axs = axs.reshape(1, 3)

    with torch.no_grad():
        for example in range(nb_examples):
            # Original
            im0 = axs[example, 0].matshow(images[example].squeeze(0), aspect='auto')
            axs[example, 0].set_title(f"Original\npred={labels[example].item()}", fontsize=14)
            fig.colorbar(im0, ax=axs[example, 0])

            # Perturbation
            im1 = axs[example, 1].matshow(pert[example].squeeze(0).numpy(), aspect='auto')
            axs[example, 1].set_title(f"Perturbation\nnorm={pert_norm[example].item()}", fontsize=14)
            fig.colorbar(im1, ax=axs[example, 1])

            # Adversarial
            im2 = axs[example, 2].matshow(img_adv[example].squeeze(0), aspect='auto')
            axs[example, 2].set_title(f"Adversarial\npred={pred_adv[example].item()}", fontsize=14)
            fig.colorbar(im2, ax=axs[example, 2])

    plt.tight_layout()
    plt.show()

import torch
from tqdm.auto import tqdm          # auto picks the right backend (notebook, console…)

def eval_loop(model, testloader, attack, device):
    model.eval()

    N = len(testloader.dataset)                # total samples
    norms       = torch.empty(N, device=device)
    confidences = torch.empty(N, device=device)
    labels      = torch.empty(N, dtype=torch.long, device=device)

    correct_clean = 0
    correct_adv = 0
    flipped = 0
    idx = 0      
    with torch.no_grad():
        for X, y in tqdm(testloader, desc="Evaluating", unit="batch"):
            X, y = X.to(device), y.to(device)
            B    = X.size(0)

            # --- clean ----------------------------------------------------------
            logits = model(X)
            pred   = logits.argmax(dim=1)
            correct_clean += (pred == y).sum().item()

            # confidence per sample
            conf_batch = step_estimator(model, X)

            # --- adversarial ----------------------------------------------------
            X_adv      = attack(model, X)
            logits_adv = model(X_adv)
            pred_adv   = logits_adv.argmax(dim=1)
            correct_adv += (pred_adv == y).sum().item()

            flipped += (pred_adv != pred).sum().item()

            delta      = (X_adv - X)
            norm_batch = batchnorm(delta)

            # --- store everything ----------------------------------------------
            norms[idx:idx+B] = norm_batch
            confidences[idx:idx+B] = conf_batch
            labels[idx:idx+B] = y            # save ground-truth classes
            idx += B

    return {
        "accuracy"     : correct_clean / N,
        "adv_accuracy" : correct_adv   / N,
        "flipped_pct"  : flipped       / N,
        "norms"        : norms.cpu(),
        "confidences"  : confidences.cpu(),
        "labels"       : labels.cpu()
    }

def AutoAttackConversor(model, attack, eps, device, norm='L2'):
    #Returns an attack function in format attack(model, batch), i.e. adapted to eval_loop()
    attack_obj = AutoAttack(model=model, norm=norm, eps=eps, device=device, version='custom', attacks_to_run=[attack], verbose=False)
    return lambda model, batch : attack_obj.run_standard_evaluation(x_orig=batch.to(device), y_orig=torch.zeros(batch.shape[0], dtype=torch.int64).to(device), bs=batch.shape[0])

def project_image_with_target_norm(
        img: torch.Tensor,
        k: float,
        *,
        eps: float = 1e-12
) -> torch.Tensor:
    """
    Scale a (C,H,W) image by a single scalar, clamp to [0,1], and choose the
    scalar so the clamped image has Euclidean norm k whenever feasible.
    If k exceeds the maximum attainable norm, return that maximally saturated
    image instead.

    Parameters
    ----------
    img : torch.Tensor, shape (C,H,W)
        Input image (CPU or GPU, any dtype).
    k   : float
        Desired ℓ₂-norm after clamping.
    eps : float, optional
        Numerical tolerance.

    Returns
    -------
    torch.Tensor  # shape (C,H,W)
        The projected image.
    """
    if img.ndim != 3:
        raise ValueError("Expected a 3-D tensor of shape [C,H,W].")

    flat = img.reshape(-1)                           # view (n,)
    if k <= eps or torch.all(flat <= 0):
        return torch.zeros_like(img)                 # trivial cases

    # ---- positive entries only --------------------------------------------
    pos_mask = flat > 0
    v_pos    = flat[pos_mask]                        # length m
    m        = v_pos.numel()
    if m == 0:
        return torch.zeros_like(img)

    # ---- maximum norm if all positives saturate at 1 -----------------------
    max_norm = m ** 0.5
    if k >= max_norm - eps:
        saturated = torch.zeros_like(flat)
        saturated[pos_mask] = 1.0
        return saturated.reshape_as(img)

    # ---- break-points t_i = 1 / v_i  (ascending) ---------------------------
    t_i, perm   = (1.0 / v_pos).sort()
    v_sorted    = v_pos[perm]
    v_sq        = v_sorted.pow(2)

    # suffix_sq[j] = Σ_{i=j}^{m-1} v_sorted[i]²
    suffix_sq = torch.flip(
        torch.cumsum(torch.flip(v_sq, dims=[0]), dim=0), dims=[0]
    )

    k_sq   = k * k
    s_star = None
    prev_t = 0.0

    # ---- walk the m+1 intervals -------------------------------------------
    for idx in range(m + 1):
        A_idx = suffix_sq[idx] if idx < m else torch.tensor(0.0, device=img.device)
        rhs   = k_sq - idx                      # equation: idx + A_idx s² = k²

        if rhs >= -eps and A_idx > eps:
            s_cand = torch.sqrt(torch.clamp(rhs / A_idx, min=0.0)).item()
            hi = t_i[idx] if idx < m else float("inf")
            if prev_t - eps <= s_cand <= hi + eps:
                s_star = s_cand
                break
        prev_t = t_i[idx] if idx < m else prev_t

    # ---- numerical fallback (should be rare) -------------------------------
    if s_star is None:
        saturated = torch.zeros_like(flat)
        saturated[pos_mask] = 1.0
        return saturated.reshape_as(img)

    # ---- build and return the projected image ------------------------------
    projected = torch.clamp(flat * s_star, 0.0, 1.0)
    return projected.reshape_as(img)
