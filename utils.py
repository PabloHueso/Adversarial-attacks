import torch
import torch.func

def flat(x):            # x : (B,C,H,W)  contiguous
    return x.reshape(x.size(0), -1)           # (B, C*H*W)

def unflat(v, C, H, W):  # v : (B, C*H*W)
    return v.reshape(v.size(0), C, H, W)      # (B,C,H,W)


def jacobian_batch(model, x_batch):
    # Function that takes one single image as input (no batch dim)
    f_single = lambda x: model(x.unsqueeze(0))[0]          # logits shape [C_out]

    # jacobian over single input function
    jac_single = torch.func.jacrev(f_single)               # callable

    # Batch that function
    J = torch.vmap(jac_single)(x_batch)                    # [B, C_out, C, H, W]

    return J.reshape(J.size(0), J.size(1), -1)             # [B, C_out, C*H*W]


def flipping_vector(model, x_batch, attacked_class=2): 
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

    flipping[batch_indices, i] = -1
    flipping[batch_indices, j] = 1

    return flipping 

def step_estimator(model, batch, attacked_class=2):
    topk = torch.topk(model(batch), k=attacked_class)
    log_diff = topk.values[:, 0] - topk.values[:,-1] 
    return torch.ceil(log_diff)

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