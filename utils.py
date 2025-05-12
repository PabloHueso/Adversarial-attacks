import torch
import torch.func

def flat(x):            # x : (B,C,H,W)  contiguous
    return x.reshape(x.size(0), -1)           # (B, C*H*W)

def unflat(v, C, H, W):  # v : (B, C*H*W)
    return v.reshape(v.size(0), C, H, W)      # (B,C,H,W)


def jacobian_batch(model, x_batch): 
    # Compute jacobian at a point (w.r.t. x only)
    B = len(x_batch) # batch size
    J_all = torch.func.jacrev(model)(x_batch) # yields torch.Size([B, nb_classes, B, C, H, W])
    nb_classes = J_all.shape[1]
    J_diag = J_all.diagonal(dim1=0, dim2=2)   # (B,nb_classes,C,H,W)
    J_diag = J_diag.reshape(B, nb_classes, -1)        # (B,nb_classes,CxHxW)
    return J_diag

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

def ocf_attack(model, batch, attacked_class=2, nb_steps=1, total_budget=5):
    B, C, H, W = batch.shape # Batch size, number of channels, height and width respectively
    adversarial_batch = batch
    for i in range(nb_steps):
        jacobian_b = jacobian_batch(model, adversarial_batch) # jacobian_b has shape [B, nb_classes, CxHxW] # jacobian needs B, this could be streamlined.
        J_pinv = torch.linalg.pinv(jacobian_b) # [B, CxHxW, nb_classes]
        flips = flipping_vector(model, adversarial_batch, attacked_class=attacked_class) # [B, c] (B flipping vectors) #this also needs B
        attack = torch.bmm(J_pinv, flips.unsqueeze(2)) # Add unsqueeze to flips; dim [B,c] -> [B, c, 1]. Attack has shape [B, CxHxW, 1]
        fro_norms = torch.linalg.matrix_norm(attack, ord='fro', dim=(-2, -1)) # Shape [B]
        fro_norms = fro_norms.view(B, 1, 1) # Shape [B] -> [B,1,1]
        attack_scaled = (attack/fro_norms) * (total_budget/nb_steps) 
        attack_matrix_batch = unflat(attack_scaled.squeeze(2), C, H, W)
        adversarial_batch = adversarial_batch + attack_matrix_batch 
    return adversarial_batch 