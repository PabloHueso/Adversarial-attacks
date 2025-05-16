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
    

def ocf_attack(model, batch, attacked_class=2, nb_steps=1, total_budget=5):
    B, C, H, W = batch.shape # Batch size, number of channels, height and width respectively
    adversarial_batch = batch.clone()
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

'''
def ocf_attack_while(model, batch, attacked_class=2, max_budget=5, max_steps=1000, ):
    B, C, H, W = batch.shape # Batch size, number of channels, height and width respectively
    adversarial_batch = batch
    current_step = 1
    current_norm = torch.zeros(B)
    netFooled = False
    #Puesto que estoy trabajando con un batch de B imagenes, tengo que encontrar una forma de parar el proceso en solo algunos de las imagenes del batch.
    while ((not netFooled) and current_norm <= max_budget and current_step<=max_steps):
        jacobian_b = jacobian_batch(model, adversarial_batch) # jacobian_b has shape [B, nb_classes, CxHxW] # jacobian needs B, this could be streamlined.
        J_pinv = torch.linalg.pinv(jacobian_b) # [B, CxHxW, nb_classes]
        flips = flipping_vector(model, adversarial_batch, attacked_class=attacked_class) # [B, c] (B flipping vectors) #this also needs B
        attack = torch.bmm(J_pinv, flips.unsqueeze(2)) # Add unsqueeze to flips; dim [B,c] -> [B, c, 1]. Attack has shape [B, CxHxW, 1]
        fro_norms = torch.linalg.matrix_norm(attack, ord='fro', dim=(-2, -1)) # Shape [B]
        current_norm =+ fro_norms
        fro_norms = fro_norms.view(B, 1, 1) # Shape [B] -> [B,1,1]
        attack_scaled = (attack/fro_norms) 
        attack_matrix_batch = unflat(attack_scaled.squeeze(2), C, H, W)
        adversarial_batch = adversarial_batch + attack_matrix_batch 
        current_step =+ 1 
    return adversarial_batch 
'''