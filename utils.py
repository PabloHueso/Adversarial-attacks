import torch
import torch.func

def jacobian_batch(model, x_batch): 
    # Compute jacobian at a point (w.r.t. x only)
    B = len(x_batch) # batch size
    J_all = torch.func.jacrev(model)(x_batch) #yields torch.Size([B, 10, B, 1, 28, 28])
    J_diag = J_all.diagonal(dim1=0, dim2=2)   # (B,10,1,28,28)
    J_diag = J_diag.reshape(B, 10, -1)        # (B,10,784)
    return J_diag

def flipping_vector(model, x_batch, attacked_class): 
    model_output = model(x_batch) # [B, c]
    B, c = model_output.shape

    assert 2 <= attacked_class <= c, "attacked_class must be in 2,...,c"

    largest_logits = torch.topk(model_output, k=attacked_class, dim=1).indices
    flipping_vector = torch.zeros(model_output.shape) # zero tensor of shape [B, c]

    # i: index of max logit per sample (argmax) -> [B]
    i = largest_logits[:, 0]

    # j: index of attacked_class-th largest logit -> [B]
    j = largest_logits[:, -1] 
    
    flipping = torch.zeros_like(model_output)  # [B, c]

    batch_indices = torch.arange(B) #vector [0, 1, ... , B-1]

    flipping[batch_indices, i] = -1
    flipping[batch_indices, j] = 1

    return flipping



