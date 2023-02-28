import torch


def MMD_loss(D, D_hat, gamma=1):
    """
    Computes MMD loss of D and D_hat with parameter gamma.
    """
    batch_size = D.shape[0]
    empirical_batch_size = D_hat.shape[0]

    D_squared = torch.mm(D, D.t())
    D_hat_squared = torch.mm(D_hat, D_hat.t())
    D_D_hat = torch.mm(D, D_hat.t())

    # Gaussian Kernel
    D_diag = (D_squared.diag().unsqueeze(0).expand_as(D_squared))
    D_squared_K = torch.exp(-gamma * (D_diag.t() + D_diag - 2 * D_squared))

    D_hat_diag = (D_hat_squared.diag().unsqueeze(0).expand_as(D_hat_squared))
    D_hat_squared_K = torch.exp(-gamma * (D_hat_diag.t() + D_hat_diag - 2 * D_hat_squared))

    D_diag_both = (D_squared.diag().unsqueeze(1).expand_as(D_D_hat))
    D_hat_diag_both = (D_hat_squared.diag().unsqueeze(0).expand_as(D_D_hat))
    D_D_hat_K = torch.exp(-gamma * (D_diag_both + D_hat_diag_both - 2 * D_D_hat))

    norm_term_D = 1.0 / (batch_size * (batch_size - 1))
    norm_term_D_hat = 1.0 / (empirical_batch_size * (empirical_batch_size - 1))
    norm_term_both = 2.0 / (batch_size * empirical_batch_size)

    loss = norm_term_D * torch.sum(D_squared_K) + norm_term_D_hat * torch.sum(D_hat_squared_K) \
        - norm_term_both * torch.sum(D_D_hat_K)
    return loss
