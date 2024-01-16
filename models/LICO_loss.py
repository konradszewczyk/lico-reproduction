import torch
from torch import nn
from models.mm_loss import ManifoldMatchingLoss


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        # print(x.size(), y.size(), C.shape)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        # print(x.dim(), x_points, y_points)
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().cuda()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().cuda()

        u = torch.zeros_like(mu).cuda()
        v = torch.zeros_like(nu).cuda()
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-3

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            # print(i, err.item(), thresh)
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        # print(x.shape, y.shape)
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        # print(x_col.shape, y_lin.shape)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        # C.detach()
        return C

    @staticmethod
    def _cost_matrix_cosine(x, y, squared=False):
        print(x.shape, y.shape)
        prod = torch.mm(x, y.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        C = prod / norm
        eps = 1e-6
        if squared:
            C.diag == 0  # todo: what is this for?
            return C.clamp(min=eps)
        else:
            C = C.clamp(min=eps).sqrt()
            return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


class LICOLoss(nn.Module):
    """
    Calculates the LICO loss as described in https://arxiv.org/abs/2310.09821
    Args:
        alpha (float): the weight assigned to MM component of the loss
        beta (float): the weight assigned to OT component of the loss
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    """
    def __init__(self, alpha=10., beta=1., reduction='none'):
        super(LICOLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mm_loss = ManifoldMatchingLoss(norm_order=2, reduction='none')
        self.ot_loss = SinkhornDistance(eps=1e-4, max_iter=100, reduction='none')

    def forward(self, y, t, features_visual, features_text):
        # todo: are y and t - predictions and targets?
        batch_size = y.shape[0]
        
        # features_visual is F with shape (batch_size, num_channels, d_prime)
        # features_text is G with shape (batch_size, prompt_size, d_prime)
        assert len(features_visual.shape) == 3
        assert len(features_text.shape) == 3

        total_loss = self.ce_loss(y, t)
        if self.alpha != 0:
            mm_loss = self.mm_loss(features_visual, features_text)
            total_loss = total_loss + self.alpha * mm_loss
        if self.beta != 0:
            ot_loss, _, _ = self.ot_loss(features_visual, features_text)
            total_loss = total_loss + self.beta * ot_loss

        if self.reduction == 'mean':
            total_loss = total_loss.mean()
        elif self.reduction == 'sum':
            total_loss = total_loss.sum()
        return total_loss


if __name__ == '__main__':
    y = torch.randn(10, 5).cuda()
    t = torch.randn(10, 5).cuda()

    features_visual = torch.randn(10, 2, 10).cuda().half()
    features_text = torch.randn(10, 5, 10).cuda().half()

    criterion = LICOLoss(alpha=1.0, beta=1.0)
    loss = criterion(y, t, features_visual, features_text)
    print(loss)
