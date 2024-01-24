import torch
from torch import nn
from torch.nn import functional as F
from models.mm_loss import ManifoldMatchingLoss
from models.sinkhorn_distance import SinkhornDistance


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
    def __init__(self, alpha=10., beta=1., reduction='none', train_mm_temperature=True):
        super(LICOLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mm_loss = ManifoldMatchingLoss(reduction='none',
                                            implementation='ours',
                                            train_temperature=train_mm_temperature)
        self.ot_loss = SinkhornDistance(eps=1e-4, max_iter=100, reduction='none', normalise_features=True)

    def forward(self, predictions, targets, features_visual, features_text):
        # features_visual is F with shape (batch_size, num_channels, d_prime)
        # features_text is G with shape (batch_size, prompt_size, d_prime)
        assert len(features_visual.shape) == 3
        assert len(features_text.shape) == 3

        mm_part = torch.tensor(0., device=features_visual.device)
        ot_part = torch.tensor(0., device=features_visual.device)
        total_loss = self.ce_loss(predictions, targets)
        if self.alpha != 0:
            mm_loss = self.mm_loss(features_visual, features_text)
            mm_part = self.alpha * mm_loss
        if self.beta != 0:
            ot_loss, _, _ = self.ot_loss(features_visual, features_text)
            ot_part = self.beta * ot_loss

        total_loss = total_loss + mm_part + ot_part

        if self.reduction == 'mean':
            total_loss = total_loss.mean()
            mm_part = mm_part.mean()
            ot_part = ot_part.mean()
        elif self.reduction == 'sum':
            total_loss = total_loss.sum()
            mm_part = mm_part.sum()
            ot_part = ot_part.sum()
        return total_loss, mm_part, ot_part


if __name__ == '__main__':
    y = torch.randn(10, 5).cuda()
    t = torch.randn(10, 5).argmax(dim=-1).cuda()

    features_visual = torch.randn(10, 512, 49).cuda().half()
    features_text = torch.randn(10, 5, 49).cuda().half()

    criterion = LICOLoss(alpha=10.0, beta=1.0)
    total_loss, mm_loss, ot_loss = criterion(y, t, features_visual, features_text)

    print(total_loss)
    print(mm_loss)
    print(ot_loss)
