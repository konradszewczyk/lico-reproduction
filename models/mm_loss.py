import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ManifoldMatchingLoss(nn.Module):
    def __init__(self, norm_order=2, reduction='none'):
        """Manifold matching loss from LICO

        Args:
            distance_fn (callable): Computes the distance between features
            temperature (float): Adjacent matrix temperature
        """
        super(ManifoldMatchingLoss, self).__init__()
        self.norm_order = norm_order
        self.reduction = reduction
        # Trainable temperature
        self.temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float16))
    
    def create_adjacent_matrix(self, feats):
        """Create adjacent matrix from a matrix of features

        Args:
            feats (torch.tensor): Features
        
        Returns:
            torch.tensor: Adjacent matrix
        """
        
        batch_size = feats.shape[0]
        # Flatten
        feats = feats.view(batch_size, -1)
        # Pairwise distance
        pairwise_diff = feats.unsqueeze(0) - feats.unsqueeze(1)
        pairwise_dist = torch.linalg.vector_norm(pairwise_diff, ord=self.norm_order, dim=-1)
        pre_softmax = -pairwise_dist / self.temperature
        A = F.log_softmax(pre_softmax, dim=1)
        
        return A

    def forward(self, image_feats, lang_feats):
        """Forward function for loss

        Args:
            image_feats (torch.tensor): image features, denoted as F in the LICO paper
            - assumes the image features are flattened
            - shape should be (batch_size, num_channels, d_prime)
            lang_feats (torch.tensor): language features, denoted as G in the LICO paper
            - assumes the text features are mapped to dimensionality of image features (d_prime)
            - shape should be (batch_size, prompt_size, d_prime)

        Returns:
            torch.float32: Manifold matching loss value
        """
        # image_feats is F with shape (batch_size, num_channels, d_prime)
        # lang_feats is G with shape (batch_size, prompt_size, d_prime)
        assert len(image_feats.shape) == 3
        assert len(lang_feats.shape) == 3

        # Adjacent matrices (eq. 1)
        A_f = self.create_adjacent_matrix(image_feats)
        A_g = self.create_adjacent_matrix(lang_feats)

        # MM loss
        # - KL(A_g || A_f) is input as kl_div(A_f, A_g) according to https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        # - "this loss expects the argument input in the log-space"
        mm_loss = F.kl_div(A_f, A_g, log_target=True, reduction=self.reduction)
        mm_loss = mm_loss.sum(dim=1)
        return mm_loss


if __name__ == '__main__':
    #torch.cuda.set_device(4)
    # y = torch.randn(10, 5).cuda()
    # t = torch.randn(10, 5).cuda()

    features_visual = torch.randn(20, 3, 10).cuda().half()
    features_text = torch.randn(20, 2, 10).cuda().half()

    criterion = ManifoldMatchingLoss(norm_order=2)
    loss = criterion(features_visual, features_text)
    print(loss)
