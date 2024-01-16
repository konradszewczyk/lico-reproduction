import numpy as np
import torch
import torch.nn as nn


class ManifoldMatchingLoss(nn.Module):
    def __init__(self, distance_fn):
        """Manifold matching loss from LICO

        Args:
            distance_fn (callable): Computes the distance between features
            temperature (float): Adjacent matrix temperature
        """
        super(ManifoldMatchingLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.softmax = nn.Softmax(dim=1)
        self.distance_fn = distance_fn
        # Trainable temperature
        self.temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    
    def create_adjacent_matrix(self, feats):
        """Create adjacent matrix from a matrix of features

        Args:
            feats (torch.tensor): Features
        
        Returns:
            torch.tensor: Adjacent matrix
        """
        batch_size = feats.shape[0]
        exponents_mat = torch.zeros(batch_size, batch_size)
        # Compute input to softmax
        for i in range(batch_size):
            for j in range(batch_size):
                dist_ij = self.distance_fn(feats[i], feats[j])
                exponent = -dist_ij / self.temperature
                exponents_mat[i, j] = exponent
        # Softmax across dim=1
        A = self.softmax(exponents_mat)
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
        # - "this loss expects the argument input in the log-space" -> hence A_f.log()
        # - KL loss already performs mean reduction across minibatch
        mm_loss = self.kl_loss(A_f.log(), A_g)
        return mm_loss


if __name__ == '__main__':
    #torch.cuda.set_device(4)
    # y = torch.randn(10, 5).cuda()
    # t = torch.randn(10, 5).cuda()

    features_visual = torch.randn(10, 10).cuda()
    features_text = torch.randn(10, 10).cuda()
    
    def euclidean_distance(x, y):
        return torch.norm(x - y, p=2)

    criterion = ManifoldMatchingLoss(distance_fn=euclidean_distance)
    loss = criterion(features_visual, features_text)
    print(loss)