import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Based on: https://github.com/Mayurji/Pytorch-Lightning-Face-Recognition/blob/main/model.py
    """

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive