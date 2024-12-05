import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=0.30):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, output, target):
        cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")(output, target)
        pt = torch.exp(-cross_entropy_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * cross_entropy_loss

        return loss.mean()


def get_binary_loss():
    loss_functions = {
        "Binary Cross-Entropy Loss": nn.CrossEntropyLoss(),
        "Hinge Loss": nn.HingeEmbeddingLoss(),
        "Focal Loss": FocalLoss()
    }

    return loss_functions
