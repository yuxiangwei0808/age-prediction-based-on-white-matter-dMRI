import torch
import torch.nn as nn


class PrioriLoss(nn.Module):
    def __init__(self, min, max, scale=2):
        super(PrioriLoss, self).__init__()
        self.min = min
        self.max = max
        self.scale = scale

    def forward(self, pred, true):
        loss = 0
        for i in range(pred.shape[0]):
            if pred[i] < self.min or pred[i] > self.max:
                loss += torch.pow(pred[i] - true[i], 2) * self.scale
            else:
                loss += torch.pow(pred[i] - true[i], 2)
        return loss / pred.shape[0]


'''class PrioriLoss(nn.Module):
    def __init__(self, min, max):
        super(PrioriLoss, self).__init__()
        self.min = min
        self.max = max

    def forward(self, pred, true):
        loss = 0
        for i in range(pred.shape[0]):
            if pred[i] < self.min or pred[i] > self.max:
                loss += torch.abs(pred[i] - true[i])
            else:
                loss += torch.pow(pred[i] - true[i], 2)
        return loss / pred.shape[0]'''