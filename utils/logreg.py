import torch
import torch.nn as nn


class LogReg(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, seq):
        return self.fc(seq)
