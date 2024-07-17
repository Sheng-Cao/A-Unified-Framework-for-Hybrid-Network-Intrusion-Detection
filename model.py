import torch.nn as nn
import torch.nn.functional as F

class Embedding_Net(nn.Module):
    def __init__(self, opt):
        super(Embedding_Net, self).__init__()

        self.fc1 = nn.Linear(opt.resSize, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.outzSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

    def forward(self, features):
        embedding= self.relu(self.fc1(features))
        out_z = F.normalize(self.fc2(embedding), dim=1)
        return embedding, out_z