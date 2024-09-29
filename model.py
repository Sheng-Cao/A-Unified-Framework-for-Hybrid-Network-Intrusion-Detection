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
        embedding = self.relu(self.fc1(features))
        out_z = F.normalize(self.fc2(embedding), dim=1)
        return embedding, out_z


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        self.hs_dim = opt.hs_dim
        self.hu_dim = opt.hu_dim
        self.fc1 = nn.Linear(opt.resSize, 2 * opt.resSize)
        self.fc2 = nn.Linear(2 * opt.resSize, opt.hidden_dim)
        self.fc3 = nn.Linear(opt.hs_dim, opt.outzSize)

        self.mu = nn.Linear(opt.hidden_dim, opt.hs_dim + opt.hu_dim)
        self.logvar = nn.Linear(opt.hidden_dim, opt.hs_dim + opt.hu_dim)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, img):
        out = self.fc1(img)
        # out = self.dropout(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        # out = self.dropout(out)
        out = self.lrelu(out)
        mu = self.mu(out)
        # logvar = self.logvar(out)
        # sigma = torch.exp(logvar)
        # z = reparameter(mu,sigma)
        # 直接是一个autoencoder
        z = mu
        hs = z[:, :self.hs_dim]
        hu = z[:, self.hs_dim:]
        # hs_l2_real = F.normalize(self.fc3(hs), dim=1)
        hs_l2_real = F.normalize(self.fc3(hs))
        # z[:, self.hs_dim:] = 0
        return mu, hs, hu, z, hs_l2_real