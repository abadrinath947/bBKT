import numpy as np
import torch
from torch import nn
import torch.optim as optim

class BKT_RNN(nn.Module):
    def __init__(self, x_size = 1, hidden_size = 1):
        super(BKT_RNN, self).__init__()
        self.Whh = torch.empty(size = (hidden_size, hidden_size))
        self.Wxh = torch.empty(size = (x_size, hidden_size))
        self.bh = torch.empty(size = (hidden_size,))

        self.Wy = torch.empty(size = (hidden_size, 4))
        self.by = torch.empty(size = (4,))
        self.prior = 0

        self.params = [self.Whh, self.Wxh, self.bh, self.Wy, self.by, self.prior]
        self.optim = optim.Adam(self.params, lr=0.001, momentum=0.9)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        Wxh, Whh, bh, Wy, by, prior = self.params
        h = torch.zeros_like(bh)
        corrects = torch.zeros_like(y)
        latents = torch.zeros_like(y)
        for i in range(len(x)):
            params = torch.sigmoid(Wy @ h + by)
            latents[i], corrects[i] = self.extract_latent_correct(params, latents[i - 1] if i > 0 else prior)
            h = torch.sigmoid(Wxh @ x[i] + Whh @ h + bh)
        loss = self.loss(y, corrects)
        return corrects, latents, loss

    def extract_latent_correct(params, latent):
        l, f, g, s = params[0], params[1], params[2], params[3]
        correct = latent * (1 - s) + (1 - latent) * g
        k_t1 = (latent * (1 - s)) / (latent * (1 - s) + (1 - latent) * g)
        k_t0 = (latent * s) / (latent * s + (1 - latent) * (1 - g))
        m_t = k_t1 * correct + k_t0 * (1 - correct)
        return correct, m_t * (1 - f) + (1 - m_t) * l
