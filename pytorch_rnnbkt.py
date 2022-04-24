import numpy as np
import itertools
import pandas as pd
import torch
from sklearn.metrics import *
from torch import nn
import torch.optim as optim

class BKT_RNN(nn.Module):
    def __init__(self, x_size = 1, hidden_size = 4):
        super(BKT_RNN, self).__init__()
        self.prior = nn.Parameter(torch.Tensor([0.1]).cuda())
        self.rnn = nn.RNN(input_size = x_size, hidden_size = hidden_size).cuda()
        self.params = self.rnn.parameters()
        self.optimizer = optim.Adam(itertools.chain([self.prior], self.rnn.parameters()))
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        corrects = torch.zeros_like(y, dtype=torch.float32, requires_grad = False).to(x.device)
        latents = torch.zeros_like(y, dtype=torch.float32, requires_grad = False).to(x.device)
        output, hidden = self.rnn(x)
        params = (output + 1) / 2
        loss = 0
        for i in range(len(x)):
            latentsi, correctsi = self.extract_latent_correct(params[i], latentsi if i > 0 else self.sigmoid(self.prior))
            loss = loss + self.loss(correctsi, y[i])
            latents[i], corrects[i] = latentsi, correctsi
        loss = loss / len(x)
        return corrects, latents, loss

    def extract_latent_correct(self, params, latent):
        l, f, g, s = params[..., 0, None], params[..., 1, None], params[..., 2, None], params[..., 3, None]
        correct = latent * (1 - s) + (1 - latent) * g
        k_t1 = (latent * (1 - s)) / (latent * (1 - s) + (1 - latent) * g)
        k_t0 = (latent * s) / (latent * s + (1 - latent) * (1 - g))
        m_t = k_t1 * correct + k_t0 * (1 - correct)
        next_latent = m_t * (1 - f) + (1 - m_t) * l
        return correct, next_latent

    def update(self, x, y):
        corrects, latents, loss = self(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def score_acc(self, batches):
        num_correct, num_total = 0, 0
        for X, y in batches_val:
            corrects, _, _ = model.forward(X, y)
            k, n = (corrects.round() == y).float().sum(), torch.numel(corrects) 
            num_correct, num_total = num_correct + k, num_total + n
        return num_correct, num_total

    def score_auc(self, batches):
        ypred, ytrue = [], []
        for X, y in batches:
            corrects, _, _ = model.forward(X, y)
            ypred.append(corrects.ravel().detach().cpu().numpy())
            ytrue.append(y.ravel().detach().cpu().numpy())
        ypred = np.concatenate(ypred)
        ytrue = np.concatenate(ytrue)
        return roc_auc_score(ytrue, ypred)

def construct_batches(seqs):
    batches = []
    for lens in seqs.str.len().unique():
        if lens > 5:
            batch = pd.DataFrame(seqs[seqs.str.len() == lens].to_list()).to_numpy()
            batch = batch.T[..., None]
            X, y = torch.tensor(batch[:-1], requires_grad=True, dtype=torch.float32).cuda(), torch.tensor(batch[1:], requires_grad=True, dtype=torch.float32).cuda()
            batches.append([X, y])
    return batches

def train(model, batches_train, batches_val, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for X, y in batches_train:
            model.update(X, y)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{num_epochs} - [VALIDATION ACCURACY: {model.score_acc(batches_val)}, VALIDATION AUC: {model.score_auc(batches_val)}]")
            torch.save(model.state_dict(), f"ckpts/model-{epoch}.pth")

data = pd.read_csv('as.csv', encoding = 'latin')
data = data[data['skill_name'] == 'Equation Solving Two or Fewer Steps']
seqs = data.groupby('user_id')['correct'].agg(list)
seqs = seqs.sample(frac = 1, random_state = 42)
seqs_train = seqs.iloc[100:]
seqs_val = seqs.iloc[:100]
print(seqs_train.shape, seqs_val.shape)
batches_train = construct_batches(seqs_train)
batches_val = construct_batches(seqs_val)
model = BKT_RNN()
num_epochs = 5000

train(model, batches_train, batches_val, num_epochs)
