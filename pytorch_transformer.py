import numpy as np
import sys
import itertools
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import *
from torch import nn
import torch.optim as optim

class BKT_RNN(nn.Module):
    def __init__(self, x_size = 1, hidden_size = 4):
        super(BKT_RNN, self).__init__()
        self.prior = nn.Parameter(torch.Tensor([0.1]).cuda())
        self.preprocess = nn.Sequential(nn.Linear(x_size, 512), nn.ReLU()).cuda()
        self.rnn = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8).cuda(), num_layers=6).cuda()
        self.postprocess = nn.Sequential(nn.Linear(512, hidden_size), nn.Sigmoid()).cuda()
        self.optimizer = optim.Adam(itertools.chain([self.prior], self.rnn.parameters(), 
                                    self.preprocess.parameters(), self.postprocess.parameters()))
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

    def mask_ahead(self, seq_len):
        mask = (torch.triu(torch.ones((seq_len, seq_len)).cuda()) == 1)
        mask = mask.transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, y):
        corrects = torch.zeros_like(y, dtype=torch.float32, requires_grad = False).to(x.device)
        latents = torch.zeros_like(y, dtype=torch.float32, requires_grad = False).to(x.device)
        x = self.preprocess(x)
        output = self.rnn(x)
        params = self.postprocess(output)
        loss = 0
        for i in range(len(x)):
            correctsi, latentsi = self.extract_latent_correct(params[i], latentsi if i > 0 else self.sigmoid(self.prior))
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

def preprocess_data(data):
    ohe_data = ohe.transform(data[['template_id', 'assignment_id', 'first_action', 'skill_id']])
    ohe_columns = [f'ohe{i}' for i in range(len(ohe_data[0]))]
    ohe_data = pd.DataFrame(ohe_data, index = data.index, columns = ohe_columns)
    data = data.join(ohe_data)
    data['response_time'] = data['ms_first_response'] / 10000
    features = ['correct', 'response_time', 'attempt_count', 'hint_count'] + ohe_columns
    seqs = data.groupby(['user_id', 'skill_name'])[features].apply(lambda x: x.values.tolist())
    return seqs


def construct_batches(raw_data):
    lengths = raw_data.groupby(['user_id', 'skill_name']).size().reset_index().rename(columns = {0: 'length'})
    vc = lengths['length'].value_counts()
    for lens in vc[vc >= batch_size].index:
        if lens > 5:
            relevant_lengths = lengths[lengths['length'] == lens]
            filtered_data = raw_data.merge(relevant_lengths, on = ['user_id', 'skill_name']).sort_values(['user_id', 'skill_name', 'order_id'])
            for b in range(len(filtered_data) // (batch_size * lens)):
                l, u = b * batch_size * lens, (b + 1) * batch_size * lens
                batch_preprocessed = preprocess_data(filtered_data.iloc[l:u])
                batch = np.array(batch_preprocessed.to_list())
                batch = np.transpose(batch, (1, 0, 2))
                X = torch.tensor(batch[:-1], requires_grad=True, dtype=torch.float32).cuda()
                y = torch.tensor(batch[1:,...,0:1], requires_grad=True, dtype=torch.float32).cuda()
                yield [X, y]

def train(model, batches_train, batches_val, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        batches_train = construct_batches(data_train)
        for X, y in batches_train:
            model.update(X, y)
        if epoch % 1 == 0:
            batches_val = construct_batches(data_val)
            model.eval()
            print(f"Epoch {epoch}/{num_epochs} - [VALIDATION AUC: {model.score_auc(batches_val)}]")
            torch.save(model.state_dict(), f"ckpts/model-{tag}-{epoch}.pth")

def train_test_split(data, skill_list = None):
    np.random.seed(42)
    if skill_list is not None:
        data = data[data['skill_name'].isin(skill_list)]
    data = data.set_index(['user_id', 'skill_name'])
    idx = np.random.permutation(data.index.unique())
    train_idx, test_idx = idx[:int(train_split * len(idx))], idx[int(train_split * len(idx)):]
    data_train = data.loc[train_idx].reset_index()
    data_val = data.loc[test_idx].reset_index()
    return data_train, data_val
    
data = pd.read_csv('as.csv', encoding = 'latin')
tag = sys.argv[1]
num_epochs = 500
batch_size = 32
train_split = 0.8
"""
Equation Solving Two or Fewer Steps              24253
Percent Of                                       22931
Addition and Subtraction Integers                22895
Conversion of Fraction Decimals Percents         20992
"""

data_train, data_val = train_test_split(data)
print("Train-test split complete...")
ohe = OneHotEncoder(sparse = False, handle_unknown='ignore')
ohe.fit(data_train[['template_id', 'assignment_id', 'first_action', 'skill_id']])
print("OHE complete...")

batches_train = construct_batches(data_train)
batches_val = construct_batches(data_val)
model = BKT_RNN(x_size = 4367)

print("Beginning training...")
train(model, data_train, data_val, num_epochs)
