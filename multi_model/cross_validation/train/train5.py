import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable
from models.SpectralCRNN_hybrid import SpectralCRNN_Reg_Dropout_CQT_ph as SpectralCRNN
from tensorboard_logger import configure, log_value
from dataLoaders.SpectralDataset_ph import SpectralDataset, SpectralDataLoader
from sklearn import metrics
from torch.optim import lr_scheduler
from scipy.stats import pearsonr as pearsoncorr
from scipy.stats import spearmanr as spearmancorr
import pdb


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate_classification(targets, predictions):
    r2 = metrics.r2_score(targets, predictions)
    predictions[predictions < -1] = -1
    predictions[predictions > 1] = 1
    pearson = pearsoncorr(targets.flatten(), predictions.flatten())
    pearson_corre = pearson[0]
    print("Pearson Corr = ", pearson)
    print("Spearman Corr = ", spearmancorr(targets.flatten(), predictions.flatten()))
    #accuracy = metrics.accuracy_score(np.round(targets).astype(int), np.round(predictions).astype(int))
    print("R2 = ", r2)
    # print("Accuracy = ", accuracy)
    return r2, pearson_corre


# Configure tensorboard logger
configure('runs/CQT_ph_reg_lr0.0001_big_ELU_Adam_noteacc_bws-1to1', flush_secs=2)  # modify

# Parameteres for Spectral Representation
rep_params = {'method': 'CQT', 'hop_length': 512, 'n_bins': 96, 'bins_per_octave': 24, 'normalize': True}

# Load Datasets
train_dataset = SpectralDataset('/data07/huanglin/SingEval/LeaderboardData/data_ph/train_5.dill', 0, rep_params)
train_dataloader = SpectralDataLoader(train_dataset, batch_size=5, num_workers=4, shuffle=True)
# pdb.set_trace()

test_dataset = SpectralDataset('/data07/huanglin/SingEval/LeaderboardData/data_ph/test_5.dill', 0, rep_params)
test_dataloader = SpectralDataLoader(test_dataset, batch_size=5, num_workers=1, shuffle=True)

valid_dataset = SpectralDataset('/data07/huanglin/SingEval/LeaderboardData/data_ph/val_5.dill', 0, rep_params)
valid_dataloader = SpectralDataLoader(valid_dataset, batch_size=5, num_workers=4, shuffle=True)

# Define Model
model = SpectralCRNN().cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

batch_time = AverageMeter()
data_time = AverageMeter()

train_loss = 0
validation_loss = 0

num_epochs = 250
best_val_r2 = 0.0
best_val_corr = 0.0
epoch_time = time.time()
for epoch in range(num_epochs):
    print('\n')
    print('Epoch:', epoch)
    print('training set: ')
    model.train()
    # scheduler.step()
    avg_loss = 0.0
    end = time.time()
    all_predictions = []
    all_targets = []
    losses = AverageMeter()
    for i, (data) in enumerate(train_dataloader):
        inputs, targets, ph_notes = data
        data_time.update(time.time() - end)
        inputs = Variable(inputs.cuda(), requires_grad=False)
        targets = Variable(targets.cuda(), requires_grad=False)
        targets = targets.view(-1, 1)
        PH = Variable(ph_notes.cuda(), requires_grad=False)
        model.init_hidden(inputs.size(0))
        out = model(inputs, PH)[0]
        all_predictions.extend(out.data.cpu().numpy())
        all_targets.extend(targets.data.cpu().numpy())
        loss = criterion(out, targets)
        loss_value = loss.data
        losses.update(loss_value, inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
    train_loss = losses.avg
    train_r2, train_pearson = evaluate_classification(np.array(all_targets), np.array(all_predictions))

    print('validation set: ')
    model.eval()
    losses = AverageMeter()
    all_predictions = []
    all_targets = []
    for i, (data) in enumerate(valid_dataloader):
        inputs, targets, ph_notes = data
        data_time.update(time.time() - end)
        inputs = Variable(inputs.cuda(), requires_grad=False)
        targets = Variable(targets.cuda(), requires_grad=False)
        targets = targets.view(-1, 1)
        PH = Variable(ph_notes.cuda(), requires_grad=False)
        model.init_hidden(inputs.size(0))
        out = model(inputs, PH)[0]
        all_predictions.extend(out.data.cpu().numpy())
        all_targets.extend(targets.data.cpu().numpy())
        loss = criterion(out, targets)
        loss_value = loss.data
        losses.update(loss_value, inputs.size(0))
    valid_loss = losses.avg
    val_r2, val_pearson = evaluate_classification(np.array(all_targets), np.array(all_predictions))
    log_value('Train Loss', train_loss, epoch)
    log_value('Validation Loss', valid_loss, epoch)
    log_value('Training R2', train_r2, epoch)
    log_value('Validation R2', val_r2, epoch)
    # if val_r2 > best_val_r2:
    #     best_val_r2 = val_r2
    #     print('save r2 model')
    #     torch.save(model, './models_hybrid/model_SpectralCRNN_reg_lr0.0001_big_ELU_Adam_noteacc_bws-1to1_CQT_ph_r2')

    if val_pearson > best_val_corr:
        best_val_corr = val_pearson
        print('save pearson corr model: ')
        torch.save(model, './models_hybrid/model_SpectralCRNN_reg_lr0.0001_big_ELU_Adam_noteacc_bws-1to1_CQT_ph_v5')




