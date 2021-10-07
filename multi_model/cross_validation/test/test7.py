import torch
import time
import numpy as np
from torch import nn
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from torch.autograd import Variable
from models.SpectralCRNN_hybrid import SpectralCRNN_Reg_Dropout_CQT_ph as SpectralCRNN
from tensorboard_logger import configure, log_value
from dataLoaders.SpectralDataset_ph import SpectralDataset, SpectralDataLoader
from sklearn import metrics
from torch.optim import lr_scheduler
import dill

def evaluate_classification(targets, predictions):
    r2 = metrics.r2_score(targets, predictions)
    predictions[predictions < -1] = -1
    predictions[predictions > 1] = 1
    pearson = pearsonr(targets.flatten(), predictions.flatten())
    pearson_corre = pearson[0]
    print("Pearson Corr = ", pearson)
    print("Spearman Corr = ", spearmanr(targets.flatten(), predictions.flatten()))
    #accuracy = metrics.accuracy_score(np.round(targets).astype(int), np.round(predictions).astype(int))
    print("R2 = ", r2)
    # print("Accuracy = ", accuracy)
    return pearson_corre, predictions, targets


def evaluate_model(model, dataloader):
    model.eval()
    all_predictions = []
    all_targets = []
    for i, (data) in enumerate(dataloader):
        inputs, targets, ph_notes = data
        inputs = Variable(inputs.cuda(), requires_grad = False)
        targets = Variable(targets.cuda(), requires_grad = False)
        targets = targets.view(-1,1)
        PH = Variable(ph_notes.cuda(), requires_grad=False)
        model.init_hidden(inputs.size(0))
        out = model(inputs, PH)[0]
        all_predictions.extend(out.data.cpu().numpy())
        all_targets.extend(targets.data.cpu().numpy())
    return evaluate_classification(np.array(all_targets), np.array(all_predictions))

rep_params = {'method': 'CQT', 'hop_length': 512, 'n_bins': 96, 'bins_per_octave': 24, 'normalize': True}

train_dataset = SpectralDataset('/data07/huanglin/SingEval/LeaderboardData/data_ph/train_7.dill', 0, rep_params)
train_dataloader = SpectralDataLoader(train_dataset, batch_size = 5, num_workers = 4, shuffle = False)

test_dataset = SpectralDataset('/data07/huanglin/SingEval/LeaderboardData/data_ph/test_7.dill', 0, rep_params)
test_dataloader = SpectralDataLoader(test_dataset, batch_size = 5, num_workers = 1, shuffle = False)

valid_dataset = SpectralDataset('/data07/huanglin/SingEval/LeaderboardData/data_ph/val_7.dill', 0, rep_params)
valid_dataloader = SpectralDataLoader(valid_dataset, batch_size = 5, num_workers = 4, shuffle = False)

model_path = './models_hybrid/model_SpectralCRNN_reg_lr0.0001_big_ELU_Adam_noteacc_bws-1to1_CQT_ph_v7'
model = SpectralCRNN().cuda()
model = torch.load(model_path)

criterion = nn.MSELoss()

print('training set: ')
train_metrics = evaluate_model(model, train_dataloader)

print('validation set: ')
val_metrics = evaluate_model(model, valid_dataloader)

print('test set: ')
test_metrics = evaluate_model(model, test_dataloader)
#dill.dump({'machine': test_metrics[1].flatten(), 'GT': test_metrics[2].flatten()}, open('./abs_outputs/test_cqt_ph.dill', 'wb'))



