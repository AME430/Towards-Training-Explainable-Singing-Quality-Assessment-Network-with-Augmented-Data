import torch
import time
import numpy as np
from torch import nn
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from torch.autograd import Variable
from models.SpectralCRNN import SpectralCRNN_Reg_Dropout_tsne as SpectralCRNN
from tensorboard_logger import configure, log_value
from dataLoaders.SpectralDataset import SpectralDataset, SpectralDataLoader
from sklearn.manifold import TSNE

def evaluate_model(model, dataloader):
    model.eval()
    #all_predictions = []
    all_targets = []
    all_feature = []
    for i, (data) in enumerate(dataloader):
        inputs, targets = data
        inputs = Variable(inputs.cuda(), requires_grad = False)
        targets = Variable(targets.cuda(), requires_grad = False)
        targets = targets.view(-1,1)
        model.init_hidden(inputs.size(0))
        feature = model(inputs)[1]
        feature_np = feature.data.cpu().numpy()
        all_feature.append(feature_np)
        all_targets.extend(targets.data.cpu().numpy())
    feature_array = np.concatenate(all_feature, axis=0)
    target_array = np.concatenate(all_targets, axis=0)
    return feature_array, target_array


rep_params = {'method': 'CQT', 'hop_length': 512, 'n_bins': 96, 'bins_per_octave': 24, 'normalize': True}

train_dataset = SpectralDataset('/data07/huanglin/SingEval/LeaderboardData/data/train_1.dill', 0, rep_params)
train_dataloader = SpectralDataLoader(train_dataset, batch_size = 5, num_workers = 4, shuffle = False)

test_dataset = SpectralDataset('/data07/huanglin/SingEval/LeaderboardData/data/test_1.dill', 0, rep_params)
test_dataloader = SpectralDataLoader(test_dataset, batch_size = 5, num_workers = 1, shuffle = False)

valid_dataset = SpectralDataset('/data07/huanglin/SingEval/LeaderboardData/data/val_1.dill', 0, rep_params)
valid_dataloader = SpectralDataLoader(valid_dataset, batch_size = 5, num_workers = 4, shuffle = False)

model_path = './models_CQT/model_SpectralCRNN_reg_lr0.0001_big_ELU_Adam_noteacc_bws-1to1_CQT_v1_corr' # modify
model = SpectralCRNN().cuda()
model = torch.load(model_path)

criterion = nn.MSELoss()

feature_array_train, target_array_train = evaluate_model(model, train_dataloader)
#feature_array_val, target_array_val = evaluate_model(model, valid_dataloader)
feature_array_test, target_array_test = evaluate_model(model, test_dataloader)

np.savetxt('./tsne_CQT/feature_train.txt', feature_array_train)
np.savetxt('./tsne_CQT/target_train.txt', target_array_train)
# np.savetxt('./tsne_CQT/feature_val.txt', feature_array_val)
# np.savetxt('./tsne_CQT/target_val.txt', target_array_val)
np.savetxt('./tsne_CQT/feature_test.txt', feature_array_test)
np.savetxt('./tsne_CQT/target_test.txt', target_array_test)










