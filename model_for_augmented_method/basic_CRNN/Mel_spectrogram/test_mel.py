import torch
import time
import numpy as np
from torch import nn
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import sys
sys.path.append('E:\\Codes_For_Python\\.vscode\\Huang')
from torch.autograd import Variable
from models.SpectralCRNN import SpectralCRNN_Reg_Dropout_tsne as SpectralCRNN
from tensorboard_logger import configure, log_value
from dataLoaders.SpectralDataset import SpectralDataset, SpectralDataLoader
from sklearn import metrics
from torch.optim import lr_scheduler
import dill

def evaluate_classification(targets, predictions):
    predictions[predictions < -1] = -1
    predictions[predictions > 1] = 1
    pearson = pearsonr(targets.flatten(), predictions.flatten())
    pearson_corre = pearson[0]
    print("Pearson Corr = ",pearson)
    print("Spearman Corr = ",spearmanr(targets.flatten(), predictions.flatten()))
    return pearson_corre, predictions, targets


def evaluate_model(model, dataloader):
    model.eval()
    all_predictions = []
    all_targets = []
    for i, (data) in enumerate(dataloader):
        inputs, targets = data
        inputs = Variable(inputs.cuda(), requires_grad = False)
        targets = Variable(targets.cuda(), requires_grad = False)
        targets = targets.view(-1,1)
        model.init_hidden(inputs.size(0))
        out = model(inputs)[0]
        all_predictions.extend(out.data.cpu().numpy())
        all_targets.extend(targets.data.cpu().numpy())
    return evaluate_classification(np.array(all_targets), np.array(all_predictions))


if __name__ == '__main__':
    rep_params = {'method':'Mel Spectrogram', 'n_fft':2048, 'n_mels': 96, 'hop_length': 1024, 'normalize': True}

    # train_dataset = SpectralDataset('/data07/huanglin/SingEval/LeaderboardData/data/train_1.dill', 0, rep_params)
    train_dataset = SpectralDataset('D:/dill_data/train_1.dill', 0, rep_params)
    # train_dataloader = SpectralDataLoader(train_dataset, batch_size = 10, num_workers = 4, shuffle = False)
    train_dataloader = SpectralDataLoader(train_dataset, batch_size = 1, num_workers = 1, shuffle = False)

    # test_dataset = SpectralDataset('/data07/huanglin/SingEval/LeaderboardData/data/test_1.dill', 0, rep_params)
    test_dataset = SpectralDataset('D:/dill_data/test_1.dill', 0, rep_params)
    # test_dataloader = SpectralDataLoader(test_dataset, batch_size = 10, num_workers = 1, shuffle = False)
    test_dataloader = SpectralDataLoader(test_dataset, batch_size = 1, num_workers = 1, shuffle = False)

    # valid_dataset = SpectralDataset('/data07/huanglin/SingEval/LeaderboardData/data/val_1.dill', 0, rep_params)
    valid_dataset = SpectralDataset('D:/dill_data/val_1.dill', 0, rep_params)
    # valid_dataloader = SpectralDataLoader(valid_dataset, batch_size = 10, num_workers = 4, shuffle = False)
    valid_dataloader = SpectralDataLoader(valid_dataset, batch_size = 1, num_workers = 1, shuffle = False)

    # model_path = './models_mel/model_SpectralCRNN_reg_lr0.0001_big_ELU_Adam_noteacc_bws-1to1_mel_v2'
    model_path = 'D:/nuspro/model/model_SpectralCRNN_reg_lr0.0001_big_ELU_Adam_noteacc_bws-1to1_mel_v2'
    model = SpectralCRNN().cuda()
    model = torch.load(model_path)

    criterion = nn.MSELoss()

    print('training set: ')
    train_metrics = evaluate_model(model, train_dataloader)
    #dill.dump({'machine': train_metrics[1].flatten(), 'GT': train_metrics[2].flatten()}, open('./abs_outputs/train_melspec_bws0to1.dill', 'wb'))

    print('validation set: ')
    val_metrics = evaluate_model(model, valid_dataloader)
    #dill.dump({'machine': val_metrics[1].flatten(), 'GT': val_metrics[2].flatten()}, open('./abs_outputs/val_melspec_bws0to1.dill', 'wb'))

    print('test set: ')
    test_metrics = evaluate_model(model, test_dataloader)
    #dill.dump({'machine': test_metrics[1].flatten(), 'GT': test_metrics[2].flatten()}, open('./abs_outputs/test_melspec.dill', 'wb'))


