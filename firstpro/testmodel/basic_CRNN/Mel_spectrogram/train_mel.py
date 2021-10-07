import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable
import sys
sys.path.append('E:\\Codes_For_Python\\.vscode\\Huang')
from dataLoaders.SpectralDataset import SpectralDataset, SpectralDataLoader
from models.SpectralCRNN import SpectralCRNN_Reg_Dropout_tsne as SpectralCRNN
from tensorboard_logger import configure, log_value
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
    print("Spearman Corr = ",
          spearmancorr(targets.flatten(), predictions.flatten()))
    print("R2 = ", r2)
    return r2, pearson_corre

if __name__ == '__main__':
    # Configure tensorboard logger
    # configure('runs/MelSpec_reg_lr0.0001_big_ELU_Adam_noteacc_bws-1to1',
    configure('D:/nuspro/logg/MelSpec_reg_lr0.0001_big_ELU_Adam_noteacc_bws-1to1',
            flush_secs=2)

    # Parameteres for Spectral Representation
    rep_params = {
        'method': 'Mel Spectrogram',
        'n_fft': 2048,
        'n_mels': 96,
        'hop_length': 1024,
        'normalize': True
    }

    # Load Datasets
    train_dataset = SpectralDataset(
        # '/data07/huanglin/SingEval/LeaderboardData/data/train_1.dill', 0,
        # 'D:/dill_data/train_1.dill', 0,
        'D:/dill_data/test/train_1.dill', 0,
        rep_params)
    train_dataloader = SpectralDataLoader(train_dataset,
                                        # batch_size=10,
                                        batch_size=1,
                                        # num_workers=4,
                                        num_workers=1,
                                        shuffle=True)
    # pdb.set_trace()

    test_dataset = SpectralDataset(
        # '/data07/huanglin/SingEval/LeaderboardData/data/test_1.dill', 0,
        # 'D:/dill_data/test_1.dill', 0,
        'D:/dill_data/test/test_1.dill', 0,
        rep_params)
    # test_dataloader = SpectralDataLoader(test_dataset,
    #                                     batch_size=10,
    #                                     num_workers=1,
    #                                     shuffle=True)
    test_dataloader = SpectralDataLoader(test_dataset,
                                        batch_size=1,
                                        num_workers=1,
                                        shuffle=True)

    valid_dataset = SpectralDataset(
        # '/data07/huanglin/SingEval/LeaderboardData/data/val_1.dill', 0, rep_params)
        # 'D:/dill_data/val_1.dill', 0, rep_params)
        'D:/dill_data/test/val_1.dill', 0, rep_params)
    # valid_dataloader = SpectralDataLoader(valid_dataset,
    #                                     batch_size=10,
    #                                     num_workers=4,
    #                                     shuffle=True)
    valid_dataloader = SpectralDataLoader(valid_dataset,
                                        batch_size=1,
                                        num_workers=1,
                                        shuffle=True)
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
        print('Epoch: ', epoch)
        print('training set: ')
        model.train()
        # scheduler.step()
        avg_loss = 0.0
        end = time.time()
        all_predictions = []
        all_targets = []
        losses = AverageMeter()
        for i, (data) in enumerate(train_dataloader):
            inputs, targets = data
            data_time.update(time.time() - end)
            inputs = Variable(inputs.cuda(), requires_grad=False)
            targets = Variable(targets.cuda(), requires_grad=False)
            targets = targets.view(-1, 1)
            model.init_hidden(inputs.size(0))
            out = model(inputs)[0]
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
        train_r2, train_pearson = evaluate_classification(
            np.array(all_targets), np.array(all_predictions))

        print('validation set: ')
        model.eval()
        losses = AverageMeter()
        all_predictions = []
        all_targets = []
        for i, (data) in enumerate(valid_dataloader):
            inputs, targets = data
            data_time.update(time.time() - end)
            inputs = Variable(inputs.cuda(), requires_grad=False)
            targets = Variable(targets.cuda(), requires_grad=False)
            targets = targets.view(-1, 1)
            model.init_hidden(inputs.size(0))
            out = model(inputs)[0]
            all_predictions.extend(out.data.cpu().numpy())
            all_targets.extend(targets.data.cpu().numpy())
            loss = criterion(out, targets)
            loss_value = loss.data
            losses.update(loss_value, inputs.size(0))
        valid_loss = losses.avg
        val_r2, val_pearson = evaluate_classification(np.array(all_targets),
                                                    np.array(all_predictions))
        log_value('Train Loss', train_loss, epoch)
        log_value('Validation Loss', valid_loss, epoch)

        if val_pearson > best_val_corr:
            best_val_corr = val_pearson
            print('save pearson corr model: ')
            torch.save(
                model,
                'D:/nuspro/model/model_SpectralCRNN_reg_lr0.0001_big_ELU_Adam_noteacc_bws-1to1_mel_v2'
            )
