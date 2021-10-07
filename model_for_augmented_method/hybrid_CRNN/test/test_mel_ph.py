import torch
import time
import numpy as np
from torch import nn
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from torch.autograd import Variable
import sys
sys.path.append('./exp1')
from models.SpectralCRNN_hybrid import SpectralCRNN_Reg_Dropout_mel_ph as SpectralCRNN
from tensorboard_logger import configure, log_value
from dataLoaders.SpectralDataset_ph import SpectralDataset, SpectralDataLoader
from sklearn import metrics
from torch.optim import lr_scheduler
import dill
import scipy.io as io

def evaluate_classification(targets, predictions):
    r2 = metrics.r2_score(targets, predictions)
    predictions[predictions < -1] = -1
    predictions[predictions > 1] = 1
    pearson = pearsonr(targets.flatten(), predictions.flatten())
    pearson_corre = pearson[0]
    print("Pearson Corr = ", pearson)
    print("Spearman Corr = ", spearmanr(targets.flatten(), predictions.flatten()))
    print("R2 = ", r2)
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
        # recording.write(out)
        #print(out)
        all_predictions.extend(out.data.cpu().numpy())
        all_targets.extend(targets.data.cpu().numpy())
    #result1 = np.array(all_predictions)
    #result2 = np.array(all_targets)
    #file=open('/hpctmp/e0572686/temp.txt','w')
    #for i in all_predictions:
    #  file.write(str(i))
    #  file.write('\n')

    #file.close()
    #file=open('/hpctmp/e0572686/temp2.txt','w')
    #for i in all_targets:
    #  file.write(str(i))
    #  file.write('\n')
    #file.close()
    return evaluate_classification(np.array(all_targets), np.array(all_predictions))

if __name__ == '__main__':
    rep_params = {'method':'Mel Spectrogram', 'n_fft':2048, 'n_mels': 96, 'hop_length': 1024, 'normalize': True}
    # recording = open('/hpctmp/e0572686/temp.txt','w')
    # Load Datasets
    # train_dataset = SpectralDataset('/data07/huanglin/SingEval/LeaderboardData/data_ph/train_1.dill', 0, rep_params)
    train_dataset = SpectralDataset('/hpctmp/e0572686/dill_data_ph/train_1.dill', 0, rep_params)
    # train_dataset = SpectralDataset('/hpctmp/e0572686/dilldata_badthree/train_1.dill', 0, rep_params)
    # train_dataloader = SpectralDataLoader(train_dataset, batch_size=10, num_workers=4, shuffle=True)
    train_dataloader = SpectralDataLoader(train_dataset, batch_size=10, num_workers=4, shuffle=True)
    # pdb.set_trace()

    # test_dataset = SpectralDataset('/data07/huanglin/SingEval/LeaderboardData/data_ph/test_1.dill', 0, rep_params)
    test_dataset = SpectralDataset('/hpctmp/e0572686/dill_data_ph/test_1.dill', 0, rep_params)
    #test_dataset = SpectralDataset('/hpctmp/e0572686/dilldata_badthree/test_1.dill', 0, rep_params)
    test_dataloader = SpectralDataLoader(test_dataset, batch_size=10, num_workers=1, shuffle=True)
    # test_dataloader = SpectralDataLoader(test_dataset, batch_size=10, num_workers=1, shuffle=True)

    valid_dataset = SpectralDataset('/hpctmp/e0572686/dill_data_ph/val_1.dill', 0, rep_params)
    #valid_dataset = SpectralDataset('/hpctmp/e0572686/dilldata_badthree/val_1.dill', 0, rep_params)
    # valid_dataset = SpectralDataset('/data07/huanglin/SingEval/LeaderboardData/data_ph/val_1.dill', 0, rep_params)
    valid_dataloader = SpectralDataLoader(valid_dataset, batch_size=10, num_workers=4, shuffle=True)
    # valid_dataloader = SpectralDataLoader(valid_dataset, batch_size=10, num_workers=4, shuffle=True)

    # model_path = './models_hybrid/model_SpectralCRNN_reg_lr0.0001_big_ELU_Adam_noteacc_bws-1to1_mel_ph_v2'
    model_path = "/hpctmp/e0572686/model/model_mel_ph_exp3_retrainonold"
    # model_path = "/hpctmp/e0572686/model/model_mel_ph"
    model = SpectralCRNN().cuda()
    model = torch.load(model_path)

    criterion = nn.MSELoss()

    print('training set: ')
    train_metrics = evaluate_model(model, train_dataloader)

    print('validation set: ')
    val_metrics = evaluate_model(model, valid_dataloader)

    print('test set: ')
    test_metrics = evaluate_model(model, test_dataloader)
    #dill.dump({'machine': test_metrics[1].flatten(), 'GT': test_metrics[2].flatten()}, open('./abs_outputs/test_melspec_ph.dill', 'wb'))
