'''
This module contains methods for training models with different loss functions.
'''

import torch
from torch.nn import functional as F
from torch import nn

from Losses.loss import cross_entropy, focal_loss, focal_loss_adaptive
from Losses.loss import mmce, mmce_weighted
from Losses.loss import brier_score
from Losses.loss import avuc
import torchbnn as bnn
from torch.cuda.amp import autocast, GradScaler

loss_function_dict = {
    'cross_entropy': cross_entropy,
    'focal_loss': focal_loss,
    'focal_loss_adaptive': focal_loss_adaptive,
    'mmce': mmce,
    'mmce_weighted': mmce_weighted,
    'brier_score': brier_score,
    'avuc': avuc
}

torch.autograd.set_detect_anomaly(True)

def train_single_epoch(kl1,
                       epoch,
                       model,
                       train_loader,
                       optimizer,
                       device,
                       loss_function='cross_entropy',
                       gamma=1.0,
                       lamda=1.0,
                       loss_mean=False):
    '''
    Util method for training a model for a single epoch.
    '''
    model.train()
    kl_weight = kl1
    ensemble_train = 1
    kl_loss = bnn.BKLLoss(reduction='sum', last_layer_only=False)
    kl_loss.to(device)
    length = len(train_loader)
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss_avg = 0
        kl_avg = 0
        # with autocast():
        if ('avuc' in loss_function):
            logits_avg = 0
            for _ in range(ensemble_train):

                logits = model(data)
                loss = (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device))

                loss_avg += loss / ensemble_train
        else:
            for _ in range(ensemble_train):

                logits = model(data)

                if ('mmce' in loss_function):
                    loss = (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device))
                else:
                    loss = loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device)


                if loss_mean:
                    loss = loss / len(data)
                loss_avg = loss_avg + loss / ensemble_train

        kl = kl_loss(model)
        loss = loss_avg + (kl_weight / length) * kl


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()


        if batch_idx % 150 == 0:
            print('Epoch:{} ---Batch:{} ---Loss:{}'.format(epoch, batch_idx, loss.item()))




def test_single_epoch(epoch,
                      model,
                      test_val_loader,
                      device,
                      loss_function='cross_entropy',
                      gamma=1.0,
                      lamda=1.0):
    '''
    Util method for testing a model for a single epoch.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            if ('mmce' in loss_function):
                loss += (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device).item())
            else:
                loss += loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device).item()
            num_samples += len(data)

    print('======> Test set loss: {:.4f}'.format(
        loss / num_samples))
    return loss / num_samples

