import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

from torch.utils.data import random_split, ConcatDataset, Subset, DataLoader
import numpy as np
import torchvision
from torch.utils.data import random_split
from torch import optim
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import argparse
import pickle
import function
from PIL import Image
from Losses.loss import cross_entropy, focal_loss, focal_loss_adaptive
from Losses.loss import mmce, mmce_weighted
from Losses.loss import brier_score
from Losses.loss import avuc
import torchbnn as bnn


loss_function_dict = {
    'cross_entropy': cross_entropy,
    'focal_loss': focal_loss,
    'focal_loss_adaptive': focal_loss_adaptive,
    'mmce': mmce,
    'mmce_weighted': mmce_weighted,
    'brier_score': brier_score,
    'avuc': avuc
}


def parse_args():
    parser = argparse.ArgumentParser(description='BNN')
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--loss', type=str, default='cross_entropy')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--ow', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    return args

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    def create_dataloader(dataset, size):

        indices = np.random.choice(len(dataset), size, replace=False)
        subset = Subset(dataset, indices)
        dataloader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)
        return dataloader

    ###########    load CIFAR-100 data set
    transform = transforms.Compose(
        [transforms.Pad(4),  # Add padding of 4 pixels to each side
         transforms.RandomCrop(32),  # Then take a random crop of 32 x 32
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # Download and load the training data
    train_set = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
    id_train, _ = random_split(train_set, [45000, 5000])



    id_train_loader = torch.utils.data.DataLoader(id_train, batch_size=32, shuffle=True)
    print(len(id_train_loader))


    #################   load resize imagenet data set
    with open('./new_models/uncertainty_train_data.pkl', 'rb') as f:
        uncertainty_train = pickle.load(f)
    train_images = [item[0] for item in uncertainty_train]
    train_labels = [item[1] for item in uncertainty_train]
    train_images = [Image.fromarray(img) if not isinstance(img, Image.Image) else img for img in train_images]

    uncertainty_train = function.CustomData(train_images, train_labels, transform=transform)

    uncertainty_train = ConcatDataset([uncertainty_train, uncertainty_train, uncertainty_train])

    uncertainty_train_loader = torch.utils.data.DataLoader(uncertainty_train, batch_size=64, shuffle=True)

    args.loss = 'wmmce'

    # load Pretrained model import
    if ('cross_entropy' in args.loss):
        model = torch.load('./new_models/bnn.pt')
    elif ('wmmce' in args.loss):
        model = torch.load('./new_models/bnn-gamma-0.8.pt')
    print(model)

    model.to(device)

    ce_loss = nn.CrossEntropyLoss(reduction='sum')
    ce_loss.to(device)

    m = torch.nn.LogSoftmax(dim=1)
    m.to(device)

    kl_loss = bnn.BKLLoss(reduction='sum', last_layer_only=False)
    kl_loss.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(step, 10 * 9000//32, 1, 1e-6 / args.lr))

    ################################       training model

    epochs = 10
    ensemble_train = 1
    kl_weight = 0.00035 * 0.8

    for e in range(epochs):
        i = 0
        scheduler.step()
        id_train_loader = create_dataloader(id_train, 9000)

        for in_set, out_set in zip(id_train_loader, uncertainty_train_loader):
            data = torch.cat((in_set[0], out_set[0]), 0)
            target = in_set[1]

            data, target = data.cuda(), target.cuda()

            x = model(data)
            scheduler.step()
            optimizer.zero_grad()

            loss_avg = 0
            for _ in range(ensemble_train):
                if ('wmmce' in args.loss):
                    loss = (len(target) * loss_function_dict['mmce_weighted'](x[:len(in_set[0])], target, gamma=0, lamda=0.8, device=device))
                else:
                    loss = nn.functional.cross_entropy(x[:len(in_set[0])], target, reduction='sum')

                loss += args.ow * torch.sum(-m(x[len(in_set[0]):])) / 100

                loss_avg += loss / ensemble_train

            kl = kl_loss(model)
            loss = loss_avg + ((kl_weight / len(id_train_loader)) * kl)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4, error_if_nonfinite=True)
            optimizer.step()

            print("Epoch {} - Batch {} - Training loss: {}".format(e, i, loss.item()))
            i += 1
    if ('wmmce' in args.loss):
        torch.save(model, './new_models/cb_cm-{}.pt'.format(args.ow))

    else:
        torch.save(model, './new_models/bnn_cm-{}.pt'.format(args.ow))




if __name__ == '__main__':
    args = parse_args()
    main(args)