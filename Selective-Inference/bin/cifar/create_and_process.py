"""Simple script to split CIFAR10 to train/cal/val + test (fixed)."""

import argparse
import os
import numpy as np
import torchvision.datasets as datasets
from torchvision import transforms

"""Process CIFAR-10 datasets to be in the desired format.

For each dataset we compute a tuple of:
    (1) Input features \phi(x), here defined as last layer network features.
    (2) Output probabilities p(y | x) for every y \in [K], here K = 10.
    (3) Model confidence f(X), here defined as max p(y | x).
    (4) Target label Y, here defined as 1{y = argmax p(y | x)}.

See src.data.InputDataset.
"""
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import argparse
import os
import tqdm
from Net.wideresnet import wideresnet_40_2
from Net.bayes_wideresnet import bayes_wideresnet_40_2
from torch.utils.data import random_split
import numpy as np
import function_list as pf
import torch
import pickle
from PIL import Image
# from third_party.ResNeXt_DenseNet.models.densenet import densenet
# from third_party.ResNeXt_DenseNet.models.resnext import resnext29
# from third_party.WideResNet_pytorch.wideresnet import wideresnet
import torchvision
from torch.utils.data import random_split
from torchvision import datasets, transforms


import third_party.augmentations
from src.data import InputDataset, BatchedInputDataset
from src import utils

from src.data import image_datasets

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data-dir', type=str, default='./data/datasets/CIFAR-100',
    help='Path to CIFAR-100 dataset (or where it should be downloaded).')

parser.add_argument(
    '--train-p', type=float, default=0.9,
    help='Percentage of training data to reserve for a proper train set.')

parser.add_argument(
    '--cal-p', type=float, default=0.10,
    help='Percentage of training data to reserve for a calibration set.')

parser.add_argument(
    '--output-dir', type=str, default='./data/processed/cifar',
    help='Path where new splits will be saved.')

parser.add_argument(
    '--model', type=str, default='wrn',
    choices=['wrn', 'densenet', 'resnext'],
    help='Model architecture type.')

parser.add_argument(
    '--model-file', type=str,
    default='data/models/cifar/models_saved/fnn-gamma-0.0.pt',
    help='Filename for saved model.')

parser.add_argument(
    '--net_type', type=str, default='FNN')

args = parser.parse_args()


def compute_subbatches(net, images, network_type, feature_projection=None):
    """Compute batch features and logits in increments.

    Args:
        net: nn.Module that implements 'get_features' and 'get_logits'.
        images: Batch of image tensors.
        feature_projection: Projection matrix to apply to features.

    Returns:
        Features and logits for the input images.
    """
    GPU_BATCH_SIZE = 256
    random_seed = 1
    torch.manual_seed(random_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    net.eval()
    # print(network_type)
    device = next(net.parameters()).device
    batch_size = len(images)
    indices = list(range(batch_size))
    splits = [indices[i:i + GPU_BATCH_SIZE]
              for i in range(0, batch_size, GPU_BATCH_SIZE)]

    features, logits, probs = [], [], []
    ensemble = 20
    # for split in splits:
    if ('FNN' in network_type):
        split_features = net.get_features(images.to(device))
        split_logits = net.get_logits(split_features)
        avg_prob = torch.nn.functional.softmax(split_logits.data, dim=1)
    elif ('BNN' in network_type):
        ans_feature = 0
        ans_logits = 0
        avg_prob = 0
        for _ in range(ensemble):
            split_features = net.get_features(images.to(device))
            split_logits = net.get_logits(split_features)
            ans_feature += split_features / ensemble
            ans_logits += split_logits / ensemble
            ps = torch.nn.functional.softmax(split_logits.data, dim=1)
            avg_prob += ps / ensemble
        split_features = ans_feature
        split_logits = ans_logits

    # Maybe project features...
    if feature_projection is not None:
        split_features = torch.mm(split_features, feature_projection)

    features.append(split_features.cpu())
    logits.append(split_logits.cpu())
    probs.append(avg_prob.cpu())

    return torch.cat(features, dim=0), torch.cat(logits, dim=0), torch.cat(probs, dim=0)

def convert_image_dataset(
    net,
    loader,
    network_type,
    temperature=1,
    keep_batches=True,
    svd=None,
):
    """Convert a dataset of images into an InputDataset.

    Args:
        net: nn.Module that implements 'get_features' and 'get_logits'.
        loader: DataLoader yielding batches of images and targets.
        temperature: Value to use for temperature scaling.
        keep_batches: Stack rather than concatenate all input batches.
        svd: Projection matrix of TruncatedSVD, as a torch tensor.

    Returns:
        An instance of InputDataset or BatchedInputDataset.
    """
    # print(network_type)
    N, M = len(loader), loader.batch_size
    all_features = None
    all_logits = None
    all_probs = None
    all_confidences = torch.empty(N, M)
    all_true_label_confidences = torch.empty(N, M)
    all_labels = torch.empty(N, M)
    all_pred_labels = torch.empty(N, M)
    all_targets = torch.empty(N, M)
    idx = count = 0
    with torch.no_grad():
        for images, targets in tqdm.tqdm(loader, desc='processing dataset'):

            features, logits, probs = compute_subbatches(net.module, images, network_type, svd)
            # probs = F.softmax(logits / temperature, dim=1)
            confidences, pred_labels = torch.max(probs, dim=1)
            labels = pred_labels.eq(targets).float()


            true_label_confidences = probs.gather(1, targets.unsqueeze(1))

            # Remove the extra dimension resulting from gather
            true_label_confidences = true_label_confidences.squeeze(1)

            if all_features is None:
                all_features = torch.empty(N, M, features.size(-1))
                all_logits = torch.empty(N, M, logits.size(-1))
                all_probs = torch.empty(N, M, probs.size(-1))

            all_features[idx] = features
            all_logits[idx] = logits
            all_confidences[idx] = confidences
            all_true_label_confidences[idx] = true_label_confidences
            all_probs[idx] = probs
            all_labels[idx] = labels
            all_targets[idx] = targets
            all_pred_labels[idx] = pred_labels

            idx += 1
            count += len(images)
    print(all_labels)
    print(all_labels.shape)
    print(all_labels.sum())

    if not keep_batches:
        all_features = all_features.view(N * M, -1)[:count]
        all_logits = all_logits.view(N * M, -1)[:count]
        all_probs = all_probs.view(N * M, -1)[:count]
        all_confidences = all_confidences.view(N * M)[:count]
        all_true_label_confidences = all_confidences.view(N * M)[:count]
        all_labels = all_labels.view(N * M)[:count]
        all_targets = all_targets.view(N * M)[:count]
        all_pred_labels = all_pred_labels.view(N * M)[:count]

    cls = InputDataset if not keep_batches else BatchedInputDataset
    return cls(
        input_features=all_features,
        output_probs=all_probs,
        confidences=all_confidences,
        true_confidences=all_true_label_confidences,
        labels=all_labels,
        logits=all_logits,
        pred_labels=all_pred_labels,
        targets=all_targets)



random_seed = 1
torch.manual_seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

# Load standard training split.
transform = transforms.Compose(
    [transforms.Pad(4),  # Add padding of 4 pixels to each side
     transforms.RandomCrop(32),  # Then take a random crop of 32 x 32
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# Download and load the training data
train_set = torchvision.datasets.CIFAR100(root='./data/datasets/CIFAR-100', train=True, download=True, transform=transform)
train, cal = random_split(train_set, [45000, 5000])

train_loader = torch.utils.data.DataLoader(train, batch_size=1000, shuffle=True)
cal_loader = torch.utils.data.DataLoader(cal, batch_size=1000, shuffle=True)

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
test_dataset = datasets.CIFAR100(root='./data/datasets/CIFAR-100', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)



transform_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
])
with open('./ood_test_data.pkl', 'rb') as f:
    ood_test_data = pickle.load(f)
train_images = [item[0] for item in ood_test_data]
train_labels = [item[1] for item in ood_test_data]
train_images = [Image.fromarray(img) if not isinstance(img, Image.Image) else img for img in train_images]

ood_test = pf.CustomData(train_images, train_labels, transform=transform)


ood_test_loader = torch.utils.data.DataLoader(ood_test, batch_size=len(ood_test), shuffle=True)

#######################  load model  ######################
if ('FNN' == args.net_type):
    net = torch.load('./models_saved/fnn-gamma-0.0.pt')
elif ('FNN-CM' == args.net_type):
    net = torch.load('./models_saved/fnn_cm-0.5.pt')
elif ('CA-FNN' == args.net_type):
    net = torch.load('./models_saved/fnn-gamma-3.0.pt')
elif ('CA-FNN-CM' == args.net_type):
    net = torch.load('./models_saved/cf_cm-0.5.pt')
elif ('BNN' == args.net_type):
    net = torch.load('./models_saved/bnn.pt')
elif ('BNN-CM' == args.net_type):
    net = torch.load('./models_saved/bnn_cm-0.5.pt')
elif ('CA-BNN' == args.net_type):
    net = torch.load('./models_saved/bnn-gamma-0.8.pt')
elif ('CA-BNN-CM' == args.net_type):
    net = torch.load('./models_saved/cb_cm-0.5.pt')

print('Base model weights loaded from:', args.net_type)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

################################     process images

temperature = 1
svd = None


# Process a limited amount of training data for features/derived models.

net.eval()
with torch.no_grad():
    print('\n[Processing training data]')
    ds = convert_image_dataset(
        net, train_loader, args.net_type, temperature,  keep_batches=True, svd=None)
    a = 'train_' + args.net_type
    torch.save(ds, os.path.join(args.output_dir, f'{a}.pt'))
    del ds

    for split in ['cal']:
        print(f'\n[Processing without pertubed {split} training dataset]')
        ds = convert_image_dataset(
            net, cal_loader, args.net_type, temperature, keep_batches=False, svd=svd)
        a = split + '_' + args.net_type
        torch.save(ds, os.path.join(args.output_dir, f'{a}.pt'))

        del ds


    # Process all test datasets.
    print('\n[Processing test datasets]')

    ds = convert_image_dataset(
        net, test_loader, args.net_type, temperature, keep_batches=False, svd=svd)
    a = 'test_' + args.net_type
    torch.save(ds, os.path.join(args.output_dir, f'{a}.pt'))
    del ds


    print('\n[Processing OOD test datasets]')

    ds = convert_image_dataset(
        net, ood_test_loader, args.net_type, temperature, keep_batches=False, svd=svd)
    a = 'ood_test_' + args.net_type
    torch.save(ds, os.path.join(args.output_dir, f'{a}.pt'))
    del ds