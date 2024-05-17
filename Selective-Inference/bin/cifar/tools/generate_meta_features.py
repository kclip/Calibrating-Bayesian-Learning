"""Generate meta features from an input dataset."""

import argparse
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import torch

from src.data import features as features_lib

parser = argparse.ArgumentParser()

parser.add_argument(
    '--train-dataset', type=str,
    help='Path to InputDataset storing training data.')

parser.add_argument(
    '--cal-dataset', type=str,
    help='Path to BatchedInputDataset storing perturbed calibration data.')

parser.add_argument(
    '--val-dataset', type=str,
    help='Path to BatchedInputDataset storing perturbed validation data.')

parser.add_argument(
    '--test-datasets', type=str, nargs='+', default=[],
    help='Paths to InputDataset storing testing data.')

parser.add_argument(
    '--ood-dataset', type=str,
    help='Path to BatchedInputDataset storing perturbed ood data.')

parser.add_argument(
    '--skip-class-based-features', action='store_true',
    help='If true, do not add full confidence/class info as meta features.')

parser.add_argument(
    '--output-dir', type=str, default='../data/processed/cifar/',
    help='Output dir. Saved in same dirs as input paths if left unspecified.')

parser.add_argument(
    '--num-workers', type=int, default=32,
    help='Number of parallel processes for processing.')

parser.add_argument(
    '--net_type', type=str, default='FNN')


def save(original_path, output_dir, dataset):
    """Helper to save a dataset to disk."""
    dirname = os.path.dirname(original_path)
    basename, ext = os.path.splitext(os.path.basename(original_path))
    output_dir = output_dir if output_dir is not None else dirname
    torch.save(dataset, os.path.join(output_dir, f'{basename}.meta{ext}'))


def main(args):
    torch.manual_seed(1)
    np.random.seed(1)

    if args.output_dir is not None:
        print(f'Will save to {args.output_dir}.')
        # os.makedirs(args.output_dir, exist_ok=True)

    # Load clean calibration data.
    print('Loading data...')

    if ('FNN' == args.net_type):
        args.train_dataset = '../data/processed/cifar/train_FNN.pt'
        args.cal_dataset = '../data/processed/cifar/cal_FNN.pt'
        # args.val_dataset = '../data/processed/cifar/val_FNN.pt'
        args.test_datasets = '../data/processed/cifar/test_FNN.pt'
    elif ('FNN-CM' == args.net_type):
        args.train_dataset = '../data/processed/cifar/train_FNN-CM.pt'
        args.cal_dataset = '../data/processed/cifar/cal_FNN-CM.pt'
        # args.val_dataset = '../data/processed/cifar/val_FNN-CM.pt'
        args.test_datasets = '../data/processed/cifar/test_FNN-CM.pt'
    elif ('CA-FNN-CM' == args.net_type):
        args.train_dataset = '../data/processed/cifar/train_CA-FNN-CM.pt'
        args.cal_dataset = '../data/processed/cifar/cal_CA-FNN-CM.pt'
        # args.val_dataset = '../data/processed/cifar/val_CA-FNN-CM.pt'
        args.test_datasets = '../data/processed/cifar/test_CA-FNN-CM.pt'
    elif ('CA-FNN' == args.net_type):
        args.train_dataset = '../data/processed/cifar/train_CA-FNN.pt'
        args.cal_dataset = '../data/processed/cifar/cal_CA-FNN.pt'
        # args.val_dataset = '../data/processed/cifar/val_CA-FNN.pt'
        args.test_datasets = '../data/processed/cifar/test_CA-FNN.pt'
    elif ('BNN-CM' == args.net_type):
        args.train_dataset = '../data/processed/cifar/train_BNN-CM.pt'
        args.cal_dataset = '../data/processed/cifar/cal_BNN-CM.pt'
        # args.val_dataset = '../data/processed/cifar/val_BNN-CM.pt'
        args.test_datasets = '../data/processed/cifar/test_BNN-CM.pt'
        args.ood_dataset = '../data/processed/cifar/ood_test_BNN-CM.pt'

    elif ('BNN' == args.net_type):
        args.train_dataset = '../data/processed/cifar/train_BNN.pt'
        args.cal_dataset = '../data/processed/cifar/cal_BNN.pt'
        # args.val_dataset = '../data/processed/cifar/val_BNN.pt'
        args.test_datasets = '../data/processed/cifar/test_BNN.pt'
    elif ('CA-BNN-CM' == args.net_type):
        args.train_dataset = '../data/processed/cifar/train_CA-BNN-CM.pt'
        args.cal_dataset = '../data/processed/cifar/cal_CA-BNN-CM.pt'
        # args.val_dataset = '../data/processed/cifar/val_CA-BNN-CM.pt'
        args.test_datasets = '../data/processed/cifar/test_CA-BNN-CM.pt'
        args.ood_dataset = '../data/processed/cifar/ood_test_CA-BNN-CM.pt'

    elif ('CA-BNN' == args.net_type):
        args.train_dataset = '../data/processed/cifar/train_CA-BNN.pt'
        args.cal_dataset = '../data/processed/cifar/cal_CA-BNN.pt'
        # args.val_dataset = '../data/processed/cifar/val_CA-BNN.pt'
        args.test_datasets = '../data/processed/cifar/test_CA-BNN.pt'

    train_dataset = torch.load(args.train_dataset)
    cal_dataset = torch.load(args.cal_dataset)
    # val_dataset = torch.load(args.val_dataset)

    test_datasets = torch.load(args.test_datasets)

    ood_dataset = torch.load(args.ood_dataset)

    print(test_datasets.labels.sum())
    print(test_datasets.labels.shape)

    # test_datasets = [torch.load(dataset) for dataset in args.test_datasets]

    # Get meta features.
    print('Generating meta features...')
    meta_datasets = features_lib.process_dataset_splits(
        train_dataset=train_dataset,
        cal_dataset=cal_dataset,
        test_datasets=test_datasets,
        ood_dataset=ood_dataset,
        skip_class_based_features=True,
        num_workers=1)

    # Save to disk.
    print('Saving new datasets...')
    args.output_dir = '../data/processed/cifar/'
    cal_dataset, test_datasets, ood_dataset = meta_datasets
    save(args.cal_dataset, args.output_dir, cal_dataset)


    save(args.test_datasets, args.output_dir, test_datasets)
    save(args.ood_dataset, args.output_dir, ood_dataset)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
