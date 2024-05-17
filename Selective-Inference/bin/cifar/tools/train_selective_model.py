"""Train selector model."""

import argparse
import logging
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))


from src import losses
from src import metrics
from src import utils
from src.data import BatchedInputDataset
from src.models import SelectiveNet

parser = argparse.ArgumentParser()

parser.add_argument(
    '--cal-dataset', type=str,
    help='Path to BatchedInputDataset storing perturbed calibration data.')
parser.add_argument(
    '--net_type', type=str, default='FNN')
parser.add_argument(
    '--loss', type=str, default='MMCE')
parser.add_argument(
    '--val-dataset', type=str,
    help='Path to BatchedInputDataset storing perturbed validation data.')

parser.add_argument(
    '--model-dir', type=str, default='./models',
    help='Directory for saving model and logs.')

parser.add_argument(
    '--overwrite', action='store_false',
    help='Overwrite model_dir if it already exists.')

parser.add_argument(
    '--hidden-dim', type=int, default=64,
    help='Hidden dimension for selector MLP.')

parser.add_argument(
    '--num_layers', type=int, default=1,
    help='Number of layers for the selector MLP.')

parser.add_argument(
    '--dropout', type=float, default=0.0,
    help='Dropout level for the selector MLP.')

parser.add_argument(
    '--kappa', type=int, default=-1,
    help='DRO kappa parameter. If -1, uses the whole batch (not kappa-worst).')

parser.add_argument(
    '--seed', type=int, default=42,
    help='Random seed to use for training.')

parser.add_argument(
    '--epochs', type=int, default=5,
    help='Number of training epochs.')

parser.add_argument(
    '--learning-rate', type=float, default=1e-3,
    help='Optimizer learning rate (for Adam).')

parser.add_argument(
    '--weight-decay', type=float, default=1e-5,
    help='L2 regularization strength (for Adam).')

parser.add_argument(
    '--train-batch-size', type=int, default=32,
    help='Batch size during training (batches of perturbed datasets).')

parser.add_argument(
    '--eval-batch-size', type=int, default=32,
    help='Batch size during evaluation (batches of perturbed datasets).')

parser.add_argument(
    '--p-norm', type=float, default=2,
    help='L_p norm parameter for use in S-MMCE and ECE calculations.')

parser.add_argument(
    '--smmce-weight', type=float, default=1,
    help='Weight for coverage regularizer (collapse of g).')

parser.add_argument(
    '--coverage-weight', type=float, default=1e-2,
    help='Weight for coverage regularizer (collapse of g).')

parser.add_argument(
    '--clip-grad-norm', type=float, default=10,
    help='Max grad norm for gradient clipping.')

parser.add_argument(
    '--print-freq', type=int, default=100,
    help='Print every n steps.')

parser.add_argument(
    '--use-cpu', action='store_true',
    help='Force to use CPU even if GPU is available.')

parser.add_argument(
    '--num-workers', type=int, default=8,
    help='Number of data loader background processes.')


def train_model(args, selector, data_loader, optimizer, epoch):
    """Train the selector model for one epoch.

    Args:
        args: Namespace object containing relevant training arguments.
        selector: SelectiveNet nn.Module implementing the selector g(X).
        data_loader: DataLoader for training data, where each batch can be
            converted into a BatchedInputDataset.
        optimizer: Torch optimizer for updating the selector.

    Returns:
        A reference to the selector.
    """
    selector.train()
    device = next(selector.parameters()).device

    for cc in range(10240):
        for i, batch in enumerate(data_loader):
            batch = BatchedInputDataset(*[ex.to(device) for ex in batch])

            m, n = 1,  batch.labels.shape[0]

            features_flat = batch.input_features.view(m * n, -1)
            logits = selector(features_flat.float()).view(m, n)
            weights = torch.sigmoid(logits).squeeze()

            # Compute losses for each sub-dataset (perturbation) in the batch.
            smmce_loss = torch.zeros(m).to(device)
            cov_loss = torch.zeros(m).to(device)
            sbce_aucs = torch.zeros(m).to(device)
            for j in range(m):

                if (args.loss == 'MMCE'):
                    # Compute the S-MMCE_u loss over dataset j.
                    # if using selective calibration, use below smmce loss to replace the CE

                    smmce_loss[j] = losses.compute_smmce_loss(
                        outputs=batch.confidences,
                        targets=batch.labels,
                        weights=weights,
                        pnorm=args.p_norm)

                if (args.loss == 'CE'):

                    # loss for using selective classification, use below CE loss to replace smmce
                    a = batch.true_confidences
                    b = weights.detach()
                    smmce_loss[j] = torch.dot(-torch.log(a), b)

                smmce_loss[j] *= (args.smmce_weight / np.sqrt(n))

                # Compute the regularization term over dataset j.
                cov_loss[j] = F.binary_cross_entropy_with_logits(
                    logits[j], torch.ones_like(logits[j]), reduction='sum')
                cov_loss[j] *= (args.coverage_weight / n)

                # Optionally, compute the S-BCE AUC to use for computing the kappa-
                # worst datasets out of the batch to use for DRO-style optimization.

                if args.kappa > 0:
                    with torch.no_grad():
                        sbce_aucs[j] = metrics.compute_metric_auc(
                            outputs=batch.confidences[j],
                            targets=batch.labels[j],
                            weights=weights[j],
                            metric_fn=metrics.compute_ece,
                            num_auc_values=10).auc

            # Optional: If we are using kappa-worst DRO, then we take the
            # kappa-worst batches. Otherwise, we take all batches.
            if args.kappa > 0:
                indices = torch.topk(sbce_aucs, args.kappa).indices
            else:
                indices = torch.arange(len(sbce_aucs), device=device)

            # Aggregate total loss.
            smmce_loss = torch.index_select(smmce_loss, 0, indices).mean()
            cov_loss = torch.index_select(cov_loss, 0, indices).mean()
            loss = smmce_loss + cov_loss

            if i % 100 == 0 and i != 0:
                print("Epoch - {} - Iteration - {} - Training loss: {}".format(epoch, cc, loss.item()))

            # Update parameters.
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                selector.parameters(),
                args.clip_grad_norm,
                error_if_nonfinite=True)
            optimizer.step()

    return selector


def evaluate_model(selector, data_loader):
    """Evaluate the selective calibration error AUC of the selector.

    Args:
        selector: SelectiveNet nn.Module implementing the selector g(X).
        data_loader: DataLoader for evaluation data, where each batch can be
            converted into a BatchedInputDataset.

    Returns:
        A tuple of (average-case, worst-case) selective calibration error AUC.
    """
    selector.eval()
    device = next(selector.parameters()).device
    all_outputs, all_targets, all_weights = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = BatchedInputDataset(*[ex.to(device) for ex in batch])

            # Forward pass.
            m, n = 1, batch.labels.shape[0]
            features_flat = batch.input_features.view(m * n, -1)
            logits = selector(features_flat.float()).view(m, n)
            weights = torch.sigmoid(logits)

            # Aggregate.
            all_outputs.append(batch.confidences.cpu())
            all_targets.append(batch.labels.cpu())
            all_weights.append(weights.cpu())

    # Compute metrics.
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_weights = torch.cat(all_weights, dim=0)

    all_aucs = []
    for idx in range(len(all_outputs)):
        all_aucs.append(metrics.compute_metric_auc(
            outputs=all_outputs[idx],
            targets=all_targets[idx],
            weights=all_weights[idx],
            metric_fn=metrics.compute_ece).auc)

    logging.info(utils.format_eval_metrics(all_aucs))

    return 100 * np.mean(all_aucs), 100 * np.max(all_aucs)


# ------------------------------------------------------------------------------
#
# Main.
#
# ------------------------------------------------------------------------------


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Save args.
    torch.save(args, os.path.join(args.model_dir, 'args.pt'))

    # Set device.
    args.cuda = torch.cuda.is_available() and not args.use_cpu
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # --------------------------------------------------------------------------
    #
    # Load data.
    #
    # --------------------------------------------------------------------------
    logging.info('Loading data...')

    if ('FNN' == args.net_type):
        args.cal_dataset = '../data/processed/cifar/cal_FNN.meta.pt'
    elif ('FNN-CM' == args.net_type):
        args.cal_dataset = '../data/processed/cifar/cal_FNN-CM.meta.pt'

    elif ('CA-FNN-CM' == args.net_type):
        args.cal_dataset = '../data/processed/cifar/cal_CA-FNN-CM.meta.pt'
    elif ('CA-FNN' == args.net_type):
        args.cal_dataset = '../data/processed/cifar/cal_CA-FNN.meta.pt'
    elif ('BNN-CM' == args.net_type):
        args.cal_dataset = '../data/processed/cifar/cal_BNN-CM.meta.pt'
    elif ('BNN' == args.net_type):
        args.cal_dataset = '../data/processed/cifar/cal_BNN.meta.pt'
    elif ('CA-BNN-CM' == args.net_type):
        args.cal_dataset = '../data/processed/cifar/cal_CA-BNN-CM.meta.pt'
    elif ('CA-BNN' == args.net_type):
        args.cal_dataset = '../data/processed/cifar/cal_CA-BNN.meta.pt'

    cal_dataset = torch.utils.data.TensorDataset(*torch.load(args.cal_dataset))

    cal_loader = torch.utils.data.DataLoader(
        dataset=cal_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=args.cuda)

    # --------------------------------------------------------------------------
    #
    # Initialize model and optimizer.
    #
    # --------------------------------------------------------------------------
    logging.info('Initializing model...')
    input_dim = cal_dataset.tensors[0].size(-1)
    print(f'Input dimension is {input_dim}')
    logging.info(f'Input dimension is {input_dim}')

    selector = SelectiveNet(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(
        selector.parameters(),
        args.learning_rate,
        weight_decay=args.weight_decay)

    # --------------------------------------------------------------------------
    #
    # Train model.
    #
    # --------------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        logging.info("=" * 88)
        logging.info(f'Starting epoch {epoch}/{args.epochs}...')
        selector = train_model(args, selector, cal_loader, optimizer, epoch)
        logging.info("=" * 88)

        torch.save(selector, './models/SC_{}_{}_{}.pt'.format(args.net_type, args.loss, epoch))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)