import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

from torch.utils.data import random_split
import numpy as np
import torch
import function_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random_seed = 1
torch.manual_seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)



net_type = 'BNN-CM'

#  test the performance of the selector
selectors_dir = './models/SC_' + net_type + '_MMCE_5.pt'
test_dataset_dir = '../data/processed/cifar/test_' + net_type + '.meta.pt'
ood_test_dataset_dir = '../data/processed/cifar/ood_test_' + net_type + '.meta.pt'

test = torch.load(test_dataset_dir)[0]
ood_test = torch.load(ood_test_dataset_dir)[0]

test_loader = torch.utils.data.DataLoader(dataset=test.input_features, batch_size=10000)
ood_test_loader = torch.utils.data.DataLoader(dataset=ood_test.input_features, batch_size=4000)

selector = torch.load(selectors_dir).cuda()

selector.eval()
with torch.no_grad():
    for batch in test_loader:
        features_flat = batch.view(10000, -1).cuda()
        logits = selector(features_flat.float()).view(1, 10000)
        weights = torch.sigmoid(logits)

    for batch in ood_test_loader:
        features_flat = batch.view(4000, -1).cuda()
        logits = selector(features_flat.float()).view(1, 4000)
        ood_weights = torch.sigmoid(logits)

    threshold = int(1 * weights.shape[1])
    threshold = weights.topk(k=threshold, largest=True, sorted=True).values[-1][-1]

    selector_decision = torch.where(weights >= threshold, torch.ones(weights.shape).to(device), torch.zeros(weights.shape).to(device))
    ood_selector_decision = torch.where(ood_weights >= threshold, torch.ones(ood_weights.shape).to(device), torch.zeros(ood_weights.shape).to(device))

    accept_ratio = 100 * selector_decision.sum() / 10000
    print(f'ID accept_ratio: {accept_ratio}')

    ood_accept_ratio = 100 * ood_selector_decision.sum() / 4000
    print(f'OOD accept_ratio: {ood_accept_ratio}')

    ratio = 100 * ood_selector_decision.sum() / (ood_selector_decision.sum() + selector_decision.sum())
    print(f'OOD/Total ratio: {ratio}')

    mask = selector_decision != 0
    ood_mask = ood_selector_decision != 0

    acc = test.labels.cuda()
    confidences = test.confidences.cuda()

    pred_label = test.pred_labels.cuda()
    labels = test.targets.cuda()

    ECE = function_list.expected_calibration_error(confidences[mask[0]], acc[mask[0]], num_bins=15) * 100
    print(f'ECE: {ECE}')
    ECE = function_list.expected_calibration_error_test(confidences[mask[0]], pred_label[mask[0]], labels[mask[0]], num_bins=15) * 100
    print(f'ECE: {ECE}')
    accuracy = torch.round(100 * acc[mask[0]].sum() / selector_decision.sum(), decimals=3)
    print(f'accuracy: {accuracy}')

    confidence_id = confidences[mask[0]].cpu()
    pred_label = pred_label[mask[0]].cpu()
    labels = labels[mask[0]].cpu()
    ECE = ECE.cpu()
    accuracy = accuracy.cpu()

    function_list.reliability_diagram_plot(confidence_id, pred_label, labels, ECE, accuracy, network_type='mmce')

confidence_ood = ood_test.confidences[ood_mask[0].cpu()]
function_list.confidence_distribution_plot(confidence_id, confidence_ood, 'mmce', num_bins=15)