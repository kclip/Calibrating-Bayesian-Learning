from torch.utils.data import random_split
import numpy as np
import function as pf
import torchvision
from torchvision import transforms
import torch
import pickle
from PIL import Image

import os

# Set the maximum split size to 128 MB
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

random_seed = 0
torch.manual_seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

test_set = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)


#################   load resize imagenet data set

transform_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
])
with open('./models/ood_test_data.pkl', 'rb') as f:
    ood_test_data = pickle.load(f)
train_images = [item[0] for item in ood_test_data]
train_labels = [item[1] for item in ood_test_data]
train_images = [Image.fromarray(img) if not isinstance(img, Image.Image) else img for img in train_images]

ood_test = pf.CustomData(train_images, train_labels, transform=transform)


uncertainty_test_loader = torch.utils.data.DataLoader(ood_test, batch_size=len(ood_test), shuffle=True)


ece = []
acc = []

ece_un = []
acc_un = []

#######################  load model  ######################
# model = torch.load('./new_models/fnn-gamma-3.0.pt')
# model = torch.load('./new_models/fnn-gamma-0.0.pt')
model = torch.load('./new_models/fnn_cm-0.5.pt')
# model = torch.load('./new_models/cf_cm-0.5.pt')
model.to(device)

####################### model evaluation on ID #######################
model.eval()
with torch.no_grad():
    correct_count, all_count = 0, 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logps = model(images)
        ps = torch.nn.functional.softmax(logps.data, dim=1)
        confidence_id, pred_label = torch.max(ps.data, dim=1)
        ECE = pf.expected_calibration_error(confidence_id, pred_label, labels, num_bins=15) * 100
        accuracy = 100 * torch.sum(pred_label.eq(labels)) / len(test_set)

        confidence_id = confidence_id.cpu()
        pred_label = pred_label.cpu()
        labels = labels.cpu()
        ECE = ECE.cpu()
        accuracy = accuracy.cpu()

        pf.reliability_diagram_plot(confidence_id, pred_label, labels, ECE, accuracy, network_type='cf-cm')


####################### model evaluation on OOD #######################
model.eval()
with torch.no_grad():
    correct_count, all_count = 0, 0
    for images, labels in uncertainty_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        logps = model(images)
        ps = torch.nn.functional.softmax(logps.data, dim=1)
        confidence_ood, pred_label = torch.max(ps.data, dim=1)

        ECE = pf.expected_calibration_error(confidence_ood, pred_label, labels, num_bins=15) * 100
        accuracy = 100 * torch.sum(pred_label.eq(labels)) / len(ood_test)

        confidence_ood = confidence_ood.cpu()
        pred_label = pred_label.cpu()
        labels = labels.cpu()

pf.confidence_distribution_plot(confidence_id, confidence_ood, 'cf-cm', num_bins=15)
