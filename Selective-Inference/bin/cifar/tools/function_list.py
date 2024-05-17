import math
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 12

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomData(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

num_bins=15
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'

class asn():
    def ann(self):
        print(4)
def _bin_initializer(bin_dict, num_bins=15):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0
def _populate_bins(confs, acc, num_bins=15):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        accu = acc[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        if binn == -1:
            binn = 0
        if binn == num_bins:
            binn = binn - 1
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + (1 if (accu == 1) else 0)

    a = 0
    for binn in range(0, num_bins):
        a += bin_dict[binn][ACC]
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(
                bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / float(bin_dict[binn][COUNT])
    print(f'a is {a}')
    return bin_dict


def expected_calibration_error(confs, acc, num_bins=15):
    bin_dict = _populate_bins(confs, acc, num_bins)
    num_samples = len(acc)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * abs(bin_accuracy - bin_confidence)
    return ece



def _populate_bins_test(confs, preds, labels, num_bins=15):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        if binn == -1:
            binn = 0
        if binn == num_bins:
            binn = binn - 1
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(
                bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / float(bin_dict[binn][COUNT])
    return bin_dict


def expected_calibration_error_test(confs, preds, labels, num_bins=15):
    bin_dict = _populate_bins_test(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * abs(bin_accuracy - bin_confidence)
    return ece

def reliability_diagram_plot(confs, preds, labels, ECE, accuracy, network_type, num_bins=15):
    bins = np.linspace(0, 1, num_bins + 1)
    total = len(confs)
    nsamples_each_interval = []
    for i in range(0, num_bins):
        temp = np.where((bins[i] <= confs) & (confs < bins[i + 1]))
        nsamples_each_interval.append(len(temp[0]) / total)
    for i in range(0, num_bins):
        if nsamples_each_interval[i] >= 0.03:
            index = bins[i]
            break
    acc = []
    conf = []
    stotal = len(labels)
    MCE = 0
    for i in range(0, num_bins):
        sum_conf, sum_acc = 0, 0
        temp = np.where((bins[i] <= confs) & (confs < bins[i + 1]))
        total = len(temp[0])
        prob_in_bin = stotal
        print()
        if total != 0:
            for j in range(0, total):
                sum_conf = sum_conf + confs[temp[0][j]]
                if preds[temp[0][j]] == labels[temp[0][j]]:
                    sum_acc = sum_acc + 1
            conf.append(sum_conf / total)
            acc.append(sum_acc / total)
            MCE_temp = abs(sum_acc / total - sum_conf / total)
            if MCE_temp > MCE and (total / stotal) >= 0.1:
                MCE = MCE_temp
        else:
            conf.append(0)
            acc.append(0)

    bar_width = 1 / num_bins
    bbins = np.linspace(bar_width / 2, 1 - bar_width / 2, num_bins)
    x = np.linspace(bar_width / 2, 1 - bar_width / 2, num_bins)
    bar_width = 1 / num_bins

    ECE = torch.round(ECE, decimals=3)
    ece = round(ECE.numpy().tolist(), 3)

    accuracy = torch.round(accuracy, decimals=3)
    accuracy = round(accuracy.numpy().tolist(), 3)

    plt.figure()
    left, width = 0.1, 0.8
    bottom, height = 0.1, 0.1
    bottom_h = bottom + height + 0.05
    line1 = [left, bottom, width, 0.23]
    line2 = [left, 0.4, width, 0.5]
    ax1 = plt.axes(line2) #upper
    ax2 = plt.axes(line1) #below

    ax1.grid(True, linestyle='dashed', alpha=0.5)
    ax1.set_xlim(0.4, 1)

    ax1.set_ylabel('Test Accuracy / Test Confidence')
    ax1.bar(x, conf, bar_width, align='center', facecolor='r', edgecolor='black', label='Gap', hatch='/', alpha=0.3)
    ax1.bar(x, acc, bar_width, align='center', facecolor='b', edgecolor='black', label='Outputs', alpha=0.75)
    ax1.text(0.7, 0.25, r'ECE={}'.format(ece), fontsize=16, bbox=dict(facecolor='lightskyblue', alpha=0.9))
    # ax1.text(0.7, 0.25, r'MCE={}'.format(b), fontsize=16, bbox=dict(facecolor='lightskyblue', alpha=0.9))
    ax1.text(0.7, 0.1, r'Acc={}'.format(accuracy), fontsize=16, bbox=dict(facecolor='lightskyblue', alpha=0.9))
    ax1.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=3)
    ax1.legend()
    if network_type == 'fnn':
        ax1.set_title('FNN')
    elif network_type == 'mmce':
        ax1.set_title('CF')
    elif network_type == 'cm':
        ax1.set_title('FNN-CM')
    elif network_type == 'cf-cm':
        ax1.set_title('CF-CM')
    elif network_type == 'cm_mmce':
        ax1.set_title('CAFNN-MMCE-CM')
    elif network_type == 'cm_esd':
        ax1.set_title('CAFNN-ESD-CM')

    ax2.set_xlabel('Confidence')
    ax2.bar(bbins, nsamples_each_interval, bar_width, align='center', facecolor='blue', edgecolor='black', label='Gap', alpha=0.7)
    ax2.grid(True, linestyle='dashed', alpha=0.5)
    ax2.set_xlim(0.4, 1)
    ax2.set_ylabel('Sampling frequency')
    plt.savefig('./fnn_20.jpeg', dpi=1000)
    plt.show()

def confidence_distribution_plot(confidence_id, confidence_ood, network_type, num_bins=15):
    bins = np.linspace(0, 1, num_bins + 1)
    total_id = len(confidence_id)
    total_ood = len(confidence_ood)
    nsamples_each_interval_id = []
    nsamples_each_interval_ood = []
    for i in range(0, num_bins):
        temp_id = np.where((bins[i] <= confidence_id) & (confidence_id < bins[i + 1]))
        nsamples_each_interval_id.append((len(temp_id[0]) / total_id) )

        temp_ood = np.where((bins[i] <= confidence_ood) & (confidence_ood < bins[i + 1]))
        nsamples_each_interval_ood.append((len(temp_ood[0]) / total_ood))
    bar_width = 1 / num_bins
    bbins = np.linspace(bar_width / 2, 1 - bar_width / 2, num_bins)

    TV_distance = torch.round((1+sum(abs(torch.tensor(nsamples_each_interval_id) - torch.tensor(nsamples_each_interval_ood))) / 2) / 2, decimals=4)

    plt.figure()
    plt.text(0.5, 0.3, r'P_ood={}'.format(round(torch.Tensor.tolist(TV_distance), 3)), ha='center', va='center', fontsize=16, bbox=dict(facecolor='lightskyblue', alpha=0.9))
    plt.bar(bbins, nsamples_each_interval_id, bar_width, align='center', facecolor='blue', edgecolor='black',
            label='id', alpha=0.4)
    plt.bar(bbins, nsamples_each_interval_ood, bar_width, align='center', facecolor='red', edgecolor='black',
            label='ood', alpha=0.4)
    plt.xlim(0.0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='dashed', alpha=0.5)
    plt.xlabel('Confidence')
    plt.ylabel('Sampling frequency')
    plt.legend()
    if network_type == 'fnn':
        plt.title('FNN')
    elif network_type == 'mmce':
        plt.title('CF')
    elif network_type == 'cm':
        plt.title('FNN-CM')
    elif network_type == 'cf-cm':
        plt.title('CF-CM')
    elif network_type == 'cm_mmce':
        plt.title('CAFNN-MMCE-CM')
    elif network_type == 'cm_esd':
        plt.title('CAFNN-ESD-CM')
    plt.savefig('./confidence_distribution.jpeg', dpi=1000)
    plt.show()



class ESD(nn.Module):

    def __init__(self):
        super(ESD, self).__init__()

    def ESD(self, cal_logits, cal_labels):
        confidence1, logits = torch.max(nn.functional.softmax(cal_logits, dim=1), dim=1)
        correct = torch.eq(logits, cal_labels)
        N1 = len(confidence1)  #
        val = correct.float() - confidence1  #
        val = val.view(1, N1)
        mask = torch.ones(N1, N1) - torch.eye(N1)
        mask = mask.to(device)
        confidence1_matrix = confidence1.expand(N1, N1)  # row copying
        temp = (confidence1.view(1, N1).T).expand(N1, N1)
        tri = torch.le(confidence1_matrix, temp).float()
        val_matrix = val.expand(N1, N1)
        x_matrix = torch.mul(val_matrix, tri) * mask
        mean_row = torch.sum(x_matrix, dim=1) / (N1 - 1)  # gbar _i
        x_matrix_squared = torch.mul(x_matrix, x_matrix)
        var = 1 / (N1 - 2) * torch.sum(x_matrix_squared, dim=1) - (N1 - 1) / (N1 - 2) * torch.mul(mean_row, mean_row)
        d_k_sq_vector = torch.mul(mean_row, mean_row) - var / (N1 - 1)
        esd_loss = torch.sum(d_k_sq_vector) / N1

        return esd_loss
class MMCE(nn.Module):
    """
    Computes MMCE_m loss.
    """

    def __init__(self, device):
        super(MMCE, self).__init__()
        self.device = device

    def torch_kernel(self, matrix):
        return torch.exp(-1.0 * torch.abs(matrix[:, :, 0] - matrix[:, :, 1]) / (0.4))

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

        target = target.view(
            -1
        )  # For CIFAR-10 and CIFAR-100, target.shape is [N] to begin with

        predicted_probs = F.softmax(input, dim=1)
        predicted_probs, pred_labels = torch.max(predicted_probs, 1)
        correct_mask = torch.where(
            torch.eq(pred_labels, target),
            torch.ones(pred_labels.shape).to(self.device),
            torch.zeros(pred_labels.shape).to(self.device),
        )

        c_minus_r = correct_mask - predicted_probs

        dot_product = torch.mm(c_minus_r.unsqueeze(1), c_minus_r.unsqueeze(0))

        prob_tiled = (
            predicted_probs.unsqueeze(1)
            .repeat(1, predicted_probs.shape[0])
            .unsqueeze(2)
        )
        prob_pairs = torch.cat([prob_tiled, prob_tiled.permute(1, 0, 2)], dim=2)

        kernel_prob_pairs = self.torch_kernel(prob_pairs)

        numerator = dot_product * kernel_prob_pairs
        # return torch.sum(numerator)/correct_mask.shape[0]**2
        return torch.sum(numerator) / torch.pow(
            torch.tensor(correct_mask.shape[0]).type(torch.FloatTensor), 2
        )

class MMCE_weighted_undiff(nn.Module):
    """
    Computes MMCE_w loss.
    """

    def __init__(self, device):
        super(MMCE_weighted_undiff, self).__init__()
        self.device = device

    def torch_kernel(self, matrix):
        return torch.exp(-1.0 * torch.abs(matrix[:, :, 0] - matrix[:, :, 1]) / (0.4))

    def get_pairs(self, tensor1, tensor2):
        correct_prob_tiled = (
            tensor1.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2)
        )
        incorrect_prob_tiled = (
            tensor2.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)
        )

        correct_prob_pairs = torch.cat(
            [correct_prob_tiled, correct_prob_tiled.permute(1, 0, 2)], dim=2
        )
        incorrect_prob_pairs = torch.cat(
            [incorrect_prob_tiled, incorrect_prob_tiled.permute(1, 0, 2)], dim=2
        )

        correct_prob_tiled_1 = (
            tensor1.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)
        )
        incorrect_prob_tiled_1 = (
            tensor2.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2)
        )

        correct_incorrect_pairs = torch.cat(
            [correct_prob_tiled_1, incorrect_prob_tiled_1.permute(1, 0, 2)], dim=2
        )
        return correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs

    def get_out_tensor(self, tensor1, tensor2):
        return torch.mean(tensor1 * tensor2)

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

        target = target.view(
            -1
        )  # For CIFAR-10 and CIFAR-100, target.shape is [N] to begin with

        predicted_probs = F.softmax(input, dim=1)
        predicted_probs, predicted_labels = torch.max(predicted_probs, 1)

        correct_mask = torch.where(
            torch.eq(predicted_labels, target),
            torch.ones(predicted_labels.shape).to(self.device),
            torch.zeros(predicted_labels.shape).to(self.device),
        )

        k = torch.sum(correct_mask).type(torch.int64)
        k_p = torch.sum(1.0 - correct_mask).type(torch.int64)
        cond_k = torch.where(
            torch.eq(k, 0),
            torch.tensor(0).to(self.device),
            torch.tensor(1).to(self.device),
        )
        cond_k_p = torch.where(
            torch.eq(k_p, 0),
            torch.tensor(0).to(self.device),
            torch.tensor(1).to(self.device),
        )
        k = (
            torch.max(k, torch.tensor(1).to(self.device)) * cond_k * cond_k_p
            + (1 - cond_k * cond_k_p) * 2
        )
        k_p = torch.max(k_p, torch.tensor(1).to(self.device)) * cond_k_p * cond_k + (
            (1 - cond_k_p * cond_k) * (correct_mask.shape[0] - 2)
        )

        correct_prob, _ = torch.topk(predicted_probs * correct_mask, int(k.item()))
        incorrect_prob, _ = torch.topk(predicted_probs * (1 - correct_mask), int(k_p.item()))

        (
            correct_prob_pairs,
            incorrect_prob_pairs,
            correct_incorrect_pairs,
        ) = self.get_pairs(correct_prob, incorrect_prob)

        correct_kernel = self.torch_kernel(correct_prob_pairs)
        incorrect_kernel = self.torch_kernel(incorrect_prob_pairs)
        correct_incorrect_kernel = self.torch_kernel(correct_incorrect_pairs)

        sampling_weights_correct = torch.mm(
            (1.0 - correct_prob).unsqueeze(1), (1.0 - correct_prob).unsqueeze(0)
        )

        correct_correct_vals = self.get_out_tensor(
            correct_kernel, sampling_weights_correct
        )
        sampling_weights_incorrect = torch.mm(
            incorrect_prob.unsqueeze(1), incorrect_prob.unsqueeze(0)
        )

        incorrect_incorrect_vals = self.get_out_tensor(
            incorrect_kernel, sampling_weights_incorrect
        )
        sampling_correct_incorrect = torch.mm(
            (1.0 - correct_prob).unsqueeze(1), incorrect_prob.unsqueeze(0)
        )

        correct_incorrect_vals = self.get_out_tensor(
            correct_incorrect_kernel, sampling_correct_incorrect
        )

        correct_denom = torch.sum(1.0 - correct_prob)
        incorrect_denom = torch.sum(incorrect_prob)

        m = torch.sum(correct_mask)
        n = torch.sum(1.0 - correct_mask)
        mmd_error = 1.0 / (m * m + 1e-5) * torch.sum(correct_correct_vals)
        mmd_error += 1.0 / (n * n + 1e-5) * torch.sum(incorrect_incorrect_vals)
        mmd_error -= 2.0 / (m * n + 1e-5) * torch.sum(correct_incorrect_vals)

        # print(cond_k * cond_k_p)
        return torch.max(
            (cond_k * cond_k_p).type(torch.FloatTensor).to(self.device).detach()
            * torch.sqrt(mmd_error + 1e-10),
            torch.tensor(0.0).to(self.device),
        )


class MMCE_weighted_diff(nn.Module):
    def __init__(self, device):
        super(MMCE_weighted_diff, self).__init__()
        self.device = device
        self.temperature = 0.001
        self.temperature_a = 0.01

    def torch_kernel(self, matrix, n_bins):
        auto_kernel_width = 0.4 * 20/n_bins
        # auto_kernel_width = 0.8
        # return torch.exp(-1.0 * torch.abs(matrix[:, :, 0] - matrix[:, :, 1]) / (auto_kernel_width))
        return torch.exp(-1.0*torch.abs(matrix[:, :, 0] - matrix[:, :, 1])/(auto_kernel_width))

    def r_hat(self, logits): # we use one softmax layer with temperature parameter to obtain r_hat, follow formula 3
        # logits is a tensor data type, is the output of neural network without softmax layer
        sottmaxes = F.softmax(logits, dim=1)
        argmax = F.softmax(sottmaxes/self.temperature, dim=1)
        r_hat = (sottmaxes * argmax).sum(1)
        return r_hat

    def c_hat(self, logits, target): # we use this function to obtain c_hat
        # logits is the output of neural network without softmax layer
        # target is the label of the training dataset
        label_size = logits.size(1)
        target_size = target.size(0)
        logits = F.softmax(logits, dim=1)
        R_hat = torch.ones(logits.shape)
        for i in range(0, label_size):
            temp_1 = logits[:, i]
            temp_1 = temp_1.unsqueeze(1).repeat(1, label_size - 1) # repeat temp_1 twice
            if i == 0:
                temp_2 = logits[:, 1:label_size]
            elif i == label_size:
                temp_2 = logits[:, 0:label_size-1]
            else:
                left = logits[:, 0:i].reshape(logits.size(0), i)
                right = logits[:, (i + 1):label_size].reshape(logits.size(0), label_size - i - 1)
                temp_2 = torch.cat((left, right), 1)
            sy = temp_1 - temp_2 # here, we use this to obtain the Sy,y′ = p(y | x, θ) − p(y′ | x, θ).
            b = torch.exp(-sy / self.temperature_a) / (1 + torch.exp(-sy / self.temperature_a)) # here we want to calculate formula 5, and 0.01 is the temperature parameter \tau_a of formula 5
            R_hat[:, i] = 1 + torch.sum(b, dim=1)
        R_hat = torch.round(R_hat) # since the elements of R_hat are sth like 2.000007, 0.99998, so we use round to obtain the int
        prediction_matrix = F.relu(2 - R_hat) # here is the formula 6, through this to obtain R_hat

        c_hat = torch.zeros(target_size, dtype=torch.float)
        for i in range(0, target_size): # combine the label of target, we can know whether the output is correct, that is accuracy c_hat
            if torch.eq(prediction_matrix[i][target[i]], torch.tensor(1, dtype=torch.int)):
                c_hat[i] = torch.tensor(1, dtype=torch.float)
        correct_number = torch.tensor(torch.nonzero(c_hat).size(0))
        total_number = torch.tensor(c_hat.size(0))
        return c_hat, correct_number, total_number

    def get_pairs(self, tensor1, tensor2): # position 1 correct_prob, position 2 incorrect_prob
        correct_prob_tiled = tensor1.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2) # 此处扩充到三维是为了实现pairs元素之间的一一对应，体现在kernel的计算中
        incorrect_prob_tiled = tensor2.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)

        correct_prob_pairs = torch.cat([correct_prob_tiled, correct_prob_tiled.permute(1, 0, 2)],
                                    dim=2)
        incorrect_prob_pairs = torch.cat([incorrect_prob_tiled, incorrect_prob_tiled.permute(1, 0, 2)],
                                    dim=2)

        correct_prob_tiled_1 = tensor1.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)
        incorrect_prob_tiled_1 = tensor2.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2)

        correct_incorrect_pairs = torch.cat([correct_prob_tiled_1, incorrect_prob_tiled_1.permute(1, 0, 2)],
                                    dim=2)
        return correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs

    def get_out_tensor(self, tensor1, tensor2):
        return torch.sum(tensor1*tensor2)

    def forward(self, input, target, n_bins):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        target = target.view(-1)  #For CIFAR-10 and CIFAR-100, target.shape is [N] to begin with

        predicted_probs = self.r_hat(input) # here we use our differentiable way to ontain r_hat
        predicted_probs = predicted_probs.to(device)
        # print(predicted_probs.dtype)



        """
        what makes me confuse is that even the r_hat are same for differentiable and undifferentiable way, but the finally ECE output are different.
        """

        correct_mask, _, _ = self.c_hat(input, target)
        correct_mask = correct_mask.to(device)
        # print(correct_mask.dtype)

        k = torch.sum(correct_mask).type(torch.int64) # the number of correct samples
        k_p = torch.sum(1.0 - correct_mask).type(torch.int64) # the number of incorrect samples

        cond_k = torch.where(torch.eq(k,0),torch.tensor(0).to(self.device),torch.tensor(1).to(self.device))  # result is 1 don't know why use this
        cond_k_p = torch.where(torch.eq(k_p,0),torch.tensor(0).to(self.device),torch.tensor(1).to(self.device)) # result is 1 don't know why use this

        correct_prob, _ = torch.topk(predicted_probs*correct_mask, k) # 注意这个方法，确实可以用这个剔除序列中的0元素，构成新序列 torch.topk
        # print(correct_prob)
        # correct_prob1, _ = torch.topk(predicted_probs1 * correct_mask, k)
        # print(correct_prob1)
        # sadasd
        incorrect_prob, _ = torch.topk(predicted_probs*(1 - correct_mask), k_p)

        correct_prob_pairs, incorrect_prob_pairs,\
               correct_incorrect_pairs = self.get_pairs(correct_prob, incorrect_prob)

        correct_kernel = self.torch_kernel(correct_prob_pairs, n_bins)
        incorrect_kernel = self.torch_kernel(incorrect_prob_pairs, n_bins)
        correct_incorrect_kernel = self.torch_kernel(correct_incorrect_pairs, n_bins)

        sampling_weights_correct = torch.mm((1.0 - correct_prob).unsqueeze(1), (1.0 - correct_prob).unsqueeze(0))

        correct_correct_vals = self.get_out_tensor(correct_kernel,
                                                          sampling_weights_correct)

        sampling_weights_incorrect = torch.mm(incorrect_prob.unsqueeze(1), incorrect_prob.unsqueeze(0))

        incorrect_incorrect_vals = self.get_out_tensor(incorrect_kernel,
                                                          sampling_weights_incorrect)
        sampling_correct_incorrect = torch.mm((1.0 - correct_prob).unsqueeze(1), incorrect_prob.unsqueeze(0))

        correct_incorrect_vals = self.get_out_tensor(correct_incorrect_kernel,
                                                          sampling_correct_incorrect)

        m = torch.sum(correct_mask) # correct sample
        n = torch.sum(1.0 - correct_mask) # incorrect number sample

        mmd_error = 1.0/(m*m + 1e-5) * torch.sum(correct_correct_vals)
        mmd_error += 1.0/(n*n + 1e-5) * torch.sum(incorrect_incorrect_vals)
        mmd_error -= 2.0/(m*n + 1e-5) * torch.sum(correct_incorrect_vals)
        return torch.max((cond_k*cond_k_p).type(torch.FloatTensor).to(self.device).detach()*torch.sqrt(mmd_error + 1e-10), torch.tensor(0.0).to(self.device))
        # return (cond_k*cond_k_p).type(torch.FloatTensor).to(self.device).detach()*torch.sqrt(mmd_error + 1e-10)

