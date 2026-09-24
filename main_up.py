
import argparse
import copy
import datetime
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))

ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):          # HERE inserted last => searched first
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_ROOT = os.environ.get("DOSCFN_DATA_ROOT") or ROOT

import backbones  # noqa: E402
from HAM import ResBlock_HAM  # noqa: E402
from src.spot import SPOT  # noqa: E402

CKPT_DIR = os.path.join(HERE, "checkpoints")
MATRIX_DIR = os.path.join(DATA_ROOT, "data_bsm1_pic", "data", "matrix_data")
RAW_XLSX_DIR = os.path.join(DATA_ROOT, "data_bsm1_32")

FALLBACK_CKPT_DIR = os.path.join(ROOT, "experiment", "main1", "checkpoints")


def read_ckpt(name):
    """Path to read: prefer ./checkpoints, else fall back to FALLBACK_CKPT_DIR (read-only)."""
    local = os.path.join(CKPT_DIR, name)
    if os.path.exists(local):
        return local
    fb = os.path.join(FALLBACK_CKPT_DIR, name)
    if FALLBACK_CKPT_DIR and os.path.exists(fb):
        print(f"[ckpt] {name} not found locally, reading from {fb}")
        return fb
    return local

def find_matrix_dir(*candidates):
    """Locate the signature-matrix directory that actually holds the *test_data folders.

    Tries, in order: the candidate itself -> candidate/data_bsm1_pic/data/matrix_data
    -> any sub-folder of the candidate containing *test_data (e.g. data_dry/).
    Returns None if nothing matches.
    """
    def has_data(d):
        try:
            return os.path.isdir(d) and any(x.endswith("test_data") for x in os.listdir(d))
        except OSError:
            return False

    for c in candidates:
        if not c:
            continue
        if has_data(c):
            return c
        nested = os.path.join(c, "data_bsm1_pic", "data", "matrix_data")
        if has_data(nested):
            return nested
        try:
            subs = sorted(os.path.join(c, d) for d in os.listdir(c))
        except OSError:
            continue
        for sub in subs:
            if has_data(sub):
                return sub
    return None


# backbone name -> short form used in file names
BK_ALIAS = {
    "wideresnet50": "wide50",
    "resnet50": "resnet50",
    "resnet101": "resnet100",
    "resnext50_32x4d": "resnext50",
    "resnext101": "resnext100",
}

# format tag of the self-contained checkpoint, validated on load
CKPT_FORMAT = "doscfn-ckpt-v1"

# metrics usable for early stopping; value says whether higher or lower is better
MONITORS = {
    "val_loss": "min",   # validation reconstruction loss (uses no labels)
}


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.preprocessing_modules = torch.nn.ModuleList()
        for _ in input_dims:
            self.preprocessing_modules.append(MeanMapper(output_dim))

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class Projection(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc", torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu", torch.nn.LeakyReLU(0.2))
        self.apply(init_weight)

    def forward(self, x):
        return self.layers(x)


class ForwardHook:
    def __init__(self, hook_dict, layer_name, last_layer_to_extract):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(layer_name == last_layer_to_extract)

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        return None


class LastLayerToExtractReachedException(Exception):
    pass


class NetworkFeatureAggregator(torch.nn.Module):
    def __init__(self, backbone, layers_to_extract_from, device, train_backbone=False):
        super(NetworkFeatureAggregator, self).__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        self.train_backbone = train_backbone
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}
        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(self.outputs, extract_layer, layers_to_extract_from[-1])
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    network_layer = network_layer[int(extract_idx)]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]
            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(network_layer[-1].register_forward_hook(forward_hook))
            else:
                self.backbone.hook_handles.append(network_layer.register_forward_hook(forward_hook))
        self.to(self.device)

    def forward(self, images, eval=True):
        self.outputs.clear()
        if self.train_backbone and not eval:
            self.backbone(images)
        else:
            with torch.no_grad():
                try:
                    _ = self.backbone(images)
                except LastLayerToExtractReachedException:
                    pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features


def embe(features, layers_to_extract_from, patchsize, patchstride,
         feature_dimensions, pretrain_embed_dimension, target_embed_dimension):
    patch_maker = PatchMaker(patchsize, stride=patchstride)
    preprocessing = Preprocessing(feature_dimensions, pretrain_embed_dimension)
    preadapt_aggregator = Aggregator(target_dim=target_embed_dimension)

    features = [features[layer] for layer in layers_to_extract_from]
    for i, feat in enumerate(features):
        if len(feat.shape) == 3:
            B, L, C = feat.shape
            features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

    features = [patch_maker.patchify(x, return_spatial_info=True) for x in features]
    patch_shapes = [x[1] for x in features]
    features = [x[0] for x in features]
    ref_num_patches = patch_shapes[0]

    for i in range(1, len(features)):
        _features = features[i]
        patch_dims = patch_shapes[i]
        _features = _features.reshape(
            _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
        )
        _features = _features.permute(0, -3, -2, -1, 1, 2)
        perm_base_shape = _features.shape
        _features = _features.reshape(-1, *_features.shape[-2:])
        _features = F.interpolate(
            _features.unsqueeze(1),
            size=(ref_num_patches[0], ref_num_patches[1]),
            mode="bilinear",
            align_corners=False,
        )
        _features = _features.squeeze(1)
        _features = _features.reshape(*perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
        _features = _features.permute(0, -2, -1, 1, 2, 3)
        _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
        features[i] = _features

    features = [x.reshape(-1, *x.shape[-3:]) for x in features]
    features = preprocessing(features)
    features = preadapt_aggregator(features)
    return features


def calculate_mask_index(kernel_length_now, largest_kernel_lenght):
    right_zero_mast_length = math.ceil((largest_kernel_lenght - 1) / 2) - math.ceil((kernel_length_now - 1) / 2)
    left_zero_mask_length = largest_kernel_lenght - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length + kernel_length_now


def creat_mask(number_of_input_channel, number_of_output_channel, kernel_length_now, largest_kernel_lenght):
    ind_left, ind_right = calculate_mask_index(kernel_length_now, largest_kernel_lenght)
    mask = np.ones((number_of_input_channel, number_of_output_channel, largest_kernel_lenght))
    mask[:, :, 0:ind_left] = 0
    mask[:, :, ind_right:] = 0
    return mask


def creak_layer_mask(layer_parameter_list):
    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list, init_weight_list, bias_list = [], [], []
    for i in layer_parameter_list:
        conv = torch.nn.Conv1d(in_channels=i[0], out_channels=i[1], kernel_size=i[2])
        ind_l, ind_r = calculate_mask_index(i[2], largest_kernel_lenght)
        big_weight = np.zeros((i[1], i[0], largest_kernel_lenght))
        big_weight[:, :, ind_l:ind_r] = conv.weight.detach().numpy()
        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)
        mask_list.append(creat_mask(i[1], i[0], i[2], largest_kernel_lenght))
    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)


def creak_layer_mask_dec(layer_parameter_list, len_in_layer):
    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list, init_weight_list, bias_list = [], [], []
    for i in layer_parameter_list:
        conv = torch.nn.ConvTranspose1d(in_channels=i[1], out_channels=i[0], kernel_size=i[2])
        ind_l, ind_r = calculate_mask_index(i[2], largest_kernel_lenght)
        big_weight = np.zeros((i[1], i[0], largest_kernel_lenght))
        big_weight[:, :, ind_l:ind_r] = conv.weight.detach().numpy()
        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)
        mask_list.append(creat_mask(i[1], i[0], i[2], largest_kernel_lenght))
    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.mean(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)


class build_layer_with_layer_parameter(nn.Module):
    def __init__(self, layer_parameters):
        super(build_layer_with_layer_parameter, self).__init__()
        os_mask, init_weight, init_bias = creak_layer_mask(layer_parameters)
        in_channels = os_mask.shape[1]
        out_channels = os_mask.shape[0]
        max_kernel_size = os_mask.shape[-1]
        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask), requires_grad=False)
        self.padding = nn.ConstantPad1d(
            (int((max_kernel_size - 1) / 2), int(max_kernel_size / 2)), 0
        )
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=max_kernel_size
        )
        self.conv1d.weight = nn.Parameter(torch.from_numpy(init_weight), requires_grad=True)
        self.conv1d.bias = nn.Parameter(torch.from_numpy(init_bias), requires_grad=True)
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X):
        self.conv1d.weight.data = self.conv1d.weight * self.weight_mask
        result_1 = self.padding(X)
        result_2 = self.conv1d(result_1)
        result_3 = self.bn(result_2)
        return F.relu(result_3)


class build_layer_with_layer_parameter_dec(nn.Module):
    def __init__(self, layer_parameters, len_in_layer, len_of_layers):
        super(build_layer_with_layer_parameter_dec, self).__init__()
        os_mask, init_weight, init_bias = creak_layer_mask_dec(layer_parameters, len_in_layer)
        in_channels = os_mask.shape[0]
        out_channels = os_mask.shape[1]
        max_kernel_size = os_mask.shape[-1]
        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask), requires_grad=False)
        self.padding = nn.ConstantPad1d(
            (int((max_kernel_size - 1) / 2), int(max_kernel_size / 2)), 0
        )
        p = int((int((max_kernel_size - 1) / 2) + int(max_kernel_size / 2) + max_kernel_size - 1) / 2)
        self.conv1d_t = torch.nn.ConvTranspose1d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=max_kernel_size, padding=p,
        )
        self.conv1d_t.weight = nn.Parameter(torch.from_numpy(init_weight), requires_grad=True)
        self.conv1d_t.bias = nn.Parameter(torch.from_numpy(init_bias), requires_grad=True)
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X):
        self.conv1d_t.weight.data = self.conv1d_t.weight * self.weight_mask
        result_1 = self.padding(X)
        result_2 = self.conv1d_t(result_1)
        result_3 = self.bn(result_2)
        return F.relu(result_3)


def get_out_channel_number(paramenter_layer, in_channel, prime_list):
    return int(paramenter_layer / (in_channel * sum(prime_list)))


def get_Prime_number_in_a_range(start, end):
    Prime_list = []
    for val in range(start, end + 1):
        prime_or_not = True
        for n in range(2, val):
            if (val % n) == 0:
                prime_or_not = False
                break
        if prime_or_not:
            Prime_list.append(val)
    return Prime_list


def generate_layer_parameter_list(start, end, paramenter_number_of_layer_list, in_channel=1):
    prime_list = get_Prime_number_in_a_range(start, end)
    if prime_list == []:
        print("start = ", start, "which is larger than end = ", end)
    input_in_channel = in_channel
    layer_parameter_list = []
    for paramenter_number_of_layer in paramenter_number_of_layer_list:
        out_channel = get_out_channel_number(paramenter_number_of_layer, in_channel, prime_list)
        tuples_in_layer = []
        for prime in prime_list:
            tuples_in_layer.append((in_channel, out_channel, prime))
        in_channel = len(prime_list) * out_channel
        layer_parameter_list.append(tuples_in_layer)
    tuples_in_layer_last = []
    first_out_channel = len(prime_list) * get_out_channel_number(
        paramenter_number_of_layer_list[0], input_in_channel, prime_list
    )
    tuples_in_layer_last.append((in_channel, first_out_channel, start))
    tuples_in_layer_last.append((in_channel, first_out_channel, start + 1))
    layer_parameter_list.append(tuples_in_layer_last)
    return layer_parameter_list


def inverse_list(get_list):
    return list(reversed(get_list))


def get_pram_in_os_cnn(data_channel, data_size, max_kernel_size=89, n_class=32):
    start_kernel_size = 1
    paramenter_number_of_layer_list = [8 * 128 * data_channel, 5 * 128 * 256 + 2 * 256 * 128]
    receptive_field_shape = min(int(data_size / 4), max_kernel_size)
    layer_parameter_list = generate_layer_parameter_list(
        start_kernel_size, receptive_field_shape, paramenter_number_of_layer_list,
        in_channel=data_channel,
    )
    return layer_parameter_list, n_class


# --------------------------------------------------------------------------
# ConvLSTM (temporal reconstruction branch)
# --------------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        assert hidden_channels % 2 == 0
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.padding = int((kernel_size - 1) / 2)
        self.Wxi = nn.Conv2d(input_channels, hidden_channels, kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(input_channels, hidden_channels, kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(input_channels, hidden_channels, kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(input_channels, hidden_channels, kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1, self.padding, bias=False)
        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], "Input Height Mismatched!"
            assert shape[1] == self.Wci.size()[3], "Input Width Mismatched!"
        return (
            torch.zeros(batch_size, hidden, shape[0], shape[1]).cuda(),
            torch.zeros(batch_size, hidden, shape[0], shape[1]).cuda(),
        )


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.internal_state = []
        for i in range(self.num_layers):
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size).cuda()
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = self._all_layers[i].init_hidden(
                        batch_size=bsize, hidden=self.hidden_channels[i], shape=(height, width)
                    )
                    internal_state.append((h, c))
                (h, c) = internal_state[i]
                x, new_c = self._all_layers[i](x, h, c)
                internal_state[i] = (x, new_c)
            if step in self.effective_step:
                outputs.append(x)
        return outputs, (x, new_c)

class os_cnn_AE_convlstm_Ham(nn.Module):
    def __init__(self, layer_parameter_list, n_class, w_sqr, c_convblstm, few_shot=True):
        super(os_cnn_AE_convlstm_Ham, self).__init__()
        self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.layer_parameter_list_dec = inverse_list(layer_parameter_list)
        self.layer_list = []
        self.layer_list_dec = []
        for i in range(len(layer_parameter_list)):
            self.layer_list.append(build_layer_with_layer_parameter(layer_parameter_list[i]))
        self.net = nn.Sequential(*self.layer_list)
        for i in range(len(self.layer_parameter_list_dec)):
            self.layer_list_dec.append(
                build_layer_with_layer_parameter_dec(
                    self.layer_parameter_list_dec[i],
                    len(self.layer_parameter_list_dec[i]),
                    len(self.layer_parameter_list_dec),
                )
            )
        self.net_dec = nn.Sequential(*self.layer_list_dec)
        self.averagepool = nn.AdaptiveAvgPool1d(1)
        self.conv_channel = w_sqr

        out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_numebr += final_layer_parameters[1]

        self.hidden_enc = nn.Linear(out_put_channel_numebr, n_class)
        self.hidden_dec = nn.Linear(n_class, out_put_channel_numebr)
        self.conv_lstm = ConvLSTM(
            input_channels=c_convblstm, hidden_channels=[32], kernel_size=3, step=3, effective_step=[2]
        )
        self.con_pre = nn.ConvTranspose2d(
            in_channels=32, out_channels=c_convblstm, kernel_size=1, padding=0, stride=1
        )
        self.ham1 = ResBlock_HAM(32)

    def forward(self, X):
        X = self.net(X)                 # OS-EB
        len1 = X.shape[2]
        X = self.averagepool(X)
        X = X.squeeze_(-1)
        X = self.hidden_enc(X)          # bottleneck
        X = self.hidden_dec(X)
        X = X.unsqueeze_(-1)
        X = X.repeat(1, 1, len1)
        X = self.net_dec(X)             # OS-DB -> spatial reconstruction

        b, c = X.shape[0], X.shape[1]
        w = h = self.conv_channel
        X1_4 = X.reshape(b, c, w, h)    # input of the temporal branch
        X1 = self.conv_lstm(X1_4)
        X1 = X1[0][0]
        X1 = self.con_pre(X1)           # adjust layer
        return X, X1_4, X1



def read_data_BSM1(path):

    creat_var = {}
    if path and os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith("xlsx"):
                creat_var[file.replace(".xlsx", "")] = pd.read_excel(os.path.join(path, file))
    if creat_var:
        return creat_var, list(creat_var.keys())

    keys = [d[:-len("test_data")] for d in os.listdir(MATRIX_DIR)
            if d.endswith("test_data") and os.path.isdir(os.path.join(MATRIX_DIR, d))]
    return {}, keys


def load_data(train_data_path, test_data_path):
    dataset = {}

    train_file_list = os.listdir(train_data_path)
    train_file_list.sort(key=lambda x: int(x[11:-4]))
    train_data = [np.load(os.path.join(train_data_path, obj)) for obj in train_file_list]
    dataset["train"] = torch.from_numpy(np.array(train_data)).float()

    test_data_all, label_all = [], []
    for i in range(len(test_data_path)):
        test_file_list = os.listdir(test_data_path[i])
        test_file_list.sort(key=lambda x: int(x[10:-4]))
        test_data = [np.load(os.path.join(test_data_path[i], obj)) for obj in test_file_list]

        if "good" in test_data_path[i]:
            label_all.append(np.zeros(len(test_data)))
        else:
            lab = np.zeros(len(test_data))
            lab[29:241] = 1          
            label_all.append(lab)
        test_data_all = test_data if i == 0 else test_data_all + test_data

    dataset["test"] = torch.from_numpy(np.array(test_data_all)).float()
    dataset["test_label"] = torch.from_numpy(np.array(label_all)).float()
    return dataset["train"], dataset["test"], dataset["test_label"]


def get_bsm1_data(data_path, mode, key_list):
    datalist = [s for s in key_list if mode in s]
    train_datapath = os.path.join(data_path, mode + "data_goodtrain_data")
    test_pathlist = [os.path.join(data_path, name + "test_data") for name in datalist]

    _train, _test, test_label = load_data(train_datapath, test_pathlist)
    ps = torch.nn.UpsamplingNearest2d(scale_factor=9)
    _train = ps(_train.squeeze(2))
    _test = ps(_test.squeeze(2))
    return _train, _test, test_label



def convlstm_state(model):
    return {f"cell{i}": {k: v.detach().cpu() for k, v in cell.state_dict().items()}
            for i, cell in enumerate(model.conv_lstm._all_layers)}


def load_convlstm_state(model, sd):
    if not sd:
        return 0
    n = 0
    for i, cell in enumerate(model.conv_lstm._all_layers):
        key = f"cell{i}"
        if key in sd:
            cell.load_state_dict(sd[key], strict=False)
            n += 1
    return n


def bundle_name(args, tag, monitor):
    return (f"DOSCFN_BSM1_{args.mode}_{tag}"
            f"_ad{args.adapter_layers}_best-{monitor}.pt")


def save_bundle(path, args, tag, model, pre_projection, monitor, best_value, epoch, metrics):
    payload = {
        "format": CKPT_FORMAT,
        # every hyper-parameter needed to rebuild the model on load
        "config": {
            "dataset": "BSM1",
            "mode": args.mode,
            "backbone": args.backbone,
            "layers": list(args.layers),
            "srf": args.srf,
            "adapter_layers": args.adapter_layers,
            "d_embe": args.d_embe,
            "input_shape": args.input_shape,
            "patchsize": args.patchsize,
            "patchstride": args.patchstride,
            "tag": tag,
        },
        "main_net": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "projection": {k: v.detach().cpu() for k, v in pre_projection.state_dict().items()},
        "conv_lstm": convlstm_state(model),
        "selection": {
            "monitor": monitor,
            "direction": MONITORS[monitor],
            "best_value": float(best_value),
            "epoch": int(epoch),
            "metrics": {k: float(v) for k, v in (metrics or {}).items()},
            "uses_test_labels": monitor != "val_loss",
        },
        "meta": {
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "torch": torch.__version__,
            "seed": args.seed,
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_bundle(path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device)
    if ck.get("format") != CKPT_FORMAT:
        raise ValueError(f"unrecognized checkpoint format: {ck.get('format')} (expect {CKPT_FORMAT})")
    cfg = ck["config"]

    cfg_args = argparse.Namespace(
        mode=cfg["mode"], backbone=cfg["backbone"], layers=cfg["layers"],
        srf=cfg["srf"], adapter_layers=cfg["adapter_layers"], d_embe=cfg["d_embe"],
        input_shape=cfg["input_shape"], patchsize=cfg["patchsize"],
        patchstride=cfg["patchstride"],
        batch_size=16, eval_batch_size=64, seed=ck.get("meta", {}).get("seed"),
    )
    feature_aggregator, feature_dimensions, model, pre_projection = build_model(cfg_args, device)
    model.load_state_dict(ck["main_net"])
    pre_projection.load_state_dict(ck["projection"])
    n_cell = load_convlstm_state(model, ck.get("conv_lstm"))

    sel = ck.get("selection", {})
    print(f"[load] {os.path.basename(path)}")
    print(f"       config: {cfg['mode']} / {cfg['backbone']} / {cfg['layers']} "
          f"/ SRF={cfg['srf']} / adapter={cfg['adapter_layers']}")
    print(f"       selected by {sel.get('monitor')} = {sel.get('best_value'):.6g} "
          f"@ epoch {sel.get('epoch')}"
          + ("   [uses test labels]" if sel.get("uses_test_labels") else ""))
    print(f"       conv_lstm cells restored: {n_cell}")
    return model, pre_projection, feature_aggregator, feature_dimensions, cfg_args, ck

class MetricEarlyStopping:
    def __init__(self, save_path, args, tag, monitor="val_loss",
                 patience=5, delta=0.0, min_epoch=10, verbose=True):
        if monitor not in MONITORS:
            raise ValueError(f"unknown monitor: {monitor}; choose from {list(MONITORS)}")
        self.save_path = save_path
        self.args = args
        self.tag = tag
        self.monitor = monitor
        self.direction = MONITORS[monitor]
        self.patience = patience
        self.delta = delta
        self.min_epoch = min_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = -1
        self.early_stop = False
        self.path = os.path.join(save_path, bundle_name(args, tag, monitor))

    def _better(self, score):
        if self.best_score is None:
            return True
        if self.direction == "min":
            return score <= self.best_score + self.delta
        return score >= self.best_score - self.delta

    def __call__(self, metrics, model, pre_projection, epoch):
        if epoch < self.min_epoch:
            return
        score = metrics[self.monitor]
        if self._better(score):
            if self.verbose:
                prev = "inf" if self.best_score is None else f"{self.best_score:.6g}"
                print(f"  {self.monitor} improved ({prev} -> {score:.6g}), saving checkpoint")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            save_bundle(self.path, self.args, self.tag, model, pre_projection,
                        self.monitor, score, epoch, metrics)
        else:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter} out of {self.patience} "
                  f"(best {self.monitor}={self.best_score:.6g} @ epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.early_stop = True


class EarlyStoppingDOSCFN:
    def __init__(self, save_path, tag, patience=5, verbose=True, delta=0, min_epoch=10):
        self.save_path = save_path
        self.tag = tag
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.min_epoch = min_epoch
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model1, model2, mode, epoch):
        if epoch < self.min_epoch:
            return
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model1, model2, mode)
        if score > self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model1, model2, mode)
            self.counter = 0

    def save_checkpoint(self, val_loss, model1, model2, mode):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(model1.state_dict(), os.path.join(self.save_path, f"best_networkbsm_{mode}main_netada_{self.tag}.pth"))
        torch.save(model2.state_dict(), os.path.join(self.save_path, f"best_networkbsm_{mode}projectionada_{self.tag}.pth"))
        self.val_loss_min = val_loss

def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False):
    """Point adjustment: one hit inside an anomaly segment marks the whole segment."""
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    predict = score > threshold if pred is None else pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                if not predict[j]:
                    predict[j] = True
                    latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    return predict


def calc_point2point(predict, actual):
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def pot_eval(init_score, score, label, q=1e-5):
    lms = 0.98
    while True:
        try:
            s = SPOT(q)
            s.fit(init_score, score)
            s.initialize(level=lms, min_extrema=False, verbose=False)
        except Exception:
            lms = lms * 0.999
        else:
            break
    ret = s.run(dynamic=False)
    pot_th = np.mean(ret["thresholds"]) * 1.0
    pred = adjust_predicts(score, label, pot_th, calc_latency=False)
    p_t = calc_point2point(pred, label)
    return {
        "f1": p_t[0], "precision": p_t[1], "recall": p_t[2],
        "TP": p_t[3], "TN": p_t[4], "FP": p_t[5], "FN": p_t[6],
        "threshold": pot_th,
    }, np.array(pred)

def build_model(args, device):
    backbone = backbones.load(args.backbone)
    backbone.name, backbone.seed = args.backbone, None
    feature_aggregator = NetworkFeatureAggregator(
        backbone, args.layers, device=device, train_backbone=False
    )  # backbone stays frozen, forward only
    feature_dimensions = feature_aggregator.feature_dimensions(input_shape=[3, 288, 288])

    layer_parameter_list, n_class = get_pram_in_os_cnn(
        args.input_shape, args.d_embe, max_kernel_size=args.srf
    )
    s_q = int(math.sqrt(args.d_embe))
    model = os_cnn_AE_convlstm_Ham(layer_parameter_list, n_class, s_q, args.input_shape).to(device)
    pre_projection = Projection(args.d_embe, args.d_embe, n_layers=args.adapter_layers, layer_type=0).to(device)
    return feature_aggregator, feature_dimensions, model, pre_projection


def forward_batch(data, feature_aggregator, feature_dimensions, model, pre_projection, args, device):
    batch_data = data.shape[0]
    data = data.to(device)
    feature_aggregator.eval()
    with torch.no_grad():
        feats = feature_aggregator(data)
    feats = embe(
        feats, args.layers, patchsize=args.patchsize, patchstride=args.patchstride,
        feature_dimensions=feature_dimensions,
        pretrain_embed_dimension=args.d_embe, target_embed_dimension=args.d_embe,
    )
    len_embe = feats.shape[1]
    data_1 = feats.reshape(batch_data, -1, len_embe)
    adapted = pre_projection(data_1)
    out_AE, data_reshape, data_convlstm = model(adapted)
    return data_1, out_AE, data_reshape, data_convlstm


@torch.no_grad()
def score_split(loader, feature_aggregator, feature_dimensions, model,
                pre_projection, args, device):
    l1f = nn.MSELoss(reduction="none")
    l2f = nn.MSELoss(reduction="none")
    chunks = []
    for x in loader:
        data_1, out_AE, data_reshape, data_convlstm = forward_batch(
            x, feature_aggregator, feature_dimensions, model, pre_projection, args, device
        )
        L1 = l1f(out_AE, data_1)
        L2 = l2f(data_reshape, data_convlstm)
        chunks.append((L1 + L2.view(L1.shape[0], L1.shape[1], -1)).cpu())
    out = torch.cat(chunks)
    return np.mean(np.array(out), axis=(1, 2))


def test_metrics(train_loader, test_loader, labels_1d, feature_aggregator,
                 feature_dimensions, model, pre_projection, args, device):
    model.eval()
    pre_projection.eval()
    tr = score_split(train_loader, feature_aggregator, feature_dimensions,
                     model, pre_projection, args, device)
    te = score_split(test_loader, feature_aggregator, feature_dimensions,
                     model, pre_projection, args, device)

    m = {}
    try:
        res, _ = pot_eval(tr, te, labels_1d)
        m["precision"] = res["precision"]
        m["recall"] = res["recall"]
        m["pa_f1"] = res["f1"]
        m["threshold"] = res["threshold"]
    except Exception as e:
        print(f"  [warn] POT failed ({e}); pa_f1 set to 0")
        m["precision"] = m["recall"] = m["pa_f1"] = 0.0
    return m


def train(args, device, tag):
    _, key_list = read_data_BSM1(RAW_XLSX_DIR)
    traindata, testdata, test_label = get_bsm1_data(MATRIX_DIR, args.mode, key_list)
    traindataloader = torch.utils.data.DataLoader(
        traindata, batch_size=args.batch_size, shuffle=True,
        prefetch_factor=2, pin_memory=True, drop_last=False,
    )

    # these two loaders are only needed when monitoring test metrics
    need_test = args.select != "val_loss"
    if need_test:
        eval_train_loader = torch.utils.data.DataLoader(
            traindata, batch_size=args.batch_size, shuffle=False,
            prefetch_factor=2, pin_memory=True, drop_last=False)
        eval_test_loader = torch.utils.data.DataLoader(
            testdata, batch_size=args.eval_batch_size, shuffle=False,
            prefetch_factor=2, pin_memory=True, drop_last=False)
        labels_1d = (np.sum(np.array(test_label.view(-1, 1)), axis=1) >= 1) + 0

    feature_aggregator, feature_dimensions, model, pre_projection = build_model(args, device)
    lossfunc = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    proj_opt = torch.optim.AdamW(pre_projection.parameters(), lr=args.lr)

    stopper = MetricEarlyStopping(CKPT_DIR, args, tag, monitor=args.select,
                                  patience=args.patience, min_epoch=args.min_epoch)
    legacy = EarlyStoppingDOSCFN(CKPT_DIR, tag, patience=args.patience) \
        if args.legacy_ckpt else None

    writer = None
    if not args.no_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            run_name = (f"{args.mode}_{tag}_srf{args.srf}_sel-{args.select}_"
                        f"{datetime.datetime.now().strftime('%m%d-%H%M%S')}")
            tb_dir = args.tb_dir or os.path.join(HERE, "runs", run_name)
            writer = SummaryWriter(tb_dir)
            writer.add_text("config", "  \n".join(f"{k}: {v}" for k, v in vars(args).items()))
            print(f"[tb] tensorboard --logdir {os.path.join(HERE, 'runs')}")
            print(f"[tb] run: {run_name}")
        except Exception as e:
            print(f"[tb] disabled ({e})")

    n_train_batch = args.train_batches
    print(f"[train] {len(traindata)} samples, {len(traindataloader)} batches/epoch, "
          f"first {n_train_batch} batches for training, rest for validation")
    print(f"[select] monitor={args.select} ({MONITORS[args.select]}), "
          f"patience={args.patience}, min_epoch={args.min_epoch}")
    if need_test:
        print(f"[select] WARNING: '{args.select}' is computed on the TEST set, "
              f"so model selection uses test labels.")
        print(f"[select] eval every {args.eval_every} epoch(s); "
              f"{len(testdata)} test samples per evaluation")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss_sum = val_loss_sum = 0.0
        train_num = val_num = 0

        for step, data in enumerate(traindataloader):
            if step < n_train_batch:
                model.train()
                pre_projection.train()
                proj_opt.zero_grad()
                data_1, out_AE, data_reshape, data_convlstm = forward_batch(
                    data, feature_aggregator, feature_dimensions, model, pre_projection, args, device
                )
                loss = lossfunc(out_AE, data_1) + lossfunc(data_reshape, data_convlstm)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                proj_opt.step()
                train_loss_sum += loss.item() * data.size(0)
                train_num += data.size(0)
            else:
                model.eval()
                pre_projection.eval()
                with torch.no_grad():
                    data_1, out_AE, data_reshape, data_convlstm = forward_batch(
                        data, feature_aggregator, feature_dimensions, model, pre_projection, args, device
                    )
                    loss1 = lossfunc(out_AE, data_1) + lossfunc(data_reshape, data_convlstm)
                val_loss_sum += loss1.item() * data.size(0)
                val_num += data.size(0)

        metrics = {
            "train_loss": train_loss_sum / max(train_num, 1),
            "val_loss": val_loss_sum / max(val_num, 1),
        }
        line = (f"epoch {epoch:3d}  train_loss {metrics['train_loss']:.6f}  "
                f"val_loss {metrics['val_loss']:.6f}")

        do_eval = need_test and (epoch % args.eval_every == 0 or epoch >= args.min_epoch)
        if do_eval:
            metrics.update(test_metrics(
                eval_train_loader, eval_test_loader, labels_1d, feature_aggregator,
                feature_dimensions, model, pre_projection, args, device))
            line += (f"  precision {metrics['precision']:.4f}"
                     f"  recall {metrics['recall']:.4f}  pa_f1 {metrics['pa_f1']:.4f}")
        dt = time.time() - t0
        line += f"  [{dt:.1f}s]"
        print(line)

        if writer is not None:
            writer.add_scalar("loss/train", metrics["train_loss"], epoch)
            writer.add_scalar("loss/val", metrics["val_loss"], epoch)
            for k in ("precision", "recall", "pa_f1"):
                if k in metrics:
                    writer.add_scalar(f"test/{k}", metrics[k], epoch)
            if "threshold" in metrics:
                writer.add_scalar("test/pot_threshold", metrics["threshold"], epoch)
            writer.add_scalar("time/epoch_sec", dt, epoch)
            if stopper.best_score is not None:
                writer.add_scalar(f"best/{args.select}", stopper.best_score, epoch)
            writer.flush()

        if args.select in metrics:
            stopper(metrics, model, pre_projection, epoch)
        if legacy is not None:
            legacy(metrics["val_loss"], model, pre_projection, args.mode, epoch)
        if stopper.early_stop:
            print("Early stopping")
            break

    if writer is not None:
        if stopper.best_score is not None:
            writer.add_hparams(
                {"mode": args.mode, "backbone": args.backbone, "srf": args.srf,
                 "adapter_layers": args.adapter_layers, "lr": args.lr,
                 "batch_size": args.batch_size, "select": args.select},
                {f"hparam/best_{args.select}": stopper.best_score,
                 "hparam/best_epoch": stopper.best_epoch})
        writer.close()

    if stopper.best_score is not None:
        print(f"\n[done] best {args.select} = {stopper.best_score:.6g} "
              f"@ epoch {stopper.best_epoch}")
        print(f"[done] {stopper.path}")
    else:
        print(f"\n[done] no checkpoint saved (training stopped before epoch {args.min_epoch})")


@torch.no_grad()
def evaluate(args, device, tag):
    _, key_list = read_data_BSM1(RAW_XLSX_DIR)
    traindata, testdata, test_label = get_bsm1_data(MATRIX_DIR, args.mode, key_list)

    train_dataloader = torch.utils.data.DataLoader(
        traindata, batch_size=args.batch_size, shuffle=False,
        prefetch_factor=2, pin_memory=True, drop_last=False,
    )
    test_dataloader = torch.utils.data.DataLoader(
        testdata, batch_size=args.eval_batch_size, shuffle=False,
        prefetch_factor=2, pin_memory=True, drop_last=False,
    )

    feature_aggregator, feature_dimensions, model, pre_projection = build_model(args, device)
    model.load_state_dict(torch.load(read_ckpt(f"best_networkbsm_{args.mode}main_netada_{tag}.pth")))
    pre_projection.load_state_dict(torch.load(read_ckpt(f"best_networkbsm_{args.mode}projectionada_{tag}.pth")))
    model.eval()
    pre_projection.eval()

    lossfunc1 = nn.MSELoss(reduction="none")
    lossfunc2 = nn.MSELoss(reduction="none")

    def collect(loader, name):
        chunks = []
        for step, x1 in enumerate(loader):
            data_1, out_AE, data_reshape, data_convlstm = forward_batch(
                x1, feature_aggregator, feature_dimensions, model, pre_projection, args, device
            )
            L1 = lossfunc1(out_AE, data_1)                                   
            L2 = lossfunc2(data_reshape, data_convlstm)                      
            chunks.append((L1 + L2.view(L1.shape[0], L1.shape[1], -1)).cpu())
        out = torch.cat(chunks)
        print(f"[eval] {name} score tensor {tuple(out.shape)}")
        return np.mean(np.array(out), axis=(1, 2))                            

    train_scores = collect(train_dataloader, "train")
    test_scores = collect(test_dataloader, "test")

    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(os.path.join(CKPT_DIR, f"best_networkbsm_{args.mode}train_loss{tag}.pkl"), "wb") as f:
        pickle.dump(train_scores, f)
    with open(os.path.join(CKPT_DIR, f"best_networkbsm_{args.mode}test_loss{tag}.pkl"), "wb") as f:
        pickle.dump(test_scores, f)
    np.save(os.path.join(CKPT_DIR, f"best_networkbsm_{args.mode}label{tag}.npy"),
            np.array(test_label.view(-1, 1)))
    return train_scores, test_scores, np.array(test_label.view(-1, 1))


def score(args, tag):
    with open(read_ckpt(f"best_networkbsm_{args.mode}train_loss{tag}.pkl"), "rb") as f:
        train_scores = pickle.load(f)
    with open(read_ckpt(f"best_networkbsm_{args.mode}test_loss{tag}.pkl"), "rb") as f:
        test_scores = pickle.load(f)

    label_path = read_ckpt(f"best_networkbsm_{args.mode}label{tag}.npy")
    if os.path.exists(label_path):
        labels = np.load(label_path)
    else:
        _, key_list = read_data_BSM1(RAW_XLSX_DIR)
        _, _, test_label = get_bsm1_data(MATRIX_DIR, args.mode, key_list)
        labels = np.array(test_label.view(-1, 1))

    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result, _ = pot_eval(train_scores, test_scores, labelsFinal)

    print("\n=========== DOSCFN on BSM1 ===========")
    print(f"mode={args.mode}  backbone={args.backbone}  layers={args.layers}")
    for k in ("f1", "precision", "recall"):
        print(f"  {k:<12} {result[k]}")
    return result


def set_seed(seed):
    """Fix every random source; returns the seed actually used.

    Identical to set_seed in experiment/repeat/baselines_repeat.py: covers
    random / numpy / torch (CPU+GPU), turns off cuDNN benchmark autotuning
    (which could otherwise pick different conv algorithms for the same seed)
    and turns on deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def parse_args():
    p = argparse.ArgumentParser(description="DOSCFN BSM1 main experiment")
    p.add_argument("--mode", default="dry", choices=["dry", "rain", "storm"], help="BSM1 operating condition")
    p.add_argument("--backbone", default="wideresnet50", choices=list(BK_ALIAS.keys()))
    p.add_argument("--layers", nargs="+", default=["layer2", "layer3"], help="backbone feature layers")
    p.add_argument("--srf", type=int, default=89, help="OS-CNN max kernel size")
    p.add_argument("--adapter-layers", type=int, default=2, help="number of feature adapter layers")
    p.add_argument("--d-embe", type=int, default=256, dest="d_embe", help="Dpre = Dembe")
    p.add_argument("--input-shape", type=int, default=1296, dest="input_shape", help="number of patches, 36x36")
    p.add_argument("--patchsize", type=int, default=3)
    p.add_argument("--patchstride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--eval-batch-size", type=int, default=64, dest="eval_batch_size")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-epoch", type=int, default=5, dest="min_epoch",
                   help="no early stopping and no saving for the first N epochs")
    p.add_argument("--select", default="val_loss", choices=list(MONITORS),
                   help="early stopping / model selection criterion; val_loss uses no labels")
    p.add_argument("--eval-every", type=int, default=1, dest="eval_every",
                   help="evaluate on the test set every N epochs (only when --select is not val_loss)")
    p.add_argument("--legacy-ckpt", action="store_true", dest="legacy_ckpt",
                   help="also save in the legacy two-file format (best_networkbsm_*.pth)")
    p.add_argument("--no-tb", action="store_true", dest="no_tb",
                   help="disable TensorBoard logging")
    p.add_argument("--tb-dir", default=None, dest="tb_dir",
                   help="TensorBoard log directory, default ./runs/<run name>")
    p.add_argument("--train-batches", type=int, default=37, dest="train_batches",
                   help="first N batches are used for training, the rest for validation")
    p.add_argument("--seed", type=int, default=1,
                   help="random seed, default 1. Fixes random/numpy/torch (CPU+GPU), "
                        "enables cuDNN deterministic and disables benchmark "
                        "(same as experiment/repeat; training gets slower)")
    p.add_argument("--data-root", default=DATA_ROOT, dest="data_root",
                   help="directory holding data_bsm1_pic / data_bsm1_32 "
                        "(env var DOSCFN_DATA_ROOT works too)")
    p.add_argument("--matrix-dir", default=None, dest="matrix_dir",
                   help="signature-matrix directory holding the *test_data / *train_data "
                        "folders; takes precedence over --data-root")
    p.add_argument("--xlsx-dir", default=None, dest="xlsx_dir",
                   help="raw xlsx directory (data_bsm1_32); if absent, condition names "
                        "are inferred from the signature-matrix folder names")
    p.add_argument("--fallback-ckpt-dir", default=FALLBACK_CKPT_DIR, dest="fallback_ckpt_dir",
                   help="read-only fallback directory used when a file is missing locally")
    p.add_argument("--no-fallback", action="store_true", dest="no_fallback",
                   help="disable the read-only fallback, use ./checkpoints only")
    p.add_argument("--train", action="store_true",
                   help="retrain (overwrites the weights in ./checkpoints). Without it, "
                        "existing weights are loaded and only evaluated")
    p.add_argument("--test", action="store_true",
                   help="kept for backwards compatibility; same as the default behaviour")
    p.add_argument("--score-only", action="store_true", dest="score_only",
                   help="skip training and the forward pass, score the existing loss pkl")
    return p.parse_args()


def main():
    args = parse_args()
    global FALLBACK_CKPT_DIR, DATA_ROOT, MATRIX_DIR, RAW_XLSX_DIR
    FALLBACK_CKPT_DIR = "" if args.no_fallback else (args.fallback_ckpt_dir or "")
    DATA_ROOT = args.data_root
    RAW_XLSX_DIR = args.xlsx_dir or os.path.join(DATA_ROOT, "data_bsm1_32")
    # resolution order: --matrix-dir > --data-root > the script's own directory
    MATRIX_DIR = args.matrix_dir or find_matrix_dir(DATA_ROOT, HERE) or         os.path.join(DATA_ROOT, "data_bsm1_pic", "data", "matrix_data")
    if not os.path.isdir(MATRIX_DIR):
        print(f"[warn] no signature-matrix directory found under {DATA_ROOT} or {HERE}; "
              f"pass --data-root / --matrix-dir")
    elif not args.score_only:
        print(f"[data] matrix_dir={MATRIX_DIR}")
    if args.seed is not None:
        set_seed(args.seed)
        print(f"[seed] seed={args.seed}  cudnn.benchmark=False  cudnn.deterministic=True")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ly = "layer23" if args.layers == ["layer2", "layer3"] else "".join(args.layers)
    bk = BK_ALIAS[args.backbone]
    tag = f"{ly}_{bk}"
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"device={device}  tag={tag}  ckpt_dir={CKPT_DIR}")

    if not args.score_only:
        if args.train:
            train(args, device, tag)
        evaluate(args, device, tag)
    score(args, tag)


if __name__ == "__main__":
    main()
