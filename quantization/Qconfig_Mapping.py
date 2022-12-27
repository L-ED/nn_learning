import torch
from torch.ao.quantization import QConfig
import torch.ao.quantization.observer as Observers
import torch.ao.quantization.fake_quantize as Quantizers
import torch.ao.quantization.quantize_fx as quantize_fx
import copy
import sys

# , default_symmetric_qnnpack_qconfig, default_per_channel_symmetric_qnnpack_qconfig
from torch.ao.quantization import get_default_qconfig_mapping

_FIXED_QPARAMS_OP_TO_OBSERVER = [
    torch.nn.Hardsigmoid,
    torch.nn.functional.hardsigmoid,
    "hardsigmoid",
    "hardsigmoid_",
    torch.nn.Sigmoid,
    torch.sigmoid,
    "sigmoid",
    "sigmoid_",
    torch.nn.Softmax,
    torch.nn.Tanh,
    torch.tanh,
    "tanh",
    "tanh_",
]


default_symmetric_per_channel_conf = QConfig(
    activation=Observers.MovingAverageMinMaxObserver.with_args(
        qscheme=torch.per_tensor_symmetric,
        quant_min=-127,
        quant_max=127,
        reduce_range=True,
        dtype=torch.qint8,
        eps=0.0244140625
    ),
    weight=Observers.MovingAveragePerChannelMinMaxObserver.with_args(
        quant_min=-127,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        reduce_range=True,
        ch_axis=0,
        eps=0.0244140625
    ))


default_symmetric_per_tensor_conf = QConfig(
    activation=Observers.MovingAverageMinMaxObserver.with_args(
        qscheme=torch.per_tensor_symmetric,
        quant_min=-127,
        quant_max=127,
        dtype=torch.qint8,
        reduce_range=True,
        eps=0.0244140625
    ),
    weight=Observers.MovingAverageMinMaxObserver.with_args(
        qscheme=torch.per_tensor_symmetric,
        quant_min=-127,
        quant_max=127,
        reduce_range=True,
        dtype=torch.qint8,
        eps=0.0244140625
    ))


def create_qconfig_mapping_NEW(per_channel_config=None, per_tensor_config=None):

    if per_channel_config is None:
        per_channel_config = default_symmetric_per_channel_conf
    if per_tensor_config is None:
        per_tensor_config = default_symmetric_per_tensor_conf

    PER_TENSOR_ONLY = [  # take otuside of function later
        torch.nn.functional.linear,
        torch.nn.modules.linear.Linear
    ]

    qconfig_mapping = get_default_qconfig_mapping("qnnpack") \
        .set_global(per_channel_config)

    for pattern in qconfig_mapping.object_type_qconfigs.keys():
        if pattern not in _FIXED_QPARAMS_OP_TO_OBSERVER:
            qconfig_mapping.set_object_type(pattern, per_channel_config)
            if pattern in PER_TENSOR_ONLY:
                qconfig_mapping.set_object_type(pattern, per_tensor_config)

    return qconfig_mapping