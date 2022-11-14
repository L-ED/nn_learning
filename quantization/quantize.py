import torch
from torch.ao.quantization import QConfig
import torch.ao.quantization.observer as Observers
import torch.ao.quantization.fake_quantize as Quantizers
import torch.ao.quantization.quantize_fx as quantize_fx
import copy
import sys

# , default_symmetric_qnnpack_qconfig, default_per_channel_symmetric_qnnpack_qconfig
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.backend_config import get_qnnpack_backend_config

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
        dtype=torch.qint8,
        eps=0.000244140625
    ),
    weight=Observers.MovingAveragePerChannelMinMaxObserver.with_args(
        quant_min=-127,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        reduce_range=False,
        ch_axis=0,
        eps=0.000244140625
    ))


default_symmetric_per_tensor_conf = QConfig(
    activation=Observers.MovingAverageMinMaxObserver.with_args(
        qscheme=torch.per_tensor_symmetric,
        quant_min=-127,
        quant_max=127,
        dtype=torch.qint8,
        eps=0.000244140625
    ),
    weight=Observers.MovingAverageMinMaxObserver.with_args(
        qscheme=torch.per_tensor_symmetric,
        quant_min=-127,
        quant_max=127,
        dtype=torch.qint8,
        eps=0.000244140625
    ))


def create_qconfig_mapping(per_channel_config=None, per_tensor_config=None):

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


def quant_prepare(
        model, inputs,
        qconfig_mapping=None, backend_config=None,
        engine=None, mode=None):

    if engine is None:
        engine = "qnnpack"
    torch.backends.quantized.engine = engine

    if backend_config is None:
        backend_config = get_qnnpack_backend_config()

    if qconfig_mapping is None:
        qconfig_mapping = create_qconfig_mapping()

    if mode == 'ptq':
        model_prepared = quantize_fx.prepare_fx(
            model,
            qconfig_mapping,
            inputs,
            backend_config=backend_config
        )

    elif mode == 'qat' or mode is None:
        model_prepared = quantize_fx.prepare_qat_fx(
            model,
            qconfig_mapping,
            inputs,
            backend_config=backend_config
        )

    return model_prepared


def quant_convert(model_calibrated, backend_config=None):

    if backend_config is None:
        backend_config = get_qnnpack_backend_config()

    model_calibrated.eval()
    model_quantized = quantize_fx.convert_fx(
        model_calibrated,
        backend_config=backend_config
    )
    return model_quantized


def quantize(model,
             device_list,
             trainloader,
             valloader=None,
             mode='qat',
             backend_config=None,
             qconfig_mapping=None,
             engine='qnnpack',
             epochs=10,
             optimizer=None,
             loss_fn=None,
             scheduler=None,
             val_epoch=None,
             save_dir=None,
             timing=False,
             train_func=None,
             val_func=None,
             logger=None
             ):

    inp = torch.rand((1, 3, 224, 224))  # fix later

    model_prepared = quant_prepare(model.cpu(), inp)

    model_calibrated = train_func(
        model=model_prepared,
        trainloader=trainloader,
        valloader=valloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        val_epoch=val_epoch,
        devices=device_list,
        save_dir=save_dir,
        timing=timing,
        val_func=val_func,
        logger=logger
    )

    # model_calibrated = model_calibrated.module.cpu()
    model_quantized = quant_convert(model_calibrated.cpu())

    return model_quantized
