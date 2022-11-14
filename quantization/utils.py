import torch
import json
from platform_converter.converterfx.converter.main import ConverterTorch2Plat2


def get_fx_module(fx_node, fx_trace):
    for module in fx_node.meta["module_path"]:
        fx_trace = getattr(fx_trace, module)
    return fx_trace


class QAdd_observer:

    def __init__(self, obs_name) -> None:

        self.straight_scale = 0
        self.skip_con_scale = 0
        self.__name__ = obs_name

    def __call__(self, x1, x2):

        self.skip_con_shape = x1.shape
        self.straight_shape = x2.shape
        self.straight_scale = x1.q_scale()
        self.skip_con_scale = x2.q_scale()


def create_obs_name(model, name_prefix):
    found_index = False
    i = 0
    while hasattr(model, name_prefix + str(i)):
        i += 1

    return name_prefix + str(i)


def set_observer(model):

    name_prefix = "add_observer_"
    obs_name = create_obs_name(model, name_prefix)
    module = QAdd_observer(obs_name)
    setattr(model, obs_name, module)

    return obs_name


def pooling_observer(pooling_module, inp, out):
    pooling_module.featuremap_shape = inp[0].shape


def conv_observer(module, x, out):
    x_s = x[0].q_scale()

    try:
        w_s = module.weight().q_per_channel_scales()
    except RuntimeError:
        print("expected per channel conv but got per tensor")
        w_s = module.weight().q_scale()

    op_s = module.scale

    module.iva_scale = w_s*x_s/op_s


def linear_observer(module, x, out):
    x_s = x[0].q_scale()
    w_s = module.weight().q_scale()
    op_s = module.scale

    module.iva_scale = torch.tensor(
        [w_s*x_s/op_s for i in range(out.shape[1])])


def wrap_model(model, observers_dict):

    convs = ["QuantizedConv", 'QuantizedConvReLU',  # "QuantizedLinear"
             ]

    for node in model.graph.nodes:
        if node.op == "call_module":  # and "pool" in node.target:
            node.meta["module_path"] = node.target.split(".")
            node_module = get_fx_module(node, model)
            module_class = str(node_module).split("(")[0]

            if module_class in 'AdaptiveAvgPool2d':
                node_module.register_forward_hook(observers_dict['pool'])

            elif "QuantizedLinear" in module_class:
                node_module.register_forward_hook(linear_observer)

            elif any(conv in module_class for conv in convs):
                node_module.register_forward_hook(observers_dict['conv'])

        elif node.op == 'call_function' and "add" in node.target.__qualname__:
            with model.graph.inserting_before(node):
                attr_name = set_observer(model)
                node.meta['observer'] = attr_name
                obs_n = model.graph.create_node(
                    "call_module",
                    attr_name,
                    (node.args[:2]))

    model.recompile()


def observe_model(model, input_featuremap):

    observers_dict = {
        "pool": pooling_observer,
        "conv": conv_observer
    }

    wrap_model(model, observers_dict)
    _ = model(input_featuremap)


def convert_platform(model, input_featuremap, filepath_no_ext):

    observe_model(model, input_featuremap)

    converter = ConverterTorch2Plat2(
        binary_name=filepath_no_ext + ".bin",
        torch_model=model
    )

    json_content = converter.convert()
    jsonpath = filepath_no_ext + ".json"
    with open(jsonpath, "w") as jsonfile:
        json.dump(json_content, jsonfile)
