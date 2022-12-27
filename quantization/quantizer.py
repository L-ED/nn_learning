import torch
import numpy as np
import os
import copy

import torch.ao.quantization.quantize_fx as quantize_fx

from .quantize import create_qconfig_mapping, get_qnnpack_backend_config

# from .IVA_BAKCEND_CONFIG import get_qnnpack_TPU_backend_config as get_qnnpack_backend_config
from .Qconfig_Mapping import create_qconfig_mapping_NEW as create_qconfig_mapping

from ..learning import TRAINERS


def get_quantizer(
        task_type, 
        qconfig_mapping=None, 
        backend_config=None, 
        q_engine=None, 
        mode=None, 
        **trainer_kwargs):
    class Quantizer(TRAINERS[task_type]):

        def __init__(
                self, 
                trainer_kwargs, 
                qconfig_mapping=None, 
                backend_config=None, 
                q_engine=None, 
                mode=None):
            
            super().__init__(**trainer_kwargs)

            if q_engine is None:
                self.q_engine = "qnnpack"

            torch.backends.quantized.engine = q_engine

            if backend_config is None:
                self.backend_config = get_qnnpack_backend_config()

            if qconfig_mapping is None:
                self.qconfig_mapping = create_qconfig_mapping()

            if mode.lower()=="qat" or mode is None:
                self.prepare_ = quantize_fx.prepare_qat_fx
            elif mode.lower()=="ptq":
                self.prepare_ = quantize_fx.prepare_fx
                self.epochs = 1
                self.val_epoch = 1

            self.calibration_input = torch.rand(
                (1, *next(iter(self.trainloader))[0].shape[1:])
            )
            # print("SHAPE", self.calibration_input.shape)


        def prepare(self, model):
            return self.prepare_(
                model= model,
                qconfig_mapping=self.qconfig_mapping,
                example_inputs=self.calibration_input,
                backend_config=self.backend_config
            )

        
        def convert(self, model_calibrated):
            return quantize_fx.convert_fx(
                model_calibrated,
                backend_config= self.backend_config
            )

        
        def quantize(self):
            self.model = self.prepare(self.model)
            self.learn()
            if self.local_rank == 0:
                self.load_best_weights()

                print("loaded weights")
                self.validate(self)

                self.model.eval()

                print("evaluated")
                self.validate(self)

                self.model = self.convert(
                    self.model.cpu())

                print("converted")
                self.device=torch.device('cpu')
                self.validate(self)

                self.save_quantized()

                print("saved")
                self.validate(self)


        def save_quantized(self):
            quant_model_path = os.path.join(
                self.save_dir, 
                self.best_name.replace(
                    "prepared", "quantized"))
            torch.save(
                self.model.state_dict(),
                quant_model_path)


        def create_name(self, metric):
            macc = np.round(
                metric[self.metric].last, 4)
            loss = np.round(
                metric["loss"].last, 4)
            filename = f'prepared_{macc}_{loss}.pth'
    
            return filename


        def load_best_weights(self):
            best_path = os.path.join(
                self.save_dir, self.best_name)
            self.model.load_state_dict(
                torch.load(best_path))


        def check_out(self):
            imgs, lbls = next(iter(self.valloader))
            imgs = imgs.to(self.device)
            self.model.to(self.device)
            print("pred :", self.model(imgs))



    return Quantizer(
        trainer_kwargs= trainer_kwargs,
        qconfig_mapping= qconfig_mapping,
        backend_config= backend_config,
        q_engine= q_engine,
        mode=mode
    )





