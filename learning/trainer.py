import torch 
import os
import numpy as np

import logging

from ..metrics import MetricHistoryNew, METRICS

from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:

    def __init__(
        self,
        model, metric,
        distribute, cuda_devices,
        train_fn,val_fn,loss_fn,
        trainloader,valloader,
        optimizer,scheduler,
        save_dir,
        logger=None,
        debug = False,
        local_rank=None,
        epochs=1,val_epoch=1):

        self.distribute = distribute
        self.metric = metric
        self.trainloader = trainloader
        self.valloader = valloader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.save_dir = save_dir 

        self.train = train_fn
        self.validate = val_fn
        self.loss_fn = loss_fn

        self.best_name = ""

        if debug:
            self.epochs = 1
            self.val_epoch = 1
        else:
            self.epochs = epochs
            self.val_epoch = val_epoch
        
        self.metric_calculator = METRICS[metric] # add later ability to make multiple metrics

        self.logger = self.prepare_logger(logger)
        self.model, self.device = self.prepare_model(
            model, 
            cuda_devices, 
            local_rank
        )
        
        self.val_metrics = self.init_metrics(metric)
        self.train_metrics = self.init_metrics(metric)
        self.profiler = self.init_profiler(debug)


    def learn(self):
        for epoch in range(self.epochs):
            self.cur_epoch = epoch
            self.train(self)
            if epoch%self.val_epoch or epoch+1==self.epochs:
                self.validate(self)
                if self.val_metrics[self.metric].get_better():
                    new_best = self.create_name(self.val_metrics)
                    self.update_best(new_best)


    def prepare_model(self, model, cuda_devices, local_rank):
        if len(cuda_devices)>0:
            if len(cuda_devices)>1:
                if self.distribute: 
                    assert local_rank is not None
                    self.logger("using Distributed Data Parallel")
                    card_idx = cuda_devices[local_rank]
                    device = torch.device("cuda", card_idx)
                    torch.cuda.set_device(device)
                    model = DDP(model, [card_idx])
            else:
                card_idx = cuda_devices[0]
                device = torch.device("cuda", card_idx)
                model.to(device)
        else:
            device = torch.device("cpu")

        return model, device


    def prepare_logger(self, logger):
        if logger is None:
            logger = logging.getLogger("base")
            ch = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s:%(name)s:%(levelname)s:\n\t%(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger.info
    

    def init_metrics(self, metric):
        return {
            f"{metric}": MetricHistoryNew(),
            'loss': MetricHistoryNew()
        }

    
    def init_profiler(self, is_debug):
        if is_debug:
            return torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1, # check
                    warmup=1, # check
                    active=10 # check
                ),
                on_trace_ready = torch.profiler.tensorboard_trace_handler(self.save_dir),
                profile_memory = True,
                with_flops = True
            )
        else:
            return torch.profiler.profile()


    def prepare_batch(self, batch):
        imgs, target = batch
        imgs= imgs.to(self.device)
        target= target.to(self.device).long()
        return imgs, target


    def calculate_metric(self, predict, target):
        return self.metric_calculator(predict, target)


    def optimizer_step(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


    def finalize_epoch(self, mode):
        if mode == "train":
            metric = self.train_metrics
        elif mode == 'validate':
            metric = self.val_metrics

        metric[self.metric].finalize_epoch()
        metric["loss"].finalize_epoch()

        self.logger(
            f"EPOCH {self.cur_epoch}: "+\
            f"{mode}_loss: {metric['loss'].last}, "+\
            f"{self.metric}: {metric[self.metric].last}")


    def update_best(self, new_name):
        if self.best_name != '':
            old_best_path = os.path.join(self.save_dir, self.best_name)
            os.remove(old_best_path)

        new_best_path = os.path.join(self.save_dir, new_name)
        model_state_dict = self.get_state_dict()
        torch.save(model_state_dict, new_best_path)

        self.best_name = new_name

    
    def get_state_dict(self):
        if self.distribute:
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()


    def create_name(self, metric):

        macc = np.round(
            metric[self.metric].last, 4)
        loss = np.round(
            metric["loss"].last, 4)
        filename = f'{macc}_{loss}.pth'

        return filename