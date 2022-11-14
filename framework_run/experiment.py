from distutils.command.config import config
import os, sys, yaml
import torch

# sys.path.insert(0, "/storage_labs/3030/LyginE/projects/paradigma")
# import quant_framework_new

from ..utils import Creator
from quant_framework_new.learning import TRAINERS # refactor path, insert trainer into init
from quant_framework_new.quantization import get_quantizer

from quant_framework_new.utils import *

# Class for future experiments like pruning, distillation and etc.
class Experiment_One_Model(Creator):

    def __init__(
            self, 
            global_config, 
            exp_config,
            save_dir,
            logger, 
            prev_exp_name=None):

        self.global_config = global_config
        self.exp_config = exp_config
        self.prev_exp_name = prev_exp_name
        self.logger = logger
        self.save_dir = save_dir

        self.task = self.take_config("task")
        self.metric = self.take_config("metric")
        self.distribute= self.take_config("distribute")
        self.debug= self.take_config("debug")
        self.local_rank= self.take_local_rank()
        self.epochs= self.take_config("epochs")
        self.val_epoch= self.take_config("val_epoch")

        self.cuda_devices, self.cpu_devices = self.check_devices()
        self.model = self.create_model()
        self.trainloader, self.valloader = self.create_dataloaders()
        self.loss_function = self.create_loss()
        self.optimizer, self.scheduler = self.create_optimizer()


    def __call__(self):
        pass

    def take_local_rank(self):
        if self.distribute:
            return int(os.environ["LOCAL_RANK"])
        else:
            return None


    def create_dataloaders(self):

        data_config = self.take_config("dataset")
        trainset, valset = self.create_dataset(data_config, self.log_info)

        loader_conf = self.take_config("loader")
        if "train" in loader_conf: 
            trainloader_conf = data_config["loader"]["train"]
        else:
            trainloader_conf = loader_conf
        if "validation" in loader_conf:
            valloader_conf = data_config["loader"]["validation"]
        else:
            valloader_conf = loader_conf

        trainloader = self.create_loader(
            trainset, trainloader_conf)
        valloader = self.create_loader(
            valset, valloader_conf)
        return trainloader, valloader


    def create_model(self):
        # if first experiment in list, use defined resume path
        # for first experiment prev_exp_name would be None 
        model_config = self.take_config("model")
        if self.prev_exp_name is not None:
            resume_path = os.path.join(
                self.save_dir, 
                self.prev_exp_name
            )
            model_config["resume"] = resume_path
            self.log_info("Creating model resume path from prev experiment")
        return super().create_model(model_config, self.log_info)

    
    def create_loss(self):
        loss_config = self.take_config("loss")
        return super().create_loss(loss_config)


    def create_optimizer(self):
        optimizer_config = self.take_config("optimizer")
        return super().create_optimizer(
            self.model, 
            optimizer_config,
            self.log_info
        )


    def check_devices(self):
        device_config = self.take_config("device")
        return super().check_devices(device_config, self.log_info)


    def create_train(self):
        train_fn_config = self.take_config("train_function")
        return self.create_function(train_fn_config)


    def create_val(self):
        val_fn_config = self.take_config("val_function")
        return self.create_function(val_fn_config)

    
    def log_info(self, msg):
        if self.local_rank is None or self.local_rank == 0:
            self.logger.info(msg)



class Experiment_Train(Experiment_One_Model):

    def __init__(
            self, 
            global_config, 
            train_config,
            save_dir, 
            logger,
            prev_exp_name=None):

        super().__init__(
            global_config, 
            train_config,
            save_dir, 
            logger, 
            prev_exp_name)

        self.set_trainer()


    def __call__(self):
        self.trainer.learn()


    def set_trainer(self):
        self.trainer = TRAINERS[self.task](
            model = self.model,
            metric = self.metric,
            distribute= self.distribute,
            cuda_devices= self.cuda_devices,
            train_fn= self.create_train(),
            val_fn= self.create_val(),
            loss_fn= self.loss_function,
            trainloader=self.trainloader,
            valloader= self.valloader,
            optimizer= self.optimizer,
            scheduler= self.scheduler,
            save_dir= self.save_dir,
            logger= self.logger,
            debug= self.debug,
            local_rank= self.local_rank,
            epochs= self.epochs,
            val_epoch= self.val_epoch
        )



class Experiment_Quant(Experiment_One_Model):

    def __init__(
            self, 
            global_config,
            quant_config,
            save_dir, 
            logger,
            prev_exp_name=None):

        super().__init__(
            global_config, 
            quant_config,
            save_dir, 
            logger,
            prev_exp_name)

        self.qconfig_mapping = self.create_qconfig_mapping(
            self.take_config("qconfig"))
        
        self.backend_config = self.create_backend_config(
            self.take_config("backend_config"))

        self.q_engine= self.take_config("engine")
        self.mode = self.take_config("mode")
        self.set_quantizer()


    def __call__(self):
        self.quantizer.quantize()


    def set_quantizer(self):
        self.quantizer = get_quantizer(
            task_type= self.task,
            qconfig_mapping= self.qconfig_mapping,
            backend_config= self.backend_config,
            q_engine= self.q_engine,
            mode= self.mode,

            model = self.model,
            metric = self.metric,
            distribute= self.distribute,
            cuda_devices= self.cuda_devices,
            train_fn= self.create_train(),
            val_fn= self.create_val(),
            loss_fn= self.loss_function,
            trainloader=self.trainloader,
            valloader= self.valloader,
            optimizer= self.optimizer,
            scheduler= self.scheduler,
            save_dir= self.save_dir,
            logger= self.logger,
            debug= self.debug,
            local_rank= self.local_rank,
            epochs= self.epochs,
            val_epoch= self.val_epoch)


    def create_qconfig_mapping(self, quant_config):
        return None


    def create_backend_config(self, backend_config):
        return None




class Experimentator(Creator):

    def __init__(self, path_to_config):

        self.config = dict_from_yaml(path_to_config)
        self.save_dir = self.create_dir()
        self.logger = self.create_logger(self.save_dir)
        self.experimets_configs = self.config["experiments"].copy()


    def __call__(self):

        prev_exp_name = None
        for experiment_name, experiment_config in self.experimets_configs.items():

            self.logger.info(f"Initializing experiment {experiment_name}")

            experiment = self.initialize_experiment(
                experiment_name, experiment_config, prev_exp_name)

            if experiment.distribute:
                self.logger.info("Initializing DDP")
                self.initialize_process()
            
            self.logger.info("Starting experiment")
            experiment()

            if experiment.distribute:
                if experiment.local_rank == 0:
                    self.logger.info("Destroying DDP")
                self.destroy_process()

            prev_exp_name = experiment_name

            
    def create_dir(self):
        save_dir = self.config["save_dir"]
        t=datetime.now()
        subdirs =[
            '_'.join(f'{i}' for i in [t.year, t.month, t.day]),
            '_'.join(f'{i}' for i in [t.hour, t.minute, t.second])
        ]
        return self.create_nested_tree(save_dir, subdirs)


    def initialize_experiment(self, experiment_name, experiment_config, pre_exp_name):

        supported_experiments = ["quantize", "train"]
        assert experiment_name in supported_experiments

        self.logger.info(
            "Creating experiment subdirectory "+\
                f"with name {experiment_name} "+\
                f"in directory {self.save_dir}")

        save_dir = self.create_subdir(self.save_dir, experiment_name)

        if experiment_name in "train":
            return Experiment_Train(
                self.config,
                experiment_config,
                save_dir,
                self.logger,
                pre_exp_name
            )
        elif experiment_name in "quantize":
            return Experiment_Quant(
                self.config,
                experiment_config,
                save_dir,
                self.logger,
                pre_exp_name
            )

    def initialize_process(self):
        torch.distributed.init_process_group(backend="nccl")

    def destroy_process(self):
        torch.distributed.destroy_process_group()
