import sys, os, logging, copy

from .getters import *

from datetime import datetime
import torch, torchvision

# checked (check if it work for DDP), REMOVE DEVICE

class Creator:
    def __init__(self, config):

        self.config = config

    def check_devices(self, device_config, logger):

        # devices= self.config['device']
        devices = device_config
        cuda_devices=[]
        cpu_devices=[0]

        for dev_type, device_list in devices.items():
            device_list = sorted(list(set(device_list)))
            if dev_type=="cuda":
                cuda_devices = device_list
                visible_devices_num = torch.cuda.device_count()
                print("VISIBLE DEVICES", visible_devices_num)
                assert visible_devices_num>len(cuda_devices)
                logger(f"cuda devices: {cuda_devices}")

            elif dev_type=="cpu":
                cpu_devices = device_list
                logger(f"cpu devices: {cpu_devices}")

        return cuda_devices, cpu_devices
        

    def create_transform(self, transforms, logger):
        t_list=[]
        for transform in transforms:
            if isinstance(transform, str):
                t_list.append(get_transform(transform)())
            elif isinstance(transform, dict):
                name = list(transform.keys())[0]
                params = transform[name]
                in_params = []
                if isinstance(params, list):
                    for param in params:
                        if isinstance(param, list):
                            in_params.append(tuple(param))
                        elif isinstance(param, dict):
                            keyname = list(param.keys())[0]
                            val = param[keyname]
                            if keyname in ['mean', 'std']:
                                val= torch.tensor(val)
                            in_params.append(val)
                        else:
                            in_params.append(param)
                else:
                    in_params.append(params)
                
                t_list.append(get_transform(name)(*in_params))
        logger(f"Transforms are: {t_list}")
        return t_list


    def create_dataset(self, data_dict, logger):
        transforms_dict = {}
        for mode in ["train", "val"]:
            transforms_dict[mode]= torchvision.transforms.Compose(
                self.create_transform(
                    data_dict[f"{mode}_transform"],
                    logger
            ))
        annotations_root= data_dict['annotation']
        dataset_creator = get_dataset_creator()
        logger(
            "Creating Dataset "+\
            f"from annotations {annotations_root}")
        trainset, valset= dataset_creator( 
            annotation_path= annotations_root,
            transformations_dict=transforms_dict
        )
        return  trainset, valset


    def create_loader(self, dataset, loader_dict):
        workers= loader_dict['workers']
        shuffle= loader_dict['shuffle']
        drop_last= loader_dict['drop_last']
        batch= loader_dict[f'batch_size']

        dataloader= torch.utils.data.DataLoader(
            dataset,
            batch_size= batch, 
            shuffle=shuffle, 
            num_workers=workers, 
            drop_last=drop_last
        )
        return dataloader


    def create_model(self, model_dict, logger):
        # model_dict = self.config['model']
        model_name= model_dict['name']
        model_version= model_dict['version']
        model_source = model_dict["source"]
        if 'classes' in model_dict:
            classes = model_dict['classes']
        else:
            classes = None
        
        logger(
            f"Creating model {model_name} "+\
            f"version {model_version}")
        
        model = get_model(
            model_name, model_version, model_source, classes)

        if 'resume' in model_dict:
            self.resume_state(model, model_dict["resume"], logger)

        return model


    def resume_state(self, model, resume_path, logger, key='model_state'):
        logger(
            f'Resuming state from {resume_path}')

        assert os.path.exists(resume_path)
        
        if os.path.isdir(resume_path):
            logger(
                f"resume path {resume_path} is dir, choosing best"
            )
            print("files in dir", os.listdir(resume_path))
            best_file = sorted(os.listdir(resume_path))[0]
            resume_path = os.path.join(resume_path, best_file)
            logger(f"best is {best_file}")

        state_dict = torch.load(resume_path)
        try:
            if key in state_dict.keys():
                model.load_state_dict(state_dict[key])
            else:
                model.load_state_dict(state_dict)
        except:
            logger(f'!!! Failed to load checkpoint {resume_path}')


    def create_scheduler(self, scheduler_dict, optimizer):
        # scheduler_dict = self.config['scheduler']
        parameters_config = scheduler_dict["parameters"]
        for name, parameter in parameters_config.items():
            if isinstance(parameter, list):
                parameters_config[name] = tuple(parameter)
        scheduler= get_scheduler(scheduler_dict['name'])(
            optimizer, **parameters_config
        )
            
        return scheduler


    def create_optimizer(self, model, optimizer_dict, logger):
        parameters_config = optimizer_dict["parameters"]
        for name, parameter in parameters_config.items():
            if isinstance(parameter, list):
                parameters_config[name] = tuple(parameter)

        optimizer= get_optimizer(optimizer_dict['name'])(
            model.parameters(), **parameters_config
        )

        if 'resume' in optimizer_dict:
            resume_path = optimizer_dict['resume']
            self.resume_state(optimizer, resume_path, logger,'optimizer_state')

        scheduler = None
        if 'scheduler' in optimizer_dict.keys():
            scheduler= self.create_scheduler(
                optimizer_dict["scheduler"], optimizer)

        return optimizer, scheduler


    def create_loss(self, loss_dict):
        # loss_= self.config['loss']              
        loss_fn = get_loss(loss_dict['name'])()
        return loss_fn


    def create_logger(self, save_dir):


        try:
            log_config = self.config['log']
            logger_name = log_config['name']
        except:
            logger_name = 'learning_process'

        try:
            logger_level = log_config['level'].lower()
            if logger_level == 'info':
                logger_level = logging.INFO
            elif logger_level == 'warning':
                logger_level = logging.WARNING
            elif logger_level == 'error':
                logger_level = logging.ERROR
            else:
                logger_level = logging.DEBUG
            
        except:
            logger_level = logging.INFO
            
        logger = logging.getLogger(logger_name)
        logger.setLevel(logger_level)
        
        for handler_type, handler_config in log_config['output'].items():
            
            handler_type = handler_type.lower()
            path =  handler_config["out"]
            
            if handler_type == 'file':
                if path == 'default':
                    path = os.path.join(save_dir, 'experiment.log')
                log_h = logging.FileHandler(filename=path, mode='w')
            elif handler_type == 'console':
                log_h = logging.StreamHandler(sys.stdout)
                
            log_h.setLevel(logger_level)
            try:
                formatter = logging.Formatter(fmt=handler_config['messageformat'], 
                                            datefmt=handler_config['dateformat'])
            except:
                formatter = logging.Formatter(fmt='%(asctime)s | %(levelname)s \n%(message)s\n', 
                                            datefmt='%Y-%m-%d %H:%M:%S')
            log_h.setFormatter(formatter)
            logger.addHandler(log_h)
        return logger


    def create_nested_tree(self, save_dir, nested_dirs):

        for subdir in nested_dirs:
            save_dir = os.path.join(save_dir, subdir)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

        return save_dir

    
    def create_subdir(self, save_dir, subdir):

        subdir_path = os.path.join(save_dir, subdir)
        if not os.path.exists(subdir_path):
            os.mkdir(subdir_path)

        return subdir_path


    def create_function(self, function_config):

        module_path = function_config["path"]
        func_name = function_config["name"]

        return get_object(
            source= module_path, 
            name= func_name
        )


    def take_config(self, key):
        try:
            if key in self.exp_config.keys():

                self.logger.info(f"using {key} config from experiment config")
                return copy.deepcopy(self.exp_config[key])
            else:
                self.logger.info(f"using {key} config from global config")
                return copy.deepcopy(self.global_config[key])
        except:
            return None