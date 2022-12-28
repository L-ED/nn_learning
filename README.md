# base
Quantization library for paradigma

## Library structure
    base
    |
    |_ experiment (High level): starts quantization experiment
    |
    |_ learning (Middle level): classes for model learning
    |
    |_ quantization (Middle level): classes for model quantizaton
    |
    |_ utils (Low level): model, dataloaders, optimizers and etc. creator


### Experiment

    experiments
    |
    |_ configs: folder for experiment configuration files in yaml format
    |
    |_ experiment.py: Experiment class for quantization control

###  Learning

    learning
    |
    |_ classifier: Classification task trainer class
    |
    |_ metrics: Folder for metric functions and metric accumulator class
    |
    |_ trainer.py: Base class for task dependent trainers
    |
    |_ train.py: Task independent train function
    |
    |_ validate.py: Task independent validation function


### Quantization

    quantzation
    |
    |_ quantizer.py: Task independent quantization class
    |
    |_ utils.py: Quantization utils and configs


## Config structure

    task: one of currently supported: clasification
    metric: one of currently supported: [accuracy, iou]
    distribute: whether to distribute model using DDP - True or False
    debug: learn model by 1 epoch for debug cases
    save_dir: /path/to/save_dir
    model: 
        name: one of [resnet, mobilenet, ofa] 
        version: if resnet - one of ['18', '50', etc.], for mobilenet - one of ['1', '2', '3_large', etc.], for ofa - one of ['resnet', 'mobilenet']
        source: pytorch
        resume: /storage_labs/3030/LyginE/projects/paradigma/experiments/framework_tests/newrun/2022_11_11/20_20_27/train
    dataset:
        annotation: /storage_labs/db/face_masks/dataset_mask_face_640_480.csv
        train_transform: 
            - totensor
            - randomrotation: 
                - [-5, 5]
            - randomresizedcrop:
                - 224
                - [0.8, 1.0]
            - randomhorizontalflip
            - randomverticalflip
            - normalize: 
                - mean: [0.485, 0.456, 0.406]
                - std: [0.229, 0.224, 0.225]
        val_transform:
            - totensor
            - resize: 256
            - centercrop: 224
            - normalize: 
                - mean: [0.485, 0.456, 0.406]
                - std: [0.229, 0.224, 0.225]
    loader:
        workers: 8
        shuffle: True
        drop_last: True
        batch_size: 64
    experiments:  
        train:
            timing: True
            train_function:
                path: /storage_labs/3030/LyginE/projects/paradigma/quant_framework_new/learning/train.py
                name: train_
            val_function:
                path: /storage_labs/3030/LyginE/projects/paradigma/quant_framework_new/learning/validate.py
                name: validate_
            optimizer: 
                name: Adam
                parameters: 
                    lr: 0.5
            scheduler: 
                name: reducelronplateau
                parameters: 
                    factor: 0.5
                    patience: 10
                    threshold: 0.001
            loss :
                name: crossentropyloss
            epochs: 20
            val_epoch: 2 


# TODO
- [x] Make one functionality for optimizer, loss, loader, scheduler through parameters field
- [x] Read all config keys through take_config with ability to specify config 
- [x] Check logger creator, maybe replace with default
- [ ] Change model creator behaviour for ability to work with different task
- [ ] Make model creator accept parameters kwargs
- [ ] Change config structure to parametrised model, loss  
- [ ] Add DDP

