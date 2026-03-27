import fire
import json
import os
from train import run_train
from eval import run_test

import torch
import torchvision
from torchvision.transforms import ToTensor

import wandb

with open('wandb_config.json', 'r') as f:
    wandb_config = json.load(f)

os.environ['WANDB_ENTITY'] = wandb_config['entity']
os.environ['WANDB_PROJECT'] = wandb_config['project']

wandb.login(key=wandb_config['wandb_api_key'])
WANDB_ENTITY = os.environ.get('WANDB_ENTITY')
WANDB_PROJECT = os.environ.get('WANDB_PROJECT')


def load_configs(experiment_name: str, debug=False):
    """
    Load train and test configs from configs.json for a given experiment name.
    
    Args:
        experiment_name: Name of the experiment (e.g., 'experiment_0')
    
    Returns:
        tuple: (train_config, test_config)
    """
    # Get the path to configs.json relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_fname = 'configs_debug.json' if debug else 'configs.json'
    configs_path = os.path.join(project_root, 'experiments', config_fname)
    
    with open(configs_path, 'r') as f:
        configs = json.load(f)
    
    if experiment_name not in configs:
        available_experiments = ', '.join(configs.keys())
        raise ValueError(
            f"Experiment '{experiment_name}' not found in configs.json. "
            f"Available experiments: {available_experiments}"
        )
    
    experiment_config = configs[experiment_name]
    train_config = experiment_config.get('train')
    test_config = experiment_config.get('test')
    
    if train_config is None:
        raise ValueError(f"Train config not found for experiment '{experiment_name}'")
    if test_config is None:
        raise ValueError(f"Test config not found for experiment '{experiment_name}'")
    
    return train_config, test_config


def main(experiment_name: str, debug=False):
    """
    Main function to load and display train and test configs for an experiment.
    
    Args:
        experiment_name: Name of the experiment (e.g., 'experiment_0')
    """

    PROJ_CACHE = '../data'

    # Download and load the training data
    train_dataset = torchvision.datasets.CIFAR10(root=PROJ_CACHE,
                                                train=True,
                                                transform=ToTensor(),
                                                download=True)

    # Download and load the test data
    test_dataset = torchvision.datasets.CIFAR10(root=PROJ_CACHE,
                                            train=False,
                                            transform=ToTensor(),
                                            download=True)
                                            
    train_config, test_configs_list = load_configs(experiment_name, debug)
    print('debug: ', debug)
    print(f'Running : {experiment_name}')
    print('Here is the train config:')
    print(train_config)

    # Unpack train config dictionary into variables
    mix_size = train_config['mix_size']
    classes_list = train_config['classes_list']
    classes_for_test = train_config['classes_for_test']
    n_samples_per_mix = train_config['n_samples_per_mix']
    batch_size = train_config['batch_size']
    total_epochs = train_config['total_epochs']
    lr = train_config['lr']
    weight_decay = train_config['weight_decay']
    warmup_epochs = train_config['warmup_epochs']
    random_seed = train_config['random_seed']

    model = run_train(
        mix_size=mix_size,
        classes_list=classes_list,
        classes_for_test=classes_for_test,
        n_samples_per_mix=n_samples_per_mix,
        batch_size=batch_size,
        train_dataset=train_dataset,  
        test_dataset=test_dataset,
        total_epochs=total_epochs,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        random_seed=random_seed
    )

    print('Now running tests: ')
    for i, test_config in enumerate(test_configs_list):
        print(f'Test Config #({i}) ', experiment_name)
        print(test_config)

        # Unpack test config dictionary into variables
        test_mix_size = test_config['mix_size']
        test_classes_list = test_config['classes_list']
        test_classes_for_test = test_config['classes_for_test']
        test_n_samples_per_mix = test_config['n_samples_per_mix']
        test_batch_size = test_config['batch_size']
        test_random_seed = test_config['random_seed']
        
        run_test(
            model=model,
            train_exp_name=f'train_{mix_size}-{n_samples_per_mix}-{len(train_dataset)}',
            mix_size=test_mix_size,
            classes_list=test_classes_list,
            classes_for_test=test_classes_for_test,
            n_samples_per_mix=test_n_samples_per_mix,
            batch_size=test_batch_size,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            random_seed=test_random_seed
        )

        print(f'Finished test ({i})')
    print(f'Finished train and test for: {experiment_name}')


if __name__ == "__main__":
    fire.Fire(main)