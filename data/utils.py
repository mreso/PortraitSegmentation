import torch

from data.datasets import create_datasets


def create_data_loader(config):
    batch_size = config.get('batch_size', 32)

    datasets = create_datasets(config)

    train_data_loader = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True,
                                                    num_workers=4,
                                                    pin_memory=True)

    val_data_loader = torch.utils.data.DataLoader(datasets['val'], batch_size=2 * batch_size, shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=True)

    return {'train': train_data_loader, 'val': val_data_loader}