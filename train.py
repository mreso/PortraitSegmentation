import os
import os.path as osp
from tqdm import tqdm
import argparse

from lovasz import iou_binary
import torch

from data.utils import create_data_loader
from losses import EdgeLoss
from models import SINet
from utils import Meter, SegmentationVisualizer


def parseargs():
    parser = argparse.ArgumentParser(description='SINet training.')
    parser.add_argument('--cuda', dest='use_cuda', action='store_true',
                        default=False,
                        help='Use cuda/gpu for training')

    parser.add_argument('--skip-encoder', dest='skip_encoder', action='store_true',
                        default=False,
                        help='Skip re-training of encoder')

    parser.add_argument('--debug', dest='debug', action='store_true',
                        default=False,
                        help='Enable debug visualizations')

    args = parser.parse_args()

    return args


class Trainer(object):
    def __init__(self, data_loader, model, optimizer, loss_fn,
                 debug=False, cuda=False, checkpoint_dir='checkpoints', best_model_filename='best_model.pt'):
        self._data_loader = data_loader
        self._loss_fn = loss_fn

        self.data_loader = None
        self.loss_fn = None

        self.model = model
        self.optimizer = optimizer

        self.visualizer = SegmentationVisualizer()

        self.train_loss_meter = Meter('Loss/train')
        self.train_iou_meter = Meter('IoU/train')

        self.val_loss_meter = Meter('Loss/val')
        self.val_iou_meter = Meter('IoU/val')

        self.checkpoint_dir = checkpoint_dir
        self.best_model_filename = best_model_filename

        self.debug = debug
        self.cuda = cuda

        self.best_iou = 0

        self._set_epoch(0)

    def _set_epoch(self, epoch):
        if epoch in self._data_loader:
            print('Switching data loaders')
            self.data_loader = self._data_loader[epoch]
        if epoch in self._loss_fn:
            print('Switching loss function')
            self.loss_fn = self._loss_fn[epoch]

    def train_one_epoch(self, epoch):
        self._set_epoch(epoch)

        if self.cuda and torch.cuda.is_initialized():
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()

        self.model.train()
        self.train_loss_meter.reset()
        self.train_iou_meter.reset()

        for i, (src, dst) in enumerate(tqdm(self.data_loader['train'], leave=False)):
            if self.cuda and torch.cuda.is_initialized():
                dst = dst.cuda(non_blocking=True)
                src = src.cuda(non_blocking=True)

            self.optimizer.zero_grad()

            y_head = self.model(src)

            loss = self.loss_fn(y_head, dst)

            loss.backward()
            self.optimizer.step()

            self.train_loss_meter(loss.item())

            self.train_iou_meter(iou_binary((y_head.detach() > 0), dst.detach()))

            if i % 100 == 0:
                step = epoch * len(self.data_loader['train']) + i
                data = {'loss': self.train_loss_meter.value(), 'accuracy': self.train_iou_meter.value()}
                self.visualizer.add_scalars(data, step, prefix='train_')
                if self.debug and i == 0:
                    images = {'images': src, 'gt_masks': dst, 'masks': y_head.detach()>0}
                    self.visualizer.add_images(images, epoch, prefix='train_')
        print(f'\tFinal {self.train_loss_meter.name}:\t{self.train_loss_meter.mean():.4f}\t',
              f'final {self.train_iou_meter.name}:\t{self.train_iou_meter.mean():.4f}')

    def validate(self, epoch):
        self._set_epoch(epoch)
        self.model.eval()
        self.val_loss_meter.reset()
        self.val_iou_meter.reset()
        for i, (src, dst) in enumerate(tqdm(self.data_loader['val'], leave=False)):
            if self.cuda and torch.cuda.is_available():
                dst = dst.cuda(non_blocking=True)
                src = src.cuda(non_blocking=True)

            with torch.no_grad():
                y_head = self.model(src)

            loss = self.loss_fn(y_head, dst)
            self.val_loss_meter(loss.item())

            self.val_iou_meter(iou_binary(y_head.detach() > 0, dst.detach()))
            if self.debug and i == 0 and epoch % 50 == 0:
                images = {'images': src, 'gt_masks': dst, 'masks': y_head.detach()>0}
                self.visualizer.add_images(images, epoch, prefix='val_')

        data = {'loss': self.val_loss_meter.mean(), 'accuracy': self.val_iou_meter.mean()}
        self.visualizer.add_scalars(data, epoch, prefix='val_')
        print(f'\tFinal {self.val_loss_meter.name}:\t\t{self.val_loss_meter.mean():.4f}\t',
              f'final {self.val_iou_meter.name}:\t\t{self.val_iou_meter.mean():.4f}')
        
        self.save_best_model()

    @property
    def best_model_checkpoint_filepath(self):
        return osp.join(self.checkpoint_dir, self.best_model_filename)

    def load_previous_best_model(self):
        device = torch.device('cpu')
        state_dict = torch.load(self.best_model_checkpoint_filepath, map_location=device)
        self.model.load_state_dict(state_dict)

    def save_best_model(self):
        if self.val_iou_meter.mean() > self.best_iou:
            print(f'Updating best model @{self.val_iou_meter.name}:{self.val_iou_meter.mean():.04f}')
            self.best_iou = self.val_iou_meter.mean()
            torch.save(self.model.state_dict(), self.best_model_checkpoint_filepath)


def main():
    args = parseargs()

    torch.manual_seed(42)

    model = SINet(train_encoder_only=True)

    configs = {
        0: {
            'batch_size': 36,
            'edge_size': 5,
            'mask_scale': 8,
            'image_size': (224, 224),
        },
        300: {
            'batch_size': 32,
            'edge_size': 15,
            'image_size': (224, 224),
        },
    }

    data_loader = {epoch: create_data_loader(cfg) for epoch, cfg in configs.items()}

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=2e-4)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150, 250, 450, 550], gamma=0.5)

    loss_fn = {epoch: EdgeLoss(cfg['edge_size']) for epoch, cfg in configs.items()}

    if args.use_cuda and torch.cuda.is_available():
        torch.cuda.init()

    trainer = Trainer(data_loader, model, optimizer, loss_fn, args.debug, args.use_cuda,
                      best_model_filename='best_encoder_only_model.pt')

    if not osp.exists(trainer.checkpoint_dir):
        os.makedirs(trainer.checkpoint_dir)

    initial_epoch = 0
    if args.skip_encoder:
        assert osp.exists(trainer.best_model_checkpoint_filepath), 'Checkpoint file does not exist'
        initial_epoch = 300

    for epoch in range(initial_epoch, 600):
        print(f'Epoch\t{epoch}')

        lr = 0
        for param_group in trainer.optimizer.param_groups:
            lr = param_group['lr']
        print(f'Learning rate: {str(lr)}')

        if epoch == 300:
            print(f'Enabling Information Blocking and loading best model')
            trainer.model.train_encoder_only = False

            trainer.load_previous_best_model()

            trainer.best_iou = 0
            trainer.best_model_filename = 'best_model.pt'

            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = 5e-4

        trainer.train_one_epoch(epoch)

        trainer.validate(epoch)

        lr_scheduler.step()
    print(f'Final best model @{trainer.best_iou:.04f}')


if __name__ == '__main__':
    main()
