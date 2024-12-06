import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import *
# import mkl
from torch.utils.data import DataLoader
# from ..utils import data
from utils.data import *
from utils.valid import *
from utils.scheduler import *
from torchvision.transforms import transforms
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark=True
import torch.nn.functional as F
from tqdm import tqdm as tqdm
import sys
from .eval import evaluate_train

import argparse
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
import logging


""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(opts):
    """ Seeding """
    seeding(42)

# Initialization
    import_module = 'from lib.{}.{} import {}'.format(opts.Model.file_name, opts.Model.file_name, opts.Model.name)
    exec(import_module)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = eval(opts.Model.name)().to(device)
    train_dataset = eval(opts.Train.Dataset.type)(opts.Train.Dataset.root,
                                                  transforms=model.transforms,
                                                  augmentation=model.data_augmentation)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=opts.Train.DataLoader.batch_size,
                              shuffle=opts.Train.DataLoader.shuffle,
                              num_workers=opts.Train.DataLoader.num_workers)
    optimizer = eval(opts.Train.Optimizer.type)(model.parameters(), lr=opts.Train.Optimizer.lr,
                                                weight_decay=opts.Train.Optimizer.weight_decay)
    # optimizer = eval(opts.Train.Optimizer.type)(model.parameters(), lr=opts.Train.Optimizer.lr)
    scheduler = get_scheduler(opts, optimizer, train_loader)

    train_epochs = 0
    current_epoch = 0
    train_loss = 0
    best_score = 0
    log_dir = "{}/log_{}".format(opts.Model.save_dir, opts.Train.Logging.num)
    writer = SummaryWriter(log_dir=log_dir)

    if os.path.exists(opts.Model.save_dir) is False:
        os.makedirs(opts.Model.save_dir)

    # logging config
    logging.basicConfig(filename='{}/{}_{}.log'.format(opts.Model.save_dir, opts.Train.Logging.filename, opts.Train.Logging.num),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

# Load Model

    if opts.Train.Logging.num > 1 and os.path.exists('{}/{}.pt'.format(opts.Model.save_dir, opts.Model.name)):
        checkpoint = torch.load('{}/{}.pt'.format(opts.Model.save_dir, opts.Model.name), map_location=device)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint['scheduler'])
        current_epoch = checkpoint["epoch"]
        # scheduler.last_epoch = current_epoch
        best_score = checkpoint["best_score"]
        print('model load success')
        print("current_epoch:{}".format(current_epoch))

# Print Configs
    logging.info('#' * 20 + ' Training Configs ' + '#' * 20)
    logging.info('Model: ' + opts.Model.name)
    logging.info('TrainDataset: ' + opts.Train.Dataset.name)
    logging.info('ValidationDataset: ' + opts.Validation.Dataset.name)
    logging.info(f'Batch_size: {opts.Train.DataLoader.batch_size}')
    logging.info(f'Training_num_workers: {opts.Train.DataLoader.num_workers}')
    logging.info(f'IdealDice: {opts.Validation.IdealDice}')
    logging.info(f'Logging_num: {opts.Train.Logging.num}')
    logging.info('Optimizer: ' + opts.Train.Optimizer.type)
    logging.info('Scheduler: ' + opts.Train.Scheduler.type)
    logging.info('Root of TrainDataset: ' + opts.Train.Dataset.root)
    logging.info('Root of ValidationDataset: ' + opts.Validation.Dataset.root)
    logging.info('Save_dir of Model: ' + opts.Model.save_dir)


# Train
    early_stopping_count = 0
    logging.info('#' * 20 + ' Start Training ' + '#' * 20)
    for epoch in range(opts.Train.Scheduler.epochs):

        train_loss = 0
        # print('\nEpoch: {}'.format(current_epoch+epoch+1))
        model.train(mode=True)
        bar = tqdm(train_loader, desc='Epoch {}'.format(current_epoch+epoch+1), file=sys.stdout)
        with bar as iterator:
            for x, y, shape in iterator:
                images = x.to(device)
                masks = y.to(device)
                sample = {'images': images, 'masks': masks}
                optimizer.zero_grad()
                out = model.forward(sample)
                loss = out['loss']
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                bar.set_postfix({'loss': loss.item()})
        scheduler.step()
        train_epochs += 1

        # valid
        metrics = validation(model, opts)

        # log
        logging.info('epoch: {}, dataset: {}, MAE: {:.4f}, m_dice: {:.4f}, miou: {:.4f}'.format(train_epochs + current_epoch,
                                                                                                opts.Validation.Dataset.testsets,
                                                                                                metrics['MAE'],metrics['m_dice'],metrics['m_iou']))
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("avg_train_loss", avg_train_loss, epoch)
        writer.add_scalars("metrics", {'MAE': metrics['MAE'], 'm_dice': metrics['m_dice'],
                                      'm_iou': metrics['m_iou']}, epoch)


        # save
        if best_score < metrics['m_dice']:
            early_stopping_count = 0
            best_score = metrics['m_dice']
            print('Saving model... Current best epoch:{}'.format(train_epochs + current_epoch))
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'epoch': train_epochs + current_epoch, 'scheduler': scheduler.state_dict(),
                     'best_score': best_score}
            torch.save(state, '{}/{}.pt'.format(opts.Model.save_dir, opts.Model.name))

            if best_score > opts.Validation.IdealDice:
                # print("IdealDice:", opts.Validation.IdealDice)
                evaluate_train(model, opts, log=True, epoch=train_epochs+current_epoch)
                if opts.Train.Logging.save_all_ideal_epoachs:
                    torch.save(state, '{}/{}_{}.pt'.format(opts.Model.save_dir, opts.Model.name, current_epoch + train_epochs))

            print('##############################################################################best', best_score)
            logging.info('##############################################################################best:{}'.format(best_score))
        else:
            early_stopping_count += 1
        # if(train_epochs % 10 == 0):
        #     # state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(),
        #     #          'epoch': train_epochs + current_epoch}
        #     state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(),
        #              'epoch': train_epochs + current_epoch, 'scheduler': scheduler.state_dict(),
        #              'best_score': best_score}
        #     torch.save(state, '{}/{}-{}.pt'.format(opts.Model.save_dir, opts.Model.name, current_epoch + train_epochs))
        #     torch.save(state, '{}/{}.pt'.format(opts.Model.save_dir, opts.Model.name))
        # print("loss:{}".format(loss.item()))
        # print("loss:{}".format(avg_train_loss))

        if early_stopping_count == opts.Train.early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {opts.Train.early_stopping_patience} continously.\n"
            logging.info(data_str)
            print(data_str)
            break
    data_str = '#' * 20 + ' Training finished ' + '#' * 20
    print(data_str)
    logging.info(data_str)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--save_dir', type=str, default='./Results/', help='directory ot save results')
    parser.add_argument('--model_name', type=str, default='UAASANet', help='Name of the model')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--train_dir', type=str, default='../Dataset/Polyp/PraNet/train', help='Directory of train dataset')
    opts = parser.parse_args()
    train(opts)

