import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import time
import matplotlib.pyplot as plt

from base import *
from nets import *
from utils import logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--resume', type=bool, default=False)
    args = parser.parse_args()
    return args

def main():
    cfg = parse_args()
    logger.info(cfg)

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    cudnn.benchmark = True
    logger.set_log_interval(cfg.log_interval)
    logger.info('Using GPU: {}'.format(cfg.gpu))
    
    # path
    cfg.data_path = '../data'
    cfg.model_path = '../checkpoints/' + cfg.dataset + '_' + cfg.model + '.pth' 
    
    # dataset
    logger.info('Preparing data ..')
    logger.info("DATASET : " + cfg.dataset)
    dataset, transform, stem, num_classes = set_dataset(cfg.dataset)

    # load dataset
    train_dataset = dataset(root=cfg.data_path, train=True, download=True, transform=transform(is_train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)
    eval_dataset = dataset(root=cfg.data_path, train=False, download=True, transform=transform(is_train=False))
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.test_batch_size, shuffle=False, num_workers=cfg.num_workers)

    # model
    model_zoo = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    logger.info('Building Model..')
    logger.info("MODEL : " + cfg.model)
    if cfg.model in model_zoo:
        model = eval(cfg.model)(stem=stem, num_classes=num_classes).cuda()
    else:
        assert 0, "Invalid Model!"

    params = []
    for name, param in model.named_parameters():
        params.append(param)

    # set trainer
    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    start_epoch = 0
    best_acc = 0.0
    
    # resume
    if cfg.resume:
        model, optimizer, start_epoch, best_acc = load_model(model, optimizer, cfg.model_path) 

    # train & test
    for epoch in range(start_epoch, cfg.max_epochs):
        train(epoch, train_loader, model, optimizer, criterion)
        acc = test(eval_loader, model)
        scheduler.step()

        # save model
        if acc > best_acc:
            best_acc = acc
            save_model(model, optimizer, epoch, best_acc, cfg.model_path)
        logger.test_log(epoch, acc, best_acc)

if __name__ == "__main__":
    main()