import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import time
import matplotlib.pyplot as plt
import torch.nn as nn 

from base import *
from nets import *
from quantize import quantizer, tuning, Q_Conv2d, Q_Linear
from utils import logger

import matplotlib.pyplot as plt

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
    parser.add_argument('--wbit', type=int, default=2)
    parser.add_argument('--abit', type=int, default=32)
    parser.add_argument('--ratio', type=int, default=0.0)
    parser.add_argument('--cor_factor', type=int, default=0.9)
    parser.add_argument('--cor_interval', type=int, default=1)
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
    cfg.quantized_model_path = '../checkpoints/' + cfg.dataset + '_' + cfg.model + '_quantized.pth' 
    
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
    logger.info('Building Model...')
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

    # load model
    model, _, _, best_acc = load_model(model, optimizer, cfg.model_path)
    best_acc = test(eval_loader, model)
    logger.info('Before Quantization Accuracy : ' + str(best_acc))
    start_epoch = 0
    best_acc = 0.0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs, last_epoch=-1)

    # set quantizer, tuner
    quantizer.set_quantizer(cfg.wbit, cfg.abit, cfg.ratio)
    tuning.set_tuning(cfg.cor_factor, cfg.cor_interval)
    
    # quantize & tuning
    for epoch in range(start_epoch, cfg.max_epochs):
        tuning.apply_quantization(model, epoch)

        tuning.train_mode()
        train(epoch, train_loader, model, optimizer, criterion)
        
        acc = test(eval_loader, model)
        tuning.test_mode()
        quan_acc = test(eval_loader, model)
        scheduler.step()

        if quan_acc > best_acc:
            best_acc = quan_acc
            save_model(model, optimizer, epoch, best_acc, cfg.quantized_model_path)
        
        logger.quantize_log(epoch, acc, quan_acc, best_acc, tuning.weight_ratio)
        
        # visualize
        weight1 = model.layer1[0].conv1.coord_weight.view(-1).cpu().detach().numpy()
        weight2 = model.layer1[0].conv1.weight.view(-1).cpu().detach().numpy()
        scale, weight2 = quantizer.weight_quantize_fn(torch.tensor(weight2))
        
        plt.hist(tuning.weight_ratio * weight1 + scale.numpy() * weight2.numpy(), bins=150)
        plt.yscale('log')
        plt.savefig('./assets/weight.png')
        plt.clf()
        

if __name__ == "__main__":
    main()