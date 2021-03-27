import torch
import torchvision
from tqdm import tqdm

from utils import Transform
from utils import logger

def train(epoch, train_loader, model, optimizer, criterion):
    model.train()
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        outputs = model(inputs.cuda())
        loss = criterion(outputs, targets.cuda())
        loss.backward()
        optimizer.step()
        #logger.train_log(epoch * len(train_loader) + batch_idx, loss, len(inputs))

def test(eval_loader, model):
    model.eval()
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

    acc = 100. * correct / len(eval_loader.dataset)
    return acc

def set_dataset(dataset_name):
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10
        transform = Transform.cifar_transform
        stem = 'cifar'
        num_classes = 10
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100
        transform = Transform.cifar_transform
        stem = 'cifar'
        num_classes = 100
    else:
        assert 0, "Invalid Dataset!"

    return dataset, transform, stem, num_classes

def save_model(model, optimizer, epoch, best_acc, model_path):
    torch.save({
        'epoch': epoch,
        'best_acc': best_acc,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)

def load_model(model, optimizer, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']

    return model, optimizer, start_epoch, best_acc
