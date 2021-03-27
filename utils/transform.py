import torchvision.transforms as transforms

class Transform:
    def cifar_transform(is_train=True):
        if is_train:
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        else:
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
            
        transform_list = transforms.Compose(transform_list)
        return transform_list