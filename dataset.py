import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def dataset_from_name(dataset, batch_size=128):
    if dataset == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False
        )

        grayscale = False

    elif dataset == "minst":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        trainset = torchvision.datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            "./data", train=False, download=True, transform=transforms.ToTensor()
        )

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False
        )

        grayscale = True
    else:
        raise ValueError(f"Not support dataset {dataset}.")

    return trainset, trainloader, testset, testloader, grayscale
