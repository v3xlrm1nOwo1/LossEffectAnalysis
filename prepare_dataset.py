import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import add_noise




def get_binary_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5))
    ])

    train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    # Filter for binary classes: T-shirt/top (0), Trouser (1)
    binary_classes = [0, 1]
    filtered_train = [(x, y) for x, y in train_dataset if y in binary_classes]
    filtered_test = [(x, y) for x, y in test_dataset if y in binary_classes]
    all_filtered_data = filtered_train + filtered_test

    random.shuffle(all_filtered_data)

    # Split dataset to Train (80%), Validation (1%), Test (1%)
    train_size = int(0.7 * len(all_filtered_data))
    validation_size = int(0.15 * len(all_filtered_data))
    test_size = len(all_filtered_data) - train_size - validation_size

    train_dataset = all_filtered_data[: train_size]
    validation_dataset = all_filtered_data[train_size: train_size + validation_size]
    test_dataset = all_filtered_data[train_size + validation_size: ]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    noisy_test_data = {
        "gaussian": [(add_noise(x, noise_type="gaussian", noise_factor=0.5), y) for x, y in test_dataset],
        "uniform": [(add_noise(x, noise_type="uniform", noise_factor=0.5), y) for x, y in test_dataset],
        "salt_pepper": [(add_noise(x, noise_type="salt_pepper", salt_prob=0.2, pepper_prob=0.3), y) for x, y in test_dataset],
        "outliers": [(add_noise(x, noise_type="outliers", num_outliers=5, magnitude=10), y) for x, y in test_dataset],
    }

    noisy_test_loaders = {
        noisy_type: DataLoader(data, batch_size=batch_size, shuffle=False)
        for noisy_type, data in noisy_test_data.items()
    }

    return train_loader, validation_loader, test_loader, noisy_test_loaders
