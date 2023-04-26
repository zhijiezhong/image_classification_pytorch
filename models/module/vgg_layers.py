import torch.nn as nn


def make_layers(cfg: list, batch_norm=False):
    layers = []
    # c = [num_layer, in_channels, out_channels]
    for c in cfg:
        for i in range(c[0]):
            if i == 0:
                layers.append(nn.Conv2d(c[1], c[2], (3, 3), (1, 1), (1, 1)))
            else:
                layers.append(nn.Conv2d(c[2], c[2], (3, 3), (1, 1), (1, 1)))
            if batch_norm:
                layers.append(nn.BatchNorm2d(c[2]))
            layers.append(nn.ReLU(True))
        layers.append(nn.MaxPool2d(2, 2))

    return nn.Sequential(*layers)


def get_classifier(num_classes: int):
    classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes)
    )
    return classifier
