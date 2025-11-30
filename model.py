import torchvision.models as models
import torch.nn as nn


def build_model(pretrained=False, num_classes=3):
    """
     This function defines the CNN model.
     Inputs:
      pretrained: True to use transfer learning, False otherwise.
      num_classes: Number of classes in the dataset
    """

    if pretrained:
        print('Loading pre-trained model weights...')

    model = models.efficientnet_b2(pretrained=pretrained)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, out_features=num_classes)

    return model
