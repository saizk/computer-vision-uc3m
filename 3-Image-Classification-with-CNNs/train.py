from __future__ import print_function, division
import gc
import time
import random
import copy
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models.detection as dmodels
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy.random as npr

from sklearn import metrics
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim import lr_scheduler

from dataset import IsicDataset
from transforms import get_train_transform, get_test_transform

random.seed(42)
npr.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.enabled = False

warnings.filterwarnings("ignore")  # Ignore warnings
plt.ion()  # interactive mode


def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()


def computeAUCs(scores, labels):
    aucs = np.zeros((2,))
    # Calculamos el AUC melanoma vs all
    scores_mel = scores[:, 1]
    labels_mel = (labels == 1).astype(np.int)
    aucs[0] = metrics.roc_auc_score(labels_mel, scores_mel)

    # Calculamos el AUC queratosis vs all
    scores_sk = scores[:, 2]
    labels_sk = (labels == 2).astype(np.int)
    aucs[1] = metrics.roc_auc_score(labels_sk, scores_sk)

    return aucs


def setup_datasets(dataset, transform_compose, max_size=2000):
    # Train Dataset
    train_dataset = dataset(csv_file='data/dermoscopyDBtrain.csv',
                            img_dir='data/images',
                            mask_dir='data/masks',
                            max_size=max_size,
                            transform=transform_compose['train'])
    # Val dataset
    val_dataset = dataset(csv_file='data/dermoscopyDBval.csv',
                          img_dir='data/images',
                          mask_dir='data/masks',
                          transform=transform_compose['val'])

    # Test dataset
    test_dataset = dataset(csv_file='data/dermoscopyDBtest.csv',
                           img_dir='data/images',
                           mask_dir='data/masks',
                           transform=transform_compose['test'])

    return train_dataset, val_dataset, test_dataset


# Specify training dataset, with a batch size of 8, shuffle the samples, and parallelize with 4 workers

def setup_dataloaders(dataset, transform, num_workers, batch_sizes, max_train_size):
    train_dataset, val_dataset, test_dataset = setup_datasets(dataset, transform, max_train_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_sizes['train'], shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_sizes['val'], shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_sizes['test'], shuffle=False, num_workers=num_workers)

    datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}  # 'test': len(test_dataset)

    return datasets, dataloaders, dataset_sizes


def train_model(model, criterion, optimizer, scheduler, num_epochs,
                dataset_sizes, dataloaders, device):
    since = time.time()

    num_classes = 3

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc, best_auc = 0, 0
    best_epoch = -1

    # Loop of epochs (each iteration involves train and val datasets)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Cada época tiene entrenamiento y validación
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set the model in training mode
            else:
                model.eval()  # Set the model in val mode (no grads)

            num_samples = dataset_sizes[phase]

            # Create variables to store outputs and labels
            outputs_m = np.zeros((num_samples, num_classes), dtype=np.float)
            labels_m = np.zeros((num_samples,), dtype=np.int)

            running_loss = 0.0
            running_corrects = 0

            cont_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device).float()
                labels = labels.to(device)

                batch_size = labels.shape[0]

                optimizer.zero_grad()  # Set grads to zero

                # Forward
                # Register ops only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward & parameters update only in train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Accumulate the running loss
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                outputs = F.softmax(outputs.data, dim=1)
                # Store outputs and labels
                outputs_m[cont_samples: cont_samples + batch_size, ...] = outputs.cpu().numpy()
                labels_m[cont_samples: cont_samples + batch_size] = labels.cpu().numpy()
                cont_samples += batch_size

            # At the end of an epoch, update the lr scheduler
            if phase == 'train':
                scheduler.step()

            # Accumulated loss by epoch
            epoch_loss = running_loss / num_samples
            epoch_acc = running_corrects / num_samples

            epoch_auc = computeAUCs(outputs_m, labels_m)

            print(f'{phase} -> Loss: {epoch_loss}, Avg. acc: {epoch_acc}\
                  Mel: {epoch_auc[0]} / Seb: {epoch_auc[1]}, Avg. AUC: {epoch_auc.mean()}')

            # Deep copy of the best model
            if phase == 'val' and epoch_acc > best_auc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best model in epoch {:d} avg {:4f}'.format(best_epoch, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, dataset, dataloader, device):
    model.eval()  # Ponemos el modelo en modo evaluación

    num_samples, num_classes = len(dataset), len(dataset.classes)  # Tamaño del dataset

    # Creamos las variables que almacenarán las salidas y las etiquetas
    outputs_m = np.zeros((num_samples, num_classes), dtype=np.float)
    cont_samples = 0

    # Iteramos sobre los datos
    for inputs, _labels in dataloader:
        inputs = inputs.to(device)

        batch_size = inputs.shape[0]  # Tamaño del batch

        with torch.torch.no_grad():  # Paso forward
            outputs = model(inputs)
            outputs = F.softmax(outputs.data, dim=1)  # Aplicamos un softmax a la salida
            outputs_m[cont_samples: cont_samples + batch_size, ...] = outputs.cpu().numpy()
            cont_samples += batch_size

    return outputs_m


class BaseNet(object):

    def __init__(self, model, transform_c, max_train_size, criterion, optimizer, scheduler, num_workers, batch_sizes,
                 device):

        self.datasets, self.dataloaders, self.sizes = self._setup(transform_c, num_workers, batch_sizes, max_train_size)
        self.num_classes = len(self.datasets['train'].classes)

        self.model = model(pretrained=True)
        self.model_name = model.__name__.lower()
        self.trained_model = None
        self.model_path = None

        self.criterion = criterion

        optimizer_func = optimizer['func']
        self.optimizer = optimizer_func(self.model.parameters(), **optimizer['args'])

        scheduler_func = scheduler['func']
        self.scheduler = scheduler_func(self.optimizer, **scheduler['args'])

        self._adjust_network()

        self.device = device
        self.model.to(self.device)

    @staticmethod
    def _setup(transform_c, num_workers, batch_sizes, max_train_size):
        return setup_dataloaders(transform_c['dataset'], transform_c, num_workers, batch_sizes, max_train_size)

    def _adjust_network(self):

        if isinstance(self.model, models.AlexNet) or isinstance(self.model, models.VGG) or \
                isinstance(self.model, models.EfficientNet):
            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_features, self.num_classes)

        elif isinstance(self.model, models.ResNet) or isinstance(self.model, models.ShuffleNetV2) or \
                isinstance(self.model, models.Inception3) or isinstance(self.model, models.GoogLeNet):

            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)

        elif isinstance(self.model, dmodels.MaskRCNN):
            self.model.roi_heads.mask_predictor = dmodels.mask_rcnn.MaskRCNNPredictor(256, 256, self.num_classes)

        elif isinstance(self.model, dmodels.FasterRCNN):
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = dmodels.faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)

    def trainloop(self, num_epochs=25, path='/content/drive/MyDrive'):
        torch.cuda.empty_cache()

        self.trained_model = train_model(
            self.model, self.criterion, self.optimizer, self.scheduler,
            num_epochs=num_epochs,
            dataloaders={'train': self.dataloaders['train'], 'val': self.dataloaders['val']},
            dataset_sizes=self.sizes,
            device=self.device
        )

        self.model_path = f'{path}/{self.model_name}_model_n{num_epochs}.pt'
        self.save_model(self.model_path)

        return self.trained_model

    def evaluate(self, output_path, model_path=None, save_to_csv=False):
        clean_cache()
        if model_path is None:
            model_path = self.model_path

        model = self.load_model(model_path)

        outputs = test_model(model, self.datasets['test'], self.dataloaders['test'], device=self.device)

        if save_to_csv:
            self.save_output_to_csv(outputs, path=f'{output_path}/{self.model_name}_output_test.csv')

        return outputs

    def save_model(self, path):
        model_scripted = torch.jit.script(self.trained_model)
        model_scripted.save(path)

    @staticmethod
    def load_model(model):
        return torch.jit.load(model)

    @staticmethod
    def save_output_to_csv(outputs, path):
        import csv
        with open(path, mode='w', newline='') as out_file:
            csv_writer = csv.writer(out_file, delimiter=',')
            csv_writer.writerow(['id, label'])
            csv_writer.writerows(outputs)


def gen_model_params(model, data_transforms, max_train_size):
    return {
        'model': model,
        'transform': data_transforms,
        'max_train_size': max_train_size,
        'criterion': nn.CrossEntropyLoss(),
        'optimizer':
            {'func': optim.SGD,
             'args': {'lr': 0.001, 'momentum': 0.9}},
        'scheduler':
            {'func': lr_scheduler.StepLR,
             'args': {'step_size': 7, 'gamma': 0.1}},
        'num_workers': 8,
        'batch_sizes': {
            'train': 64,
            'val': 128,
            'test': 128
        }
    }


def gen_model(params, device):
    return BaseNet(
        model=params.get('model'),
        transform_c=params.get('transform'),
        max_train_size=params.get('max_train_size'),
        criterion=params.get('criterion'),
        optimizer=params.get('optimizer'),
        scheduler=params.get('scheduler'),
        num_workers=params.get('num_workers'),
        batch_sizes=params.get('batch_sizes'),
        device=device
    )


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_composed, test_composed = get_train_transform(), get_test_transform()
    data_transforms = {'train': train_composed, 'val': train_composed, 'test': test_composed,
                        'dataset': IsicDataset}

    detection_models = [dmodels.fasterrcnn_resnet50_fpn,
                        dmodels.maskrcnn_resnet50_fpn]

    test_models = [
        # models.efficientnet_b5,
        # models.resnext50_32x4d,
        # models.resnext101_32x8d,

        # models.alexnet,
        # models.resnet18,
        # models.resnet50,
        # models.resnet101,
        # models.resnext50_32x4d,
        # models.resnext101_32x8d,
        # models.googlenet,
        models.vgg13_bn,
        # models.vgg19,
        # models.shufflenet_v2_x0_5
    ]

    model_params = [gen_model_params(model, data_transforms, max_train_size=2000) for model in test_models]

    net_models = [gen_model(params, device) for params in model_params]

    for model in net_models:
        model_trained = model.trainloop(num_epochs=25, path='models')
        outputs = model.evaluate(
            output_path='models',
            save_to_csv=True
        )


if __name__ == '__main__':
    main()
