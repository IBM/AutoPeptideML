"""
Code Adapted from the UniDL4BioPep
implementation of their model for PyTorch
in the GitHub Repository:
https://github.com/David-Dingle/UniDL4BioPep_ASL_PyTorch/
"""

import copy
import os

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..metrics import evaluate
from .dataset import UniDL4BioPep_Dataset, UniDL4BioPep_Inference
from .loss import ASLSingleLabel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BaseModel(nn.Module):
    def get_params(self):
        return self.params

    def predict_proba(
        self,
        x: np.array
    ):
        self.eval()
        x_dataloader = self._prepare_data(x, y=None)
        outputs = []
        for batch in x_dataloader:
            batch = batch.to(self.device_name)
            output = self(batch).cpu().detach().numpy()
            outputs.append(output)

        return np.concatenate(outputs)

    def predict(
        self,
        x: np.array,
        device: str
    ):
        self.to('cpu')
        self.load_state_dict(self.best_model)
        self.to(device)
        self.device_name = device
        if 'reg' not in self.task:
            return (self.predict_proba(x)[:, 1] > 0.5).astype(int)
        else:
            return self.predict_proba(x)

    def evaluate(self, x, y):
        self.eval()
        x_dataloader = self._prepare_data(x, y=None)
        for batch in x_dataloader:
            output = self(batch).cpu().detach()
        report = self._scores(output, torch.Tensor(y))
        return report

    def fit(
        self,
        train_x: np.array,
        train_y: np.array,
        valid_x: np.array,
        valid_y: np.array,
        device: str
    ):
        if not os.path.exists(self.logger):
            os.mkdir(self.logger)
        logger_training = os.path.join(self.logger, 'train.log')
        logger_validation = os.path.join(self.logger, 'valid.log')
        logger_checkpoint = os.path.join(self.logger, 'best_chckpt.pt')

        train_set = self._prepare_data(train_x, train_y)
        valid_set = self._prepare_data(valid_x, valid_y)

        self = self.to(device)
        self.device_name = device
        min_valid_loss = float("inf")
        for epoch in range(self.epochs):
            running_loss = 0.0
            train_acc = []
            valid_loss = 0.0
            self.train()
            counter = 0
            for i, (inputs, labels) in enumerate(train_set):
                self.optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                if 'class' in self.task:
                    new_labels = torch.zeros((labels.shape[0], 2))
                    new_labels[labels == 0, 0] = 1
                    new_labels[labels == 1, 1] = 1
                    labels = new_labels.to(device)
                outputs = self(inputs)
                if 'multi' in self.task:
                    loss = self.loss(outputs.unravel(), labels.unravel())
                else:
                    loss = self.loss(outputs.float(), labels.float())

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if 'reg' in self.task:
                    train_acc.append(self._scores(outputs.to("cpu"), labels.to("cpu"))["mse"])
                else:
                    train_acc.append(self._scores(outputs.to("cpu"), labels.to("cpu"))["f1_weighted"])

            self.eval()
            acc = 0
            for j, (valid_inputs, valid_labels) in enumerate(valid_set):
                valid_labels = valid_labels.to(device)
                valid_inputs = valid_inputs.to(device)
                with torch.no_grad():
                    valid_outputs = self(valid_inputs)
                if 'multi' in self.task:
                    valid_loss = self.loss(outputs.unravel(), labels.unravel())
                else:
                    valid_loss = self.loss(outputs.float(), labels.float())

                if 'reg' in self.task:
                    acc = self._scores(valid_outputs.to('cpu'), valid_labels.to('cpu'))["mse"]
                else:
                    acc = self._scores(valid_outputs.to('cpu'), valid_labels.to('cpu'))["f1_weighted"]

                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, logger_checkpoint)
                    self.cpu()
                    self.best_model = copy.deepcopy(self.state_dict())
                    self.to(device)

    def _get_confusion_matrix(self, y_pred: torch.Tensor, y_test: torch.Tensor):
        predictions = torch.argmax(y_pred, dim=-1).numpy()
        labels = torch.argmax(y_test, dim=-1).numpy()  # A:0, B:1, C:2, [D:3]
        confusion_matrix = sklearn.metrics.confusion_matrix(labels, predictions)
        return confusion_matrix

    def _scores(self, y_pred: torch.Tensor, y_test: torch.Tensor):
        predictions = torch.argmax(y_pred, dim=-1).numpy()
        labels = y_test.numpy()
        task = 'reg' if 'reg' in self.task else 'class'
        return evaluate(predictions, labels, task)

    def _prepare_data(self, x, y, shuffle: bool=False):
        if y is None:
            dataset = UniDL4BioPep_Inference(x)
        else:
            dataset = UniDL4BioPep_Dataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=x.shape[0] if y is None else 64, shuffle=shuffle)
        return dataloader

    def _get_optimizer(self, optim_algorithm: str='adam', lr: float=0.0001, weight_decay: float=0):
        OPTIMIZERS = {
            'adam': torch.optim.Adam
        }
        return OPTIMIZERS[optim_algorithm](self.parameters(), lr=lr, weight_decay=weight_decay)

    def _get_criteria(self, **kwargs):
        return ASLSingleLabel(**kwargs)


class Cnn(BaseModel):
    """
    CNN model
    """
    def __init__(
        self,
        optimizer: dict,
        logger: str,
        labels: int,
        task: str,
        epochs: int = 200,
    ):
        super().__init__()
        self.output_dim = labels
        self.input_dim = 320
        self.dropout = 0.3
        self.stride = 2
        self.kernel_1 = 3
        self.channel_1 = 32

        self.conv_1 = nn.Conv1d(kernel_size=self.kernel_1,
                                out_channels=self.channel_1,
                                in_channels=1, stride=1)
        self.normalizer_1 = nn.BatchNorm1d(self.channel_1)
        self.pooling_1 = nn.MaxPool1d(kernel_size=self.kernel_1,
                                      stride=self.stride)

        self.dropout = nn.Dropout(p=self.dropout)
        self.fc1 = nn.LazyLinear(128)
        self.normalizer_2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, self.output_dim)
        self.device_name = 'cpu'
        self.epochs = epochs
        self.optimizer = self._get_optimizer(**optimizer)
        # self.criteria = self._get_criteria(**criteria)
        self.logger = logger
        if 'multi' in task:
            self.loss = nn.BCELoss()
        elif 'class' in task:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()
        self.task = task
        self.params = {
            'epochs': self.epochs,
            'optimizer': optimizer,
        }

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # (batch, embedding_dim) -> (batch, 1, embedding_dim)
        x = self.conv_1(x)
        if x.shape[0] > 1:
            x = self.normalizer_1(x)
        c_1 = self.pooling_1(F.relu(x))

        c_2 = torch.flatten(c_1, start_dim=1)
        c_2 = self.dropout(c_2)
        c_2 = self.fc1(c_2)
        if x.shape[0] > 1:
            c_2 = self.normalizer_2(c_2)
        out = F.relu(c_2)
        out = self.fc2(out)
        if 'class' in self.task or 'multi' in self.task:
            out = torch.softmax(out, dim=-1)
        return out


class MLP(BaseModel, nn.Module):
    def __init__(
        self,
        optimizer: dict,
        logger: str,
        labels: int,
        task: str,
        epochs: int = 200,
    ):
        super().__init__()
        self.output_dim = labels
        self.input_dim = 320
        self.dropout = 0.3
        self.stride = 2
        self.kernel_1 = 3
        self.channel_1 = 32

        self.mlp = nn.Sequential(
            nn.LazyLinear(self.input_dim),
            nn.LeakyReLU(),
            nn.Linear(self.input_dim, self.input_dim),
            nn.LeakyReLU(),
            nn.Linear(self.input_dim, self.output_dim)
        )
        self.device_name = 'cpu'
        self.epochs = epochs
        self.optimizer = self._get_optimizer(**optimizer)
        # self.criteria = self._get_criteria(**criteria)
        self.logger = logger
        if 'multi' in task:
            self.loss = nn.BCELoss()
        elif 'class' in task:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()
        self.task = task
        self.params = {
            'epochs': self.epochs,
            'optimizer': optimizer,
        }

    def forward(self, x):
        # x = torch.unsqueeze(x, dim=1)  # (batch, embedding_dim) -> (batch, 1, embedding_dim)
        out = self.mlp(x)
        if 'class' in self.task or 'multi' in self.task:
            out = torch.softmax(out, dim=-1)
        # else:
        #     out = out.squeeze(1)
        return out
