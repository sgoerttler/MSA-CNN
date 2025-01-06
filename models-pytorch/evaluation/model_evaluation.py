import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score


from model_setup.optimizer import get_optimizer
from utils.utils import array2str
from utils.utils_torch import get_device


class Evaluator(object):
    """Evaluate model and save results."""
    def __init__(self, config, optimizer, criterion, train_loader, test_loader, epoch_writer, idx_k, device_data, verbose):
        self.config = config
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epoch_writer = epoch_writer
        self.idx_k = idx_k
        self.device = get_device()
        self.device_data = device_data
        self.verbose = verbose
        self.scheduler = None

        if self.epoch_writer.save_results:
            self.epoch_writer.data['fold'] = self.idx_k
            self.epoch_writer.data['train_samples'] = len(self.train_loader.dataset)
            self.epoch_writer.data['test_samples'] = len(self.test_loader.dataset)

    def train_model_one_epoch(self, epoch, model):
        model.train()

        accs = []
        nums_total = []
        losses = []
        conf_matrix = np.zeros((self.config['classes'], self.config['classes']), dtype=int)

        for idx_train_data_batch, train_data_batch in enumerate(
                tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config["epochs"]} ', delay=0.1, disable=self.verbose != 1)):
            self.optimizer.zero_grad()
            inputs, labels = train_data_batch
            if self.device_data == 'cpu':
                inputs = tuple(input.to(self.device) for input in inputs)
                labels = labels.to(self.device)
            output = model.forward(*inputs)

            loss = self.criterion(output, labels)

            y_true = labels.argmax(dim=1).detach().cpu().numpy()
            y_pred = output.argmax(dim=1).detach().cpu().numpy()
            accs.append((y_true == y_pred).astype(float).sum() / len(y_true))
            nums_total.append(len(y_true))
            conf_matrix += confusion_matrix(y_true, y_pred, labels=np.arange(self.config['classes'], dtype=int))

            losses.append(loss.detach().cpu().numpy())
            loss.backward()

            if 'gradient_clipping' in self.config.keys():
                if self.config['gradient_clipping'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['gradient_clipping'])

            self.optimizer.step()

        return model, np.average(accs, weights=nums_total), np.average(losses, weights=nums_total), sum(nums_total), conf_matrix

    def test_model(self, model):
        if len(self.test_loader) == 0:
            return 0, 0, 0, 0, np.zeros((self.config['classes'], self.config['classes']), dtype=int)

        model.eval()
        for idx_test_data_batch, test_data_batch in enumerate(self.test_loader):
            inputs, labels = test_data_batch
            with torch.no_grad():
                if self.device_data == 'cpu':
                    inputs = tuple(input.to(self.device) for input in inputs)
                    labels = labels.to(self.device)
                output = model(*inputs)
                loss = self.criterion(output, labels)

                y_true_i = labels.argmax(dim=1).detach().cpu().numpy()
                y_pred_i = output.argmax(dim=1).detach().cpu().numpy()

                if 'context_length' in self.config.keys():
                    if self.config['context_length'] > 1:
                        y_true_i = y_true_i[:, -1]
                        y_pred_i = y_pred_i[:, -1]
                if idx_test_data_batch == 0:
                    y_true = y_true_i
                    y_pred = y_pred_i
                else:
                    y_true = np.hstack((y_true, y_true_i))
                    y_pred = np.hstack((y_pred, y_pred_i))

        acc = (y_true == y_pred).astype(float).sum() / len(y_true)
        conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(self.config['classes'], dtype=int))

        return model, acc.item(), loss.item(), len(y_true), conf_matrix

    def initialize_scheduler(self):
        if self.config['scheduler'] == 'reduce_lr_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10,
                                                             verbose=self.verbose == 1)
        elif self.config['scheduler'] == 'step_lr':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def run_scheduler(self, epoch, train_loss):
        if self.config['scheduler'] == 'reduce_lr_on_plateau':
            self.scheduler.step(train_loss)
        elif self.config['scheduler'] == 'step_lr':
            self.scheduler.step()
        elif self.config['scheduler'] == 'step_10_epochs':
            if epoch + 1 == 10:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.0001
        elif self.config['scheduler'] == 'step_20_epochs':
            if epoch + 1 == 20:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.0001
        elif self.config['scheduler'] == 'step_50_epochs':
            if epoch + 1 == 50:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.0001
        elif self.config['scheduler'] == 'warmup_10':
            if epoch + 1 <= 10:
                new_learning_rate = 0.0001 * 1.25892541179 ** float(epoch + 1)
                for param_group in self.optimizer.param_groups:
                    if param_group['lr'] < 0.001:
                        param_group['lr'] = new_learning_rate
        elif self.config['scheduler'] == 'exponential_decay':
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] > self.config['learning_rate']:
                    param_group['lr'] = self.config['learning_rate_high_level'] * np.power(0.1, 1/100) ** 100

    def evaluate_model(self, model):
        history = {
            'train_accuracy': [],
            'train_loss': [],
            'test_accuracy': [],
            'test_loss': []
        }

        if 'scheduler' in self.config.keys():
            self.initialize_scheduler()

        for epoch in range(self.config['epochs']):
            if self.config['model'] == 'DeepSleepNet':
                model.update_epoch(epoch)
            model, train_accuracy, train_loss, train_total, train_conf_matrix = self.train_model_one_epoch(epoch, model)
            model, test_accuracy, test_loss, test_total, test_conf_matrix = self.test_model(model)

            if self.verbose != 0:
                print(f"Epoch {epoch + 1}/{self.config['epochs']}\t train acc.: \t{round(train_accuracy * 100, 1)}%, "
                      f"test acc.:\t{round(test_accuracy * 100, 1)}%")
                print(f"Epoch {epoch + 1}/{self.config['epochs']}\t train loss: \t{round(train_loss, 3)}, test loss:\t{round(test_loss, 3)}")
            if self.epoch_writer.save_results:
                self.epoch_writer.data['epoch'] = epoch + 1
                self.epoch_writer.data['train_accuracy'] = train_accuracy
                self.epoch_writer.data['train_loss'] = train_loss
                self.epoch_writer.data['train_conf_matrix'] = array2str(train_conf_matrix)
                self.epoch_writer.data['test_accuracy'] = test_accuracy
                self.epoch_writer.data['test_loss'] = test_loss
                self.epoch_writer.data['test_conf_matrix'] = array2str(test_conf_matrix)
                self.epoch_writer.update_data(self.idx_k, epoch)

            history['train_accuracy'].append(train_accuracy)
            history['train_loss'].append(train_loss)
            history['test_accuracy'].append(test_accuracy)
            history['test_loss'].append(test_loss)

            if 'epochs_stage_1' in self.config.keys():
                if epoch + 1 == self.config['epochs_stage_1']:
                    self.train_loader = DataLoader(
                        self.train_loader.dataset,
                        batch_size=self.train_loader.batch_size,
                        drop_last=self.train_loader.drop_last,
                        shuffle=True,
                        sampler=None
                    )
                    self.optimizer = get_optimizer(self.config, model, stage=2)

            if 'scheduler' in self.config.keys():
                self.run_scheduler(epoch, train_loss)

        return model, history


def save_model(model, config, evaluation, overview_writer):
    """Save model to disk."""
    full_state = {
        'state_dict': model.state_dict(),
        'optimizer': evaluation.optimizer.state_dict(),
        'config': config
    }
    Path(os.path.join(overview_writer.results_folder, 'models')).mkdir(parents=True, exist_ok=True)
    torch.save(full_state, os.path.join(overview_writer.results_folder, 'models',
                                        f'model__{overview_writer.time_string}.pth'))


def conf_matrix_to_ytrue_ypred(conf_matrix):
    # Number of classes
    num_classes = conf_matrix.shape[0]

    # Initialize empty lists for y_true and y_pred
    y_true = []
    y_pred = []

    # Populate y_true and y_pred based on the confusion matrix
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            count = conf_matrix[true_class, pred_class]
            y_true.extend([true_class] * count)
            y_pred.extend([pred_class] * count)

    # Convert lists to numpy arrays and return
    return np.array(y_true), np.array(y_pred)


def get_metrics(conf_matrices, weights):
    accs = np.zeros(len(weights))
    precisions = np.zeros(len(weights))
    recalls = np.zeros(len(weights))
    f1s = np.zeros(len(weights))
    kappas = np.zeros(len(weights))

    for idx_fold, conf_matrix in enumerate(conf_matrices):
        y_true, y_pred = conf_matrix_to_ytrue_ypred(conf_matrix)
        accs[idx_fold] = accuracy_score(y_true, y_pred)
        precisions[idx_fold] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recalls[idx_fold] = recall_score(y_true, y_pred, average='macro')
        f1s[idx_fold] = f1_score(y_true, y_pred, average='macro')
        kappas[idx_fold] = cohen_kappa_score(y_true, y_pred)

    metrics = {
        'acc': np.average(accs, weights=weights),
        'precision': np.average(precisions, weights=weights),
        'recall': np.average(recalls, weights=weights),
        'f1': np.average(f1s, weights=weights),
        'kappa': np.average(kappas, weights=weights)
    }

    return metrics
