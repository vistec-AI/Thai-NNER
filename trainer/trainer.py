import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class TrainerNNE(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, 
            metric_ftns, optimizer, config, 
            device, data_loader, lr_scheduler=None, 
            len_epoch=None):
        super().__init__(
            model, criterion, metric_ftns, 
            optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.train_data_loader = data_loader.get_train()
        self.valid_data_loader = data_loader.get_validation()
        self.do_validation = self.valid_data_loader is not None
        self.layer_train = self.config['trainer']['layers_train']

        if len_epoch is None:
            # epoch-based training
            print("Epoch-based training")
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            print("Iteration-based training")
            self.data_loader = inf_loop(self.train_data_loader)
            self.len_epoch = len_epoch

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch[]

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, instance in enumerate(self.train_data_loader):
            input_ids = torch.tensor(instance['input_ids']).to(self.device)
            attention_mask = torch.tensor(instance['attention_mask']).to(self.device)
            nested_lm_conll_ids = {l:None for l in range(len(self.layer_train))}
            for index, layer in enumerate(self.layer_train):
                temp_nested_lm_conll_ids = instance['nested_lm_conll_ids'][layer]
                nested_lm_conll_ids[index]=torch.tensor(temp_nested_lm_conll_ids).to(self.device)
            self.optimizer.zero_grad()
            output = self.model(input_ids, attention_mask)
            loss = 0
            for index in range(len(self.layer_train)):
                loss+=self.criterion(output[index], nested_lm_conll_ids[index])
            loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(
                    met.__name__, 
                    met(output, nested_lm_conll_ids, 
                        attention_mask, 
                        self.data_loader.boundary_type, 
                        ids2tag=self.data_loader.ids2tag, 
                        info=False))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch, self._progress(batch_idx), loss.item()))
            if batch_idx == self.len_epoch:break
        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, instance in enumerate(self.valid_data_loader):
                input_ids = torch.tensor(instance['input_ids']).to(self.device)
                attention_mask = torch.tensor(instance['attention_mask']).to(self.device)
                nested_lm_conll_ids = {l:None for l in range(len(self.layer_train))}
                for index, layer in enumerate(self.layer_train):
                    temp_nested_lm_conll_ids = instance['nested_lm_conll_ids'][layer]
                    nested_lm_conll_ids[index]=torch.tensor(temp_nested_lm_conll_ids).to(self.device)
                output = self.model(input_ids, attention_mask)
                loss = 0
                for index in range(len(self.layer_train)):
                    loss+=self.criterion( output[index], nested_lm_conll_ids[index])
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, 
                        met(output, nested_lm_conll_ids, 
                            attention_mask, 
                            self.data_loader.boundary_type,
                            info=False,
                            ids2tag=self.data_loader.ids2tag))
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()