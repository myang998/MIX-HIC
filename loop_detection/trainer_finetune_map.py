import os
import math
import datetime
import numpy as np
import os.path as osp
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import r2_score
from tqdm import tqdm
import torch


class Trainer(object):
    """build a trainer"""
    def __init__(self, model, optimizer, criterion, device, checkpoint, start_epoch, max_epoch, train_loader, valid_loader, test_loader, lr_policy, save_file, early_stopping_patience=10):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.checkpoint = checkpoint
        if not osp.exists(self.checkpoint):
            os.mkdir(self.checkpoint)
        self.LR_policy = lr_policy
        self.epoch = 0
        self.r = 0.5
        self.f1 = 0
        self.best_auc = 0
        self.precision = 0
        self.state_best = None
        self.valid_dict = {}
        self.test_dict = {}
        self.save_file = save_file
        self.early_stopping_patience = early_stopping_patience
        self.no_improvement_count = 0

    def train(self):
        """training the model"""
        self.model.to(self.device)
        self.criterion.to(self.device)
        for epoch in range(self.start_epoch, self.max_epoch):
            torch.cuda.empty_cache()
            # set training mode during the training process
            print('epoch', epoch)
            self.model.train()
            self.epoch = epoch
            running_loss = 0.0
            # self.LR_policy.step() # for cosine learning strategy
            for i_batch, sample_batch in enumerate(tqdm(self.train_loader)):
                data_graph = sample_batch[0].float().to(self.device)
                data_map = sample_batch[1].float().to(self.device)
                self.optimizer.zero_grad()
                label_p = self.model(data_graph)
                loss = self.criterion(label_p, data_map)

                if np.isnan(loss.item()):
                    raise ValueError('loss is nan while training')

                t_loss = loss
                t_loss.backward()
                # loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_loader)
            print(f'Epoch {epoch} Average Loss: {epoch_loss}')
            # validation and save the model with higher accuracy
            self.valid()

            if self.no_improvement_count >= self.early_stopping_patience:
                print(f'No improvement in validation loss for {self.early_stopping_patience} epochs. Early stopping!')
                break


        print("Best pcc: {:.4f}, srcc: {:.4f}, r2:{:.4f}".format(self.valid_dict['pcc'], self.valid_dict['srcc'], self.valid_dict['r2']))
        print("Test pcc: {:.4f}, srcc: {:.4f}, r2:{:.4f}".format(self.test_dict['pcc'], self.test_dict['srcc'], self.test_dict['r2']))

    def valid(self):
        """validate the performance of the trained model."""
        self.model.eval()
        label_p_all = []
        label_t_all = []
        for i_batch, sample_batch in enumerate(self.valid_loader):
            data_graph = sample_batch[0].float().to(self.device)
            data_map = sample_batch[1].float().to(self.device)
            with torch.no_grad():
                label_p = self.model(data_graph)

            kkk = label_p.view(-1).data.cpu().numpy()
            ttt = kkk[0]
            label_p_all.extend(label_p.view(-1).data.cpu().numpy())
            label_t_all.extend(data_map.view(-1).data.cpu().numpy())
        # loss = self.criterion(torch.tensor(label_p_all), torch.tensor(label_t_all))

        # label_p_all = np.concatenate(label_p_all, axis=0).squeeze().flatten()
        # label_t_all = np.concatenate(label_t_all, axis=0).squeeze().flatten()

        loss = self.criterion(torch.tensor(label_p_all), torch.tensor(label_t_all))

        epoch_r2 = r2_score(label_p_all, label_t_all)
        epoch_pearsonr, _ = pearsonr(label_p_all, label_t_all)
        epoch_spearmanr, _ = spearmanr(label_p_all, label_t_all)

        print("Valid loss:{:.6f}, pcc: {:.4f}, srcc: {:.4f}, r2: {:.4f}".format(loss.item(), epoch_pearsonr, epoch_spearmanr, epoch_r2))

        if epoch_pearsonr > self.best_auc:
            self.no_improvement_count = 0
            self.best_auc = epoch_pearsonr
            self.valid_dict['pcc'] = epoch_pearsonr
            self.valid_dict['srcc'] = epoch_spearmanr
            self.valid_dict['r2'] = epoch_r2
            self.state_best = self.model.state_dict()

            print("Saving model in current run. Best pcc: {:.4f}\n".format(epoch_pearsonr))

            torch.save(self.state_best, self.save_file)

            self.test()
        else:
            self.no_improvement_count += 1

    def test(self):
        """validate the performance of the trained model."""
        self.model.eval()
        label_p_all = []
        label_t_all = []
        for i_batch, sample_batch in enumerate(self.test_loader):
            data_graph = sample_batch[0].float().to(self.device)
            data_map = sample_batch[1].float().to(self.device)
            with torch.no_grad():
                label_p = self.model(data_graph)
            label_p_all.extend(label_p.view(-1).data.cpu().numpy())
            label_t_all.extend(data_map.view(-1).data.cpu().numpy())

        loss = self.criterion(torch.tensor(label_p_all), torch.tensor(label_t_all))
        epoch_r2 = r2_score(label_p_all, label_t_all)
        epoch_pearsonr, _ = pearsonr(label_p_all, label_t_all)
        epoch_spearmanr, _ = spearmanr(label_p_all, label_t_all)

        self.test_dict['r2'] = epoch_r2
        self.test_dict['pcc'] = epoch_pearsonr
        self.test_dict['srcc'] = epoch_spearmanr

        print("Test loss:{:.6f}, pcc: {:.4f}, srcc: {:.4f}, r2: {:.4f}".format(loss.item(), epoch_pearsonr, epoch_spearmanr, epoch_r2))