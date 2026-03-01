import os
import math
import datetime
import numpy as np
import os.path as osp
from sklearn.metrics import roc_auc_score, precision_score,f1_score,recall_score
from tqdm import tqdm
import torch


class Trainer(object):
    """build a trainer"""
    def __init__(self, model, optimizer, criterion, device, checkpoint, start_epoch, max_epoch, train_loader, valid_loader, test_loader, lr_policy, loss_ratio, save_file, early_stopping_patience=10):
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
        self.save_file = save_file
        if not osp.exists(self.checkpoint):
            os.mkdir(self.checkpoint)
        self.LR_policy = lr_policy
        self.epoch = 0
        self.r = 0.5
        self.f1 = 0
        self.best_auc = 0
        self.precision = 0
        self.loss_ratio = loss_ratio
        self.state_best = None
        self.valid_dict = {}
        self.test_dict = {}
        self.early_stopping_patience = early_stopping_patience
        self.no_improvement_count = 0
        # self.scheduler = scheduler

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
            contras_losses = []
            trans_losses = []
            orth_losses = []
            # self.LR_policy.step() # for cosine learning strategy
            for i_batch, sample_batch in enumerate(tqdm(self.train_loader)):
                data_graph = sample_batch[0].float().to(self.device)
                data_map = sample_batch[1].float().to(self.device)
                label = sample_batch[2].float().to(self.device)
                self.optimizer.zero_grad()
                # label_p, contras_loss = self.model(data_map, data_graph, data_seq)


                label_p, tmp_loss = self.model(data_graph, data_map)
                if tmp_loss != 0:
                    (contra_loss, trans_loss, orth_loss) = tmp_loss
                    all_loss = contra_loss + trans_loss + orth_loss
                else:
                    all_loss = 0
                    contra_loss = 0
                    trans_loss = 0
                    orth_loss = 0

                loss = self.criterion(label_p, label)

                if np.isnan(loss.item()):
                    raise ValueError('loss is nan while training')
                #
                t_loss = loss + self.loss_ratio * all_loss
                # t_loss = all_loss
                t_loss.backward()
                # loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
                running_loss += loss.item()

                if isinstance(contra_loss, torch.Tensor):
                    contra_loss = contra_loss.cpu().item()
                    trans_loss = trans_loss.cpu().item()
                    orth_loss = orth_loss.cpu().item()
                contras_losses.append(contra_loss)
                trans_losses.append(trans_loss)
                orth_losses.append(orth_loss)

            epoch_loss = running_loss / len(self.train_loader)
            print(f'Epoch {epoch} Average Loss: {epoch_loss}, contras_losses: {np.average(contras_losses)}, trans_losses: {np.average(trans_losses)}, orth_losses: {np.average(orth_losses)}')
            # validation and save the model with higher accuracy
            self.valid()
            if self.no_improvement_count >= self.early_stopping_patience:
                print(f'No improvement in validation loss for {self.early_stopping_patience} epochs. Early stopping!')
                break
        print("Best precision:{:.4f}, r: {:.4f}, f1: {:.4f}, auc:{:.4f}".format(self.valid_dict['precision'], self.valid_dict['r'], self.valid_dict['f1'], self.best_auc))
        print("Test precision:{:.4f}, r: {:.4f}, f1: {:.4f}, auc:{:.4f}".format(self.test_dict['precision'],
                                                                                self.test_dict['r'],
                                                                                self.test_dict['f1'], self.test_dict['auc']))
        return self.test_dict['precision'], self.test_dict['r'], self.test_dict['f1'], self.test_dict['auc']

    def valid(self):
        """validate the performance of the trained model."""
        self.model.eval()
        label_p_all = []
        label_t_all = []
        for i_batch, sample_batch in enumerate(self.valid_loader):
            data_graph = sample_batch[0].float().to(self.device)
            data_map = sample_batch[1].float().to(self.device)
            label = sample_batch[2].float().to(self.device)
            with torch.no_grad():
                # label_p = self.model(data_graph, data_map)
                label_p, _ = self.model(data_graph, data_map)
            label_p_all.extend(label_p.view(-1).data.cpu().numpy())
            label_t_all.extend(label.view(-1).data.cpu().numpy())
        loss = self.criterion(torch.tensor(label_p_all), torch.tensor(label_t_all))
        r = recall_score(label_t_all, [int(x > 0.5) for x in label_p_all])
        p = precision_score(label_t_all, [int(x > 0.5) for x in label_p_all])
        f1 = f1_score(label_t_all, [int(x > 0.5) for x in label_p_all])
        valid_auc = roc_auc_score(label_t_all, label_p_all)

        print("Valid loss:{:.6f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, auc:{:.4f}".format(loss.item(), p, r, f1, valid_auc))
        if valid_auc > self.best_auc:
            self.no_improvement_count = 0
            self.best_auc = valid_auc
            self.valid_dict['f1'] = f1
            self.valid_dict['r'] = r
            self.valid_dict['precision'] = p
            self.state_best = self.model.state_dict()

            print("Saving model in current run. Best auc: {:.4f}\n".format(valid_auc))

            # checkpoint_file = os.path.join(self.checkpoint, 'DLoopCaller_model_best.pth')
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
            label = sample_batch[2].float().to(self.device)
            with torch.no_grad():
                label_p, _ = self.model(data_graph, data_map)

            label_p_all.extend(label_p.view(-1).data.cpu().numpy())
            label_t_all.extend(label.view(-1).data.cpu().numpy())
        loss = self.criterion(torch.tensor(label_p_all), torch.tensor(label_t_all))
        r = recall_score(label_t_all, [int(x > 0.5) for x in label_p_all])
        p = precision_score(label_t_all, [int(x > 0.5) for x in label_p_all])
        f1 = f1_score(label_t_all, [int(x > 0.5) for x in label_p_all])
        valid_auc = roc_auc_score(label_t_all, label_p_all)

        self.test_dict['f1'] = f1
        self.test_dict['r'] = r
        self.test_dict['precision'] = p
        self.test_dict['auc'] = valid_auc

        print("Test loss:{:.6f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, auc:{:.4f}".format(loss.item(), p, r, f1, valid_auc))