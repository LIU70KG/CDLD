import os
import math
from math import isnan
import re
import pickle
import gensim
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
import models
from shutil import copyfile, rmtree

class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
    
    # @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:
            self.model = getattr(models, self.train_config.model)(self.train_config)
        
        # Final list
        for name, param in self.model.named_parameters():

            # Bert freezing customizations 
            if self.train_config.data == "mosei":
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= (8):
                        param.requires_grad = False
            elif self.train_config.data == "ur_funny":
                if "bert" in name:
                    param.requires_grad = False
            
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            # print('\t' + name, param.requires_grad)

        # Initialize weight of Embedding matrix with Glove embeddings
        if not self.train_config.use_bert and (self.train_config.data != 'cmdc' and self.train_config.data != 'CMDC' and self.train_config.data != 'DAIC-WOZ'):
            if self.train_config.pretrained_emb is not None:
                self.model.embed.weight.data = self.train_config.pretrained_emb
            self.model.embed.requires_grad = False
        
        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)


    # @time_desc_decorator('Training Start!')
    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = 2

        # self.criterion = criterion = nn.L1Loss(reduction="mean")
        if self.train_config.data == "ur_funny":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        else: # mosi and mosei are regression datasets
            self.criterion = criterion = nn.MSELoss(reduction="mean")
            self.criterion_MAE = criterion_MAE = nn.L1Loss(reduction="mean")
            self.MAE = nn.L1Loss(reduction='mean')

        ce_criterion = nn.CrossEntropyLoss(weight=self.train_config.weights.cuda(), reduction="mean")

        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()  # Similarity Loss

        best_mae, best_rmse, best_pearsonrn = float('inf'), float('inf'), float('-inf')
        best_precision, best_recall, best_f1, best_accuracy= 0.0, 0.0, 0.0, 0.0
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)

        # # 加载验证
        # if os.path.isfile('checkpoints/model_2024-11-29_18:27:51.std'):
        #     print("Loading weights...")
        #     self.model.load_state_dict(torch.load(
        #         f'checkpoints/model_2024-11-29_18:27:51.std'))
        #
        #     self.optimizer.load_state_dict(torch.load(
        #         f'checkpoints/optim_2024-11-29_18:27:51.std'))
        #
        #     mae, rmse, pearsonrn, _, _, _, _ = self.eval(mode="test")
        #     print("Record the verification results...")
        #     print('_test_MAE:%.4f.   RMSE:%.4f.  Pearson:%.4f.' % (mae, rmse, pearsonrn))

        checkpoints = './checkpoints'
        if os.path.exists(checkpoints):
                rmtree(checkpoints, ignore_errors=False, onerror=None)
        os.makedirs(checkpoints)

        for e in range(0, self.train_config.n_epoch):
            print(f"-----------------------------------epoch{e}---------------------------------------")
            print(f"//Current patience: {curr_patience}, current trial: {num_trials}.//")
            self.model.train()
            train_loss = []
            shifting_loss_ = []
            class_loss_ = []
            score_loss_ = []

            diff_loss_ = []
            similarity_loss_ = []
            recon_loss_ = []
            pred_center_score_loss_ = []
            order_center_loss_ = []
            interval_center_loss_ = []
            center_score = self.train_config.center_score
            center_score_tensor = torch.tensor(center_score).view(5, 1).cuda()
            for batch in self.train_data_loader:
                # self.model.zero_grad()
                self.optimizer.zero_grad()
                t, v, a, y, label_area, label_shifting, l, bert_sent, bert_sent_type, bert_sent_mask = batch
                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                label_area = to_gpu(label_area)
                label_shifting = to_gpu(label_shifting)
                l = to_gpu(l)
                y_tilde, h = self.model(t, v, a, l, label_area)

                loss = criterion(y_tilde, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([param for param in self.model.parameters() if param.requires_grad],
                                               self.train_config.clip)
                self.optimizer.step()
                train_loss.append(loss.item())

            print(f"--------------------------------------------")
            # mae, rmse, pearsonrn, precision, recall, f1, accuracy = self.eval(mode="dev")
            mae, rmse, pearsonrn, precision, recall, f1, accuracy = self.eval(mode="test")

            flag = 0
            if best_mae > mae:
                best_mae = mae
                rmse_bestmae = rmse
                pearsonrn_bestmae = pearsonrn
                precision_bestmae, recall_bestmae, f1_bestmae, accuracy_bestmae = precision, recall, f1, accuracy
                flag = 1
            if best_rmse > rmse:
                best_rmse = rmse
                flag = 1
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                flag = 1
            # if best_pearsonrn < pearsonrn:
            #     best_pearsonrn = pearsonrn
            #     flag = 1
            if best_f1 < f1:
                best_precision, best_recall, best_f1 = precision, recall, f1
                flag = 1
            if flag == 1:
                print("------------------Found new best model on test set!----------------")
                print(f"epoch: {e}")
                print("mae: ", mae)
                print("rmse: ", rmse)
                print("pearsonrn: ", pearsonrn)
                print("precision: ", precision)
                print("recall: ", recall)
                print("f1: ", f1)
                print("accuracy: ", accuracy)
                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'checkpoints/model_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'checkpoints/optim_{self.train_config.name}.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'checkpoints/model_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_{self.train_config.name}.std'))
                    lr_scheduler.step()
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
            
            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break

        # print("------------------best model on test set----------------")
        # # mae, rmse, pearsonrn, precision, recall, f1, accuracy = self.eval(mode="dev", best=True)
        # mae, rmse, pearsonrn, precision, recall, f1, accuracy = self.eval(mode="test", best=True)
        # print('_test_MAE:%.4f.   RMSE:%.4f.  Pearson:%.4f.' % (mae, rmse, pearsonrn))
        # print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./' % (precision, recall, f1, accuracy))

        # print("------------------best all on test set----------------")
        # print('_best_mae:%.4f. / best_rmse:%.4f. / best_f1:%.4f. / best_accuracy: %.4f.' % (best_mae, best_rmse, best_f1, best_accuracy))
        print("------------------best MAE on test set----------------")
        mae, rmse, pearsonrn = best_mae, rmse_bestmae, pearsonrn_bestmae
        precision, recall, f1, accuracy = precision_bestmae, recall_bestmae, f1_bestmae, accuracy_bestmae
        print('_test_MAE:%.4f.   RMSE:%.4f.  Pearson:%.4f.' % (mae, rmse, pearsonrn))
        print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./' % (precision, recall, f1, accuracy))

        # 判断文件是否存在
        if not os.path.exists(self.train_config.best_model_Configuration_Log):
            # 如果文件不存在，则创建文件
            with open(self.train_config.best_model_Configuration_Log, 'w') as f:
                pass  # 创建一个空文件

        with open(self.train_config.best_model_Configuration_Log, 'a', encoding="utf-8") as F1:
            line = 'class_weight:{class_weight} | shifting_weight:{shifting_weight} | order_center_weight:{order_center_weight} | ce_loss_weight:{ce_loss_weight} | pred_center_score_weight:{pred_center_score_weight}\n ' \
                   'test_best_MAE:-----------{test_MAE}------------ | RMSE:{RMSE} | Pearson:{Pearson} |\n' \
                   'precision:{precision} | recall:{recall} | f1:{f1} | accuracy:{accuracy} |\n' \
                   'best_mae:{best_mae} | best_rmse:{best_rmse} | best_f1:{best_f1} | best_accuracy:{best_accuracy} |\n' \
                .format(class_weight=self.train_config.class_weight,
                        shifting_weight=self.train_config.shifting_weight,
                        order_center_weight=self.train_config.order_center_weight,
                        ce_loss_weight=self.train_config.ce_loss_weight,
                        pred_center_score_weight=self.train_config.pred_center_score_weight,
                        test_MAE=mae,
                        RMSE=rmse,
                        Pearson=pearsonrn,
                        precision=precision,
                        recall=recall,
                        f1=f1,
                        accuracy=accuracy,
                        best_mae=best_mae,
                        best_rmse=best_rmse,
                        best_f1=best_f1,
                        best_accuracy=best_accuracy,
                        )

            print('result saved～')
            F1.write(line)

        return mae
    
    def eval(self, mode=None, to_print=False, best=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        data_id, data_segment = [], []
        eval_loss, eval_loss_diff = [], []

        if mode == "train":
            dataloader = self.train_data_loader
        elif mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

        if best:
            self.model.load_state_dict(torch.load(
                f'checkpoints/model_{self.train_config.name}.std'))
            

        with torch.no_grad():

            for batch in dataloader:
                self.model.zero_grad()
                t, v, a, y, label_area, label_shifting, l, bert_sent, bert_sent_type, bert_sent_mask = batch
                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                label_area = to_gpu(label_area)
                label_shifting = to_gpu(label_shifting)
                y_tilde, h = self.model(t, v, a, l, label_area)
                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()


        mae, rmse, pearsonrn, precision, recall, f1, accuracy = self.calc_metrics(y_true, y_pred, mode, to_print)

        return mae, rmse, pearsonrn, precision, recall, f1, accuracy

    def multiclass_acc(self, preds, truths):
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        if self.train_config.data == "ur_funny":
            test_preds = np.argmax(y_pred, 1)
            test_truth = y_true
            if to_print:
                print("Confusion Matrix (pos/neg) :")
                print(confusion_matrix(test_truth, test_preds))
                print("Classification Report (pos/neg) :")
                print(classification_report(test_truth, test_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))
            
            return accuracy_score(test_truth, test_preds)
        
        elif self.train_config.data == "cmdc" or self.train_config.data == 'DAIC-WOZ':
            test_preds = y_pred
            test_truth = y_true
            mae = np.mean(np.absolute(test_preds - test_truth))
            rmse = np.sqrt(np.mean((test_preds - test_truth) ** 2))
            pearsonrn, p_value = pearsonr(test_preds, test_truth)
            # corr = np.corrcoef(preds, y_test)[0][1]  # preds 和 y_test 之间的相关系数
            if self.train_config.data == "cmdc":
                preds_b = test_preds >= 9
                y_test_b = test_truth >= 9
            else:
                preds_b = test_preds >= 10
                y_test_b = test_truth >= 10
            precision = precision_score(y_test_b, preds_b, zero_division=1)
            recall = recall_score(y_test_b, preds_b, zero_division=1)
            f1 = f1_score(y_test_b, preds_b)
            accuracy = accuracy_score(y_test_b, preds_b)
            return mae, rmse, pearsonrn, precision, recall, f1, accuracy


        else:
            test_preds = y_pred
            test_truth = y_true

            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
            test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
            test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

            mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
            corr = np.corrcoef(test_preds, test_truth)[0][1]
            mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
            mult_a5 = self.multiclass_acc(test_preds_a5, test_truth_a5)

            f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')

            # pos - neg
            binary_truth = (test_truth[non_zeros] > 0)
            binary_preds = (test_preds[non_zeros] > 0)

            if to_print:
                print("mae: ", mae)
                print("corr: ", corr)
                print("mult_acc: ", mult_a7)
                print("Classification Report (pos/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))

            # non-neg - neg
            binary_truth = (test_truth >= 0)
            binary_preds = (test_preds >= 0)

            if to_print:
                print("Classification Report (non-neg/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))

            return accuracy_score(binary_truth, binary_preds)



    def ce_ordinal_loss(self, p_class, target, num_classes):
        # 计算交叉熵损失
        targets = target.long().squeeze()
        log_probs = F.log_softmax(p_class, dim=1)
        ce_loss = -log_probs[range(len(targets)), targets]

        # 计算有序损失
        probs = F.softmax(p_class, dim=1)
        target_expand = targets.view(-1, 1).expand(-1, num_classes)
        class_range = torch.arange(0, num_classes).float().cuda()
        class_distance = (class_range - target_expand) ** 2
        order_loss = torch.sum(class_distance * probs, dim=1)
        weights = self.train_config.weights.cuda()

        # 计算总损失
        ce_ordinal_loss = weights[targets] * (self.train_config.ce_loss_weight * ce_loss + order_loss)
        return ce_ordinal_loss.mean()
        # return ce_ordinal_loss.sum()



