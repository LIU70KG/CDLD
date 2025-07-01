import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
from create_dataset import calculate_labels
import matplotlib.pyplot as plt
from IPython.display import clear_output
from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
import models
from shutil import copyfile, rmtree

class Solver(object):
    def __init__(self, train_config, test_config, train_data_loader, test_data_loader, is_train=True, model=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
    
    # @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:
            self.model = getattr(models, self.train_config.model)(self.train_config)
        
        # Final list
        for name, param in self.model.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            # print('\t' + name, param.requires_grad)
        
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
        self.criterion = criterion = nn.MSELoss(reduction="mean")
        self.criterion_MAE  = criterion_MAE = nn.L1Loss(reduction="mean")
        self.MAE = nn.L1Loss(reduction='mean')

        ce_criterion = nn.CrossEntropyLoss(weight=self.train_config.weights.cuda(), reduction="mean")

        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()  # Similarity Loss
        
        # best_valid_loss = float('inf')
        # best_test_loss = float('inf')
        best_mae, best_rmse, best_pearsonrn = float('inf'), float('inf'), float('-inf')
        best_precision, best_recall, best_f1, best_accuracy, best_multiclass_acc= 0.0, 0.0, 0.0, 0.0, 0.0
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)

        # self.tsne(mode="test", best=True)
        # self.picture(mode="test", best=True)

        continue_epochs = 0
        # if os.path.isfile('checkpoints/model_2024-07-18_20:15:36.std'):
        #     print("Loading pretrained weights...")
        #     self.model.load_state_dict(torch.load(
        #         f'checkpoints/model_2024-07-18_20:15:36.std'))
        #
        #     self.optimizer.load_state_dict(torch.load(
        #         f'checkpoints/optim_2024-07-18_20:15:36.std'))
        #     continue_epochs = 9
        # print("continue iter:", continue_epochs)

        checkpoints = './checkpoints'
        if os.path.exists(checkpoints):
                rmtree(checkpoints, ignore_errors=False, onerror=None)
        os.makedirs(checkpoints)

        # 初始化用于存储训练和测试 MAE 的列表
        train_mae_history = []
        test_mae_history = []
        # 设置绘图窗口
        # plt.ion()  # 开启交互模式
        # fig, ax = plt.subplots(figsize=(10, 6))

        for e in range(continue_epochs, self.train_config.n_epoch):
            print(f"-----------------------------------epoch{e}---------------------------------------")
            print(f"//Current patience: {curr_patience}, current trial: {num_trials}.//")
            self.model.train()
            train_loss = []
            for batch in self.train_data_loader:
                # self.model.zero_grad()
                self.optimizer.zero_grad()
                v, a, y, label_area, label_shifting, l = batch
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                label_area = to_gpu(label_area)
                l = to_gpu(l)
                y_tilde, h = self.model(v, a, l, label_area)
                loss = criterion(y_tilde, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([param for param in self.model.parameters() if param.requires_grad],
                                               self.train_config.clip)
                self.optimizer.step()
                train_loss.append(loss.item())
                    # print("iter:%d./ loss:%.4f." %(order_center_loss_.__len__(), loss))


            print(f"Training loss: {round(np.mean(train_loss), 4)}")


            print(f"--------------------------------------------")
            # mae, rmse, pearsonrn, precision, recall, f1, accuracy = self.eval(mode="dev")
            mae_train, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc = self.eval(mode="train")
            print('_train_MAE:%.4f.   RMSE:%.4f.  Pearson:%.4f.' % (mae_train, rmse, pearsonrn))
            print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./ multiclass_acc:%.4f./' % (precision, recall, f1, accuracy, multiclass_acc))

            # mae, rmse, pearsonrn, precision, recall, f1, accuracy = self.eval(mode="dev")
            mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc = self.eval(mode="test")
            print('_test_MAE:%.4f.   RMSE:%.4f.  Pearson:%.4f.' % (mae, rmse, pearsonrn))
            print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./ multiclass_acc:%.4f./' % (precision, recall, f1, accuracy, multiclass_acc))
            # 将当前 epoch 的 MAE 添加到列表中
            train_mae_history.append(mae_train)
            test_mae_history.append(mae)
            flag = 0
            if best_mae > mae:
                best_mae = mae
                rmse_bestmae = rmse
                pearsonrn_bestmae = pearsonrn
                precision_bestmae, recall_bestmae, f1_bestmae, accuracy_bestmae, multiclass_acc_bestmae = precision, recall, f1, accuracy, multiclass_acc
                flag = 1
            if best_rmse > rmse:
                best_rmse = rmse
                flag = 1
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                flag = 1
            if best_multiclass_acc < multiclass_acc:
                best_multiclass_acc = multiclass_acc
                flag = 1
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
                print("multiclass_acc: ", multiclass_acc)
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


        with open(self.train_config.test_mae_history_path, 'w') as f:
            for item in test_mae_history:
                f.write(f"{item}\n")
        print("------------------best all on test set----------------")
        print('_best_mae:%.4f. / best_rmse:%.4f. / best_f1:%.4f. / best_accuracy: %.4f. / best_multiclass_acc: %.4f.' % (best_mae, best_rmse, best_f1, best_accuracy, best_multiclass_acc))
        print("------------------best MAE on test set----------------")
        mae, rmse, pearsonrn = best_mae, rmse_bestmae, pearsonrn_bestmae
        precision, recall, f1, accuracy, multiclass_acc= precision_bestmae, recall_bestmae, f1_bestmae, accuracy_bestmae, multiclass_acc_bestmae
        print('_test_MAE:%.4f.   RMSE:%.4f.  Pearson:%.4f.' % (mae, rmse, pearsonrn))
        print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./ multiclass_acc:%.4f./' % (precision, recall, f1, accuracy, multiclass_acc))

        # 判断文件是否存在
        if not os.path.exists(self.train_config.best_model_Configuration_Log):
            # 如果文件不存在，则创建文件
            with open(self.train_config.best_model_Configuration_Log, 'w') as f:
                pass  # 创建一个空文件

        with open(self.train_config.best_model_Configuration_Log, 'a', encoding="utf-8") as F1:
            line = 'sim_weight:{sim_weight} | class_weight:{class_weight} | shifting_weight:{shifting_weight} | order_center_weight:{order_center_weight} | ce_loss_weight:{ce_loss_weight} | pred_center_score_weight:{pred_center_score_weight}\n ' \
                   'test_best_MAE:-----------{test_MAE}------------ | RMSE:{RMSE} | Pearson:{Pearson} |\n' \
                   'precision:{precision} | recall:{recall} | f1:{f1} | accuracy:{accuracy} | multiclass_acc:{multiclass_acc} |\n' \
                   'best_mae:{best_mae} | best_rmse:{best_rmse} | best_f1:{best_f1} | best_accuracy:{best_accuracy} | best_multiclass_acc:{best_multiclass_acc} |\n' \
                .format(sim_weight=self.train_config.sim_weight,
                        class_weight=self.train_config.class_weight,
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
                        multiclass_acc=multiclass_acc,
                        best_mae=best_mae,
                        best_rmse=best_rmse,
                        best_f1=best_f1,
                        best_accuracy=best_accuracy,
                        best_multiclass_acc=best_multiclass_acc,
                        )

            print('result saved～')
            F1.write(line)

        checkpoints = 'checkpoints'
        if os.path.exists(checkpoints):
                rmtree(checkpoints, ignore_errors=False, onerror=None)
        os.makedirs(checkpoints)

        # self.tsne(mode="test", best=True)
        # self.picture(mode="test", best=True)
    
    def eval(self, mode=None, to_print=False, best=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []

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
                v, a, y, label_area, label_shifting, l = batch

                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                label_area = to_gpu(label_area)
                label_shifting = to_gpu(label_shifting)
                l = to_gpu(l)
                y_tilde, h = self.model(v, a, l, label_area)
                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()


        mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc = self.calc_metrics(y_true, y_pred)

        return mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc

    def multiclass_acc(self, preds, truths):
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


    def calc_metrics(self, y_true, y_pred):
        test_preds = y_pred
        test_truth = y_true
        mae = np.mean(np.absolute(test_preds - test_truth))
        rmse = np.sqrt(np.mean((test_preds - test_truth) ** 2))
        pearsonrn, p_value = pearsonr(test_preds, test_truth)
        # corr = np.corrcoef(preds, y_test)[0][1]  # preds 和 y_test 之间的相关系数
        preds_b = test_preds >= 10
        y_test_b = test_truth >= 10
        precision = precision_score(y_test_b, preds_b, zero_division=1)
        recall = recall_score(y_test_b, preds_b, zero_division=1)
        f1 = f1_score(y_test_b, preds_b)
        accuracy = accuracy_score(y_test_b, preds_b)
        multiclass_true = np.array(calculate_labels(y_true))
        multiclass_pred = np.array(calculate_labels(y_pred))
        multiclass_acc = np.sum(multiclass_true == multiclass_pred)/ float(len(multiclass_pred))
        return mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc


    def tsne(self, mode=None, to_print=False, best=False):
        assert (mode is not None)
        self.model.eval()

        Feature, Label_area = [], []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

        if best:
            # self.train_config.name = '2024-07-03_22:32:17'  # 文件5
            self.train_config.name = '2024-07-03_22:32:17'
            self.model.load_state_dict(torch.load(
                f'checkpoints/model_{self.train_config.name}.std'))

        with torch.no_grad():
            for batch in self.train_data_loader:
                self.model.zero_grad()
                v, a, y, label_area, label_shifting, l, bert_sent, bert_sent_type, bert_sent_mask = batch

                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                label_area = to_gpu(label_area)
                label_shifting = to_gpu(label_shifting)
                l = to_gpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                feature = self.model(v, a, l, bert_sent, bert_sent_type, bert_sent_mask,
                                                       label_area, mode="tsne")


                Feature.append(feature.detach().cpu().numpy())
                Label_area.append(label_area.detach().cpu().numpy())

            for batch in dataloader:
                self.model.zero_grad()
                v, a, y, label_area, label_shifting, l, bert_sent, bert_sent_type, bert_sent_mask = batch

                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                label_area = to_gpu(label_area)
                label_shifting = to_gpu(label_shifting)
                l = to_gpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                feature = self.model(v, a, l, bert_sent, bert_sent_type, bert_sent_mask,
                                                       label_area, mode="tsne")


                Feature.append(feature.detach().cpu().numpy())
                Label_area.append(label_area.detach().cpu().numpy())

        from sklearn.manifold import TSNE
        import pandas as pd
        import matplotlib.pyplot as plt
        expression = ['healthy', 'light', 'moderate', 'Moderate to severe', 'severe']
        colors = ['green', 'blue', 'yellow', 'orange', 'red']
        features = np.concatenate(Feature, axis=0)
        labels = np.concatenate(Label_area, axis=0)
        tsne1 = TSNE(n_components=2, init="pca", random_state=1, perplexity=5)
        x_tsne1 = tsne1.fit_transform(features)
        print(
            "Data has the {} before tSNE and the following after tSNE {}".format(features.shape[-1], x_tsne1.shape[-1]))
        x_min, x_max = x_tsne1.min(0), x_tsne1.max(0)
        X_norm1 = (x_tsne1 - x_min) / (x_max - x_min)

        ''' plot results of tSNE '''
        fake_df1 = pd.DataFrame(X_norm1, columns=['X', 'Y'])
        fake_df1['Group'] = labels

        group_codes1 = {k: colors[idx] for idx, k in enumerate(np.sort(fake_df1.Group.unique()))}
        fake_df1['colors'] = fake_df1['Group'].apply(lambda x: group_codes1[x])
        mode = ['train']*Feature[0].shape[0] + ['test']*Feature[1].shape[0]
        fake_df1['mode'] = mode

        # 将像素值转换为英寸
        width_in_inches = 1196 / 100
        height_in_inches = 802 / 100
        # 创建固定大小的图像
        fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=100)

        for i in range(fake_df1.Group.unique().shape[0]):
            ax.scatter(X_norm1[fake_df1['Group'] == i, 0],
                       X_norm1[fake_df1['Group'] == i, 1],
                       c=group_codes1[i], label=expression[i], s=70, marker='o', linewidths=1)
        # plt.title('Decomposed features', fontsize=15)
        ax.legend()

        # for i in range(fake_df1.Group.unique().shape[0]):
        #     ax.scatter(X_norm1[(fake_df1['Group'] == i) & (fake_df1['mode'] == 'train'), 0],
        #                X_norm1[(fake_df1['Group'] == i) & (fake_df1['mode'] == 'train'), 1],
        #                c=group_codes1[i], label=expression[i], s=40, marker='o', linewidths=1)
        # # plt.title('Decomposed features', fontsize=15)
        #
        # for i in range(fake_df1.Group.unique().shape[0]):
        #     ax.scatter(X_norm1[(fake_df1['Group'] == i) & (fake_df1['mode'] == 'test'), 0],
        #                X_norm1[(fake_df1['Group'] == i) & (fake_df1['mode'] == 'test'), 1],
        #                c=group_codes1[i], label=expression[i], s=80, marker='*', linewidths=1)
        # # plt.title('Decomposed features', fontsize=15)
        # ax.legend()

        #		plt.legend(loc = 1, fontsize = 'small')
        # ax.legend(fontsize=20, bbox_to_anchor=(-0.015, 0.98, 0.1, 0.1), loc='lower left', ncol=3, columnspacing=1)
        plt.savefig('./figure/TSNE-CMDC5-5-da.png', bbox_inches='tight')
        plt.close("all")


    def picture(self, mode=None, to_print=False, best=False):
        assert (mode is not None)
        self.model.eval()

        Feature, Label_area = [], []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

        if best:
            self.train_config.name = '2024-07-04_15:10:57'  # 文件5 -all
            # self.train_config.name = '2024-07-03_22:32:17'  # 文件5 -nocenter
            self.model.load_state_dict(torch.load(
                f'checkpoints/model_{self.train_config.name}.std'))

        with torch.no_grad():
            for batch in self.train_data_loader:
                self.model.zero_grad()
                t, v, a, y, label_area, label_shifting, l = batch

                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                label_area = to_gpu(label_area)
                label_shifting = to_gpu(label_shifting)
                l = to_gpu(l)

                feature = self.model(t, v, a, l, label_area, mode="tsne")

                Feature.append(feature.detach().cpu().numpy())
                Label_area.append(label_area.detach().cpu().numpy())

            for batch in dataloader:
                self.model.zero_grad()
                t, v, a, y, label_area, label_shifting, l = batch

                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                label_area = to_gpu(label_area)
                label_shifting = to_gpu(label_shifting)
                l = to_gpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                feature = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask,
                                     label_area, mode="tsne")

                Feature.append(feature.detach().cpu().numpy())
                Label_area.append(label_area.detach().cpu().numpy())

        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        expression = ['healthy', 'light', 'moderate', 'Moderate to severe', 'severe']
        colors = ['green', 'blue', 'yellow', 'orange', 'red']
        features = np.concatenate(Feature, axis=0)
        labels = np.concatenate(Label_area, axis=0)
        # 将标签和特征组合在一起，便于排序
        combined = np.hstack((labels, features))
        # 根据标签（即第一列）进行排序
        sorted_combined = combined[combined[:, 0].argsort()]
        # 分离出排序后的标签和特征
        sorted_labels = sorted_combined[:, 0].reshape(-1, 1)
        sorted_features = sorted_combined[:, 1:]
        # 交换sorted_features的两个维度
        transposed_features = sorted_features.T
        # transposed_features = 1 - transposed_features
        # 创建子图
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))

        # 绘制热力图
        sns.heatmap(transposed_features, ax=axes, cmap='viridis', cbar=True)
        # axes.set_title('(a) CMDC5-nocenter')
        axes.set_title('(b) CMDC5')
        axes.set_xlabel('sample')
        axes.set_ylabel('Feature components')
        # axes.add_patch(plt.Rectangle((4, 20), 12, 60, fill=False, edgecolor='red', lw=2))
        # 将sorted_labels转换为适合显示的格式
        sorted_labels_list = [str(int(label[0])) for label in sorted_labels]
        # 设置x轴标签
        axes.set_xticks(np.arange(len(sorted_labels)))
        axes.set_xticklabels(sorted_labels_list, rotation=90)

        # 调整布局
        plt.tight_layout()
        # plt.show()
        plt.savefig('./figure/picture-CMDC5.png', bbox_inches='tight')
        plt.close("all")