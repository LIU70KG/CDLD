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
                lr=self.train_config.learning_rate)  # , weight_decay=1e-3


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
        train_losses = []
        valid_losses = []
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

        # 初始化用于存储训练和测试 MAE 的列表
        train_mae_history = []
        test_mae_history = []
        # 设置绘图窗口
        # plt.ion()  # 开启交互模式
        # fig, ax = plt.subplots(figsize=(10, 6))

        for e in range(continue_epochs, self.train_config.n_epoch):
            # print(f"-----------------------------------epoch{e}---------------------------------------")
            # print(f"//Current patience: {curr_patience}, current trial: {num_trials}.//")
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
            center_score_tensor = torch.tensor(center_score).view(-1, 1).cuda()
            for batch in self.train_data_loader:
                # self.model.zero_grad()
                self.optimizer.zero_grad()
                v, a, y, label_area, label_shifting, l = batch
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                label_area = to_gpu(label_area)
                label_shifting = to_gpu(label_shifting)
                l = to_gpu(l)

                # if e == 200 :
                #     print(label_area.squeeze(1))
                #     okok = 1
                # with torch.autograd.detect_anomaly():
                    #     # 前向传播
                    #     outputs = model(x)
                    #     loss = criterion(outputs, y)
                    #     # 反向传播
                    #     optimizer.zero_grad()
                    #     loss.backward()
                    #     optimizer.step()
                # o = self.model(t, v, a, l, bert_sent,bert_sent_type, bert_sent_mask)
                pred, pred_center_score, p_class, p_shifting, uncertainty, order_center_loss = self.model(v, a, l, label_area, mode="train")
                # -----------------不确定性正则化，30%高不确定性，70%低不确定性，间隔为0.15------------------------
                # batch_size = t.size(1)
                # beta = 0.5
                # margin_1 = 0.5
                # tops = int(batch_size * beta)
                # # 防止出现批次中无高不确定情况 / 无低不确定情况
                # tops = 1 if tops == 0 else (batch_size - 1 if tops == batch_size else tops)
                # _, top_idx = torch.topk(uncertainty.squeeze(), tops)
                # _, down_idx = torch.topk(uncertainty.squeeze(), batch_size - tops, largest=False)
                # high_group = uncertainty[top_idx]
                # low_group = uncertainty[down_idx]
                # high_mean = torch.mean(high_group)
                # low_mean = torch.mean(low_group)
                # diff = low_mean - high_mean + margin_1
                # RR_loss = diff if diff > 0 else 0.0
                # RR_loss = 0
                # -----------------区间定位损失：动态调整方差的高斯分布标签------------------------
                # # 将不确定程度缩放到 A 到 B 之间用于放大与缩小方差
                # # uncertainty = self.scale_uncertainty(uncertainty, min_val=0.1, max_val=0.5)
                # 生成动态调整方差的高斯分布标签 (batch_size, num_classes)
                # label_area_true = self.gaussian_label_distribution(label_area, 5, uncertainty, sigmas=1)
                # class_loss = self.gaussian_kl_div_loss(p_class, label_area_true)   # 计算损失
                # shifting_loss = criterion(p_shifting, label_shifting)   # 偏移损失
                # # loss = 5*RR_loss + shifting_loss + class_loss
                # ----------------------------------------------------------------

                # -----------------区间定位损失：区间距离的平方------------------------
                class_loss = self.ce_ordinal_loss(p_class, label_area, self.train_config.interval_num)  # 有序区域定位损失  Ordered regional positioning
                # class_loss = ce_criterion(p_class, label_area.long().squeeze())
                # 问题：偏移的好像是重心，而不是区间中心。自己是0,而大多数人也都是0,不存在偏移。但我们希望它离区间中心偏移了-2
                shifting_loss = criterion_MAE(p_shifting, label_shifting)   # 偏移损失
                # 如果不加下面的，无法说样本落在区间内，基点得分将是2、7、12、17、22。因为类内样本不平衡，score中0 1 2 3 4里，大多数是0
                # y_tilde = torch.argmax(p_class, dim=1) * 5 + 2 + torch.squeeze(p_shifting)
                # y_tilde = y_tilde.unsqueeze(1)
                # # score_loss = criterion(y_tilde, y)
                # score_loss = self.MAE(y_tilde, y)  # 分数损失
                # interval_center = label_area * 5 + 2
                # score_loss = self.MAE(pred, interval_center)  # 分数损失
                # score_loss = self.interval_center_loss(pred, label_area, 5)  # 区域中心损失
                # --------------------------------------------------------------
                score_loss = criterion_MAE(pred, y)  # 直接的分数损失
                pred_center_score_loss = criterion_MAE(pred_center_score, center_score_tensor)
                # score_loss = 0
                if self.train_config.model == 'MISA_CMDC':
                    diff_loss = self.get_diff_loss()
                    domain_loss = self.get_domain_loss()
                    recon_loss = self.get_recon_loss()
                    cmd_loss = self.get_cmd_loss()

                    if self.train_config.use_cmd_sim:
                        similarity_loss = cmd_loss
                    else:
                        similarity_loss = domain_loss

                    loss = self.train_config.class_weight * class_loss + \
                           self.train_config.shifting_weight * shifting_loss + \
                           self.train_config.order_center_weight * order_center_loss + \
                        self.train_config.diff_weight * diff_loss + \
                        self.train_config.sim_weight * similarity_loss + \
                        self.train_config.recon_weight * recon_loss+ \
                        self.train_config.pred_center_score_weight * pred_center_score_loss

                    diff_loss_.append(diff_loss.item())
                    similarity_loss_.append(similarity_loss.item())
                    recon_loss_.append(recon_loss.item())
                else:
                    loss = self.train_config.class_weight * class_loss + \
                           self.train_config.shifting_weight * shifting_loss + \
                           self.train_config.order_center_weight * order_center_loss + \
                           self.train_config.pred_center_score_weight * pred_center_score_loss

                    # loss = self.train_config.order_center_weight * order_center_loss + 0.01 * score_loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_([param for param in self.model.parameters() if param.requires_grad],
                                               self.train_config.clip)
                # torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.train_config.clip)
                # if math.isnan(loss) or (e == 24 and order_center_loss_.__len__() == 170):
                #     print(torch.argmax(p_class, dim=1) * 5)
                #     print(label_area.squeeze(1) * 5 + 2)

                self.optimizer.step()
                train_loss.append(loss.item())
                shifting_loss_.append(shifting_loss.item())
                class_loss_.append(class_loss.item())
                score_loss_.append(score_loss.item())
                order_center_loss_.append(order_center_loss.item())
                pred_center_score_loss_.append(pred_center_score_loss.item())
                    # print("iter:%d./ loss:%.4f." %(order_center_loss_.__len__(), loss))

            # train_losses.append(train_loss)

            # print(f"Training loss: {round(np.mean(train_loss), 4)}")
            # print('class_loss_:%.4f./ shifting_loss_:%.4f./ score_loss_:%.4f./ order_center_loss_:%.4f./' % (
            # round(np.mean(class_loss_), 4), round(np.mean(shifting_loss_), 4), round(np.mean(score_loss_), 4),
            # round(np.mean(order_center_loss_), 4)))

            # print('class_loss_:%.4f./ shifting_loss_:%.4f./ order_center_loss_:%.4f./ score_loss_:%.4f./ pred_center_score_loss_:%.4f./' % (
            # round(np.mean(class_loss_), 4), round(np.mean(shifting_loss_), 4),
            # round(np.mean(order_center_loss_), 4), round(np.mean(score_loss_), 4), round(np.mean(pred_center_score_loss_), 4)))
            # if self.train_config.model == 'MISA_CMDC':
            #     print('similarity_loss_:%.4f./ diff_loss_:%.4f./ recon_loss_:%.4f./' % (
            #     round(np.mean(similarity_loss_), 4), round(np.mean(diff_loss_), 4), round(np.mean(recon_loss_), 4)))

            # valid_loss, mae, rmse, pearsonrn, precision, recall, f1, accuracy = self.eval(mode="dev")
            # print(f"valid loss: {valid_loss}")
            # print('-valid_MAE:%.4f.   RMSE:%.4f.  Pearson:%.4f.' % (mae, rmse, pearsonrn))
            # print('-precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./' % (precision, recall, f1, accuracy))
            # if e == 38:
            #     torch.save(self.model.state_dict(), f'checkpoints/model_epoch{e}.std')
            #     torch.save(self.optimizer.state_dict(), f'checkpoints/optim_epoch{e}.std')
            # print(f"--------------------------------------------")
            # mae, rmse, pearsonrn, precision, recall, f1, accuracy = self.eval(mode="dev")
            mae_train, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc = self.eval(mode="train")
            # print('_train_MAE:%.4f.   RMSE:%.4f.  Pearson:%.4f.' % (mae_train, rmse, pearsonrn))
            # print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./ multiclass_acc:%.4f./' % (precision, recall, f1, accuracy, multiclass_acc))

            # mae, rmse, pearsonrn, precision, recall, f1, accuracy = self.eval(mode="dev")
            mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc = self.eval(mode="test")
            # print('_test_MAE:%.4f.   RMSE:%.4f.  Pearson:%.4f.' % (mae, rmse, pearsonrn))
            # print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./ multiclass_acc:%.4f./' % (precision, recall, f1, accuracy, multiclass_acc))
            # 将当前 epoch 的 MAE 添加到列表中
            # train_mae_history.append(mae_train)
            test_mae_history.append(mae)
            # # 清除之前的图像
            # clear_output(wait=True)
            # # 更新图像
            # ax.clear()
            # ax.plot(train_mae_history, marker='o', linestyle='-', label="Train MAE", color="blue")
            # ax.plot(test_mae_history, marker='o', linestyle='-', label="Test MAE", color="orange")
            # ax.set_xlabel("Number of epochs")
            # ax.set_ylabel("Mean absolute error")
            # # ax.set_title("Training and Testing MAE over Epochs")
            # ax.legend()
            # ax.grid(True)  # 启用网格
            #
            # # 暂停 0.1 秒以更新图形并继续训练
            # plt.pause(0.1)

            # if valid_loss <= best_valid_loss:
            #     best_valid_loss = valid_loss
            #     print("Found new best model on dev set!")
            #     print(f"-----------------valid loss: {valid_loss}")
            # if best_mae > mae or best_rmse > rmse:
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
            if best_multiclass_acc < multiclass_acc:
                best_multiclass_acc = multiclass_acc
            if best_f1 < f1:
                best_precision, best_recall, best_f1 = precision, recall, f1
            if flag == 1:
                # print("------------------Found new best model on test set!----------------")
                # print(f"epoch: {e}")
                # print("mae: ", mae)
                # print("rmse: ", rmse)
                # print("pearsonrn: ", pearsonrn)
                # print("precision: ", precision)
                # print("recall: ", recall)
                # print("f1: ", f1)
                # print("accuracy: ", accuracy)
                # print("multiclass_acc: ", multiclass_acc)
                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'checkpoints/model_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'checkpoints/optim_{self.train_config.name}.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    # print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'checkpoints/model_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_{self.train_config.name}.std'))
                    lr_scheduler.step()
                    # print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
            
            if num_trials <= 0:
                # print("Running out of patience, early stopping.")
                break


        # plt.show()  # 最后显示完整图表
        # # 保存图形到文件
        # plt.savefig('mae_plot.png', dpi=300)  # 将图表保存为PNG文件，可以更改文件名和格式
        # 保存test_mae_history到文本文件（每个元素占一行）
        # with open(self.train_config.test_mae_history_path, 'w') as f:
        #     for item in test_mae_history:
        #         f.write(f"{item}\n")

        # print("------------------best all on test set----------------")
        # print('_best_mae:%.4f. / best_rmse:%.4f. / best_f1:%.4f. / best_accuracy: %.4f. / best_multiclass_acc: %.4f.' % (best_mae, best_rmse, best_f1, best_accuracy, best_multiclass_acc))
        # print("------------------best MAE on test set----------------")
        mae, rmse, pearsonrn = best_mae, rmse_bestmae, pearsonrn_bestmae
        precision, recall, f1, accuracy, multiclass_acc= precision_bestmae, recall_bestmae, f1_bestmae, accuracy_bestmae, multiclass_acc_bestmae
        # print('_test_MAE:%.4f.   RMSE:%.4f.  Pearson:%.4f.' % (mae, rmse, pearsonrn))
        # print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./ multiclass_acc:%.4f./' % (precision, recall, f1, accuracy, multiclass_acc))

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
        return mae
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
                pred, pred_center_score, p_class, p_shifting = self.model(v, a, l, label_area, mode="test")


                # weights = torch.tensor([0, 1, 2, 3, 4], device='cuda:0')
                # result = p_class * weights
                # row_sums = torch.sum(result, dim=1)
                # y_tilde = row_sums * 5 + 2 + torch.squeeze(p_shifting)
                center_score = self.train_config.center_score
                label_pre = torch.argmax(p_class, dim=1)
                # print(label_pre)
                center_score_values = torch.tensor([center_score[i] for i in label_pre], dtype=torch.float32).cuda()
                if self.train_config.shifting_weight == 0:
                    y_tilde = center_score_values
                else:
                    y_tilde = center_score_values + torch.squeeze(p_shifting)
                # y_tilde = torch.squeeze(pred) + torch.squeeze(p_shifting)
                # y_tilde = 0.5*(torch.argmax(p_class, dim=1) * 5 + 2 + torch.squeeze(p_shifting)) + 0.5*pred.squeeze()
                # -----------------------------直接回归--------------------
                # y_tilde = pred.squeeze()
                # --------------------------------------------------------
                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()


        mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc = self.calc_metrics(y_true, y_pred, mode, to_print)

        return mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc

    def multiclass_acc(self, preds, truths):
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        test_preds = y_pred
        test_truth = y_true
        mae = np.mean(np.absolute(test_preds - test_truth))
        rmse = np.sqrt(np.mean((test_preds - test_truth) ** 2))
        pearsonrn, p_value = pearsonr(test_preds, test_truth)
        # corr = np.corrcoef(preds, y_test)[0][1]  # preds 和 y_test 之间的相关系数
        preds_b = test_preds >= 14
        y_test_b = test_truth >= 14
        precision = precision_score(y_test_b, preds_b, zero_division=1)
        recall = recall_score(y_test_b, preds_b, zero_division=1)
        f1 = f1_score(y_test_b, preds_b)
        accuracy = accuracy_score(y_test_b, preds_b)
        multiclass_true = np.array(calculate_labels(y_true))
        multiclass_pred = np.array(calculate_labels(y_pred))
        multiclass_acc = np.sum(multiclass_true == multiclass_pred)/ float(len(multiclass_pred))
        return mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc



    def get_domain_loss(self,):

        if self.train_config.use_cmd_sim:
            return 0.0
        
        # Predicted domain labels
        domain_pred_v = self.model.domain_label_v
        domain_pred_a = self.model.domain_label_a

        # True domain labels
        domain_true_v = to_gpu(torch.LongTensor([0]*domain_pred_v.size(0)))
        domain_true_a = to_gpu(torch.LongTensor([1]*domain_pred_a.size(0)))

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_v, domain_true_a), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)

    def get_cmd_loss(self,):

        if not self.train_config.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd(self.model.utt_shared_a, self.model.utt_shared_v, 5)
        loss = loss

        return loss

    def get_diff_loss(self):

        shared_v = self.model.utt_shared_v
        shared_a = self.model.utt_shared_a
        private_v = self.model.utt_private_v
        private_a = self.model.utt_private_a

        # Between private and shared
        loss = self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)

        # Across privates
        loss += self.loss_diff(private_a, private_v)
        loss = loss / 3.0
        return loss
    
    def get_recon_loss(self, ):

        loss = self.loss_recon(self.model.utt_v_recon, self.model.utt_v_orig)
        loss += self.loss_recon(self.model.utt_a_recon, self.model.utt_a_orig)
        loss = loss/2.0
        return loss


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
        # class_distance = torch.abs(class_range - target_expand)
        order_loss = torch.sum(class_distance * probs, dim=1)
        weights = self.train_config.weights.cuda()

        # 计算总损失
        ce_ordinal_loss = weights[targets] * (self.train_config.ce_loss_weight * ce_loss + order_loss)
        return ce_ordinal_loss.mean()
        # return ce_ordinal_loss.sum()



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