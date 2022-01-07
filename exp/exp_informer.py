import hfai
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import os
import time

import warnings
warnings.filterwarnings('ignore')


class Exp_Informer:
    def __init__(self, args, local_rank):
        self.args = args

        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
            ).cuda()
        else:
            raise "incorrect model"

        self.model = DistributedDataParallel(model, device_ids=[local_rank])
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.rank = local_rank

        self.save_step_path = os.path.join(self.args.checkpoints, "steps")
        if self.rank == 0 and not os.path.exists(self.save_step_path):
            os.makedirs(self.args.checkpoint, exist_ok=True)
        self.steps = 0

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
        )
        print(flag, len(data_set))

        datasampler = DistributedSampler(data_set)
        data_loader = DataLoader(data_set, batch_size, sampler=datasampler, pin_memory=True, drop_last=drop_last)

        return data_set, data_loader
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        # 若模型存在，加载之前的模型
        if os.path.exists(os.path.join(self.save_step_path, 'optimizer.pt')):
            self.load()
            
        # 保存当前setting下的最好模型    
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = os.path.join(path, 'checkpoint.pth')
        if self.rank ==0 and not os.path.exists(path):
            os.makedirs(path)
        
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        criterion =  self._select_criterion()

        # 获取当前训练的进度位置
        for epoch in range(self.steps//len(train_loader)%self.args.train_epochs, self.args.train_epochs):
            train_loss = []
            
            self.model.train()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                # 对于之前的已经训练过的轮次，直接跳过
                if epoch*len(train_loader) + i < self.steps:
                    print(f'Epoch: {epoch+1}/{self.args.train_epochs}, Step: {i+1}/{len(train_loader)}, Skip.')
                    continue

                self.model_optim.zero_grad()
                pred, true = self._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                   
                # 状态更新
                self.steps += 1
                loss.backward()
                self.model_optim.step()

                # 收到打断信号，保存模型，设置当前执行的状态信息
                if self.rank == 0 and hfai.receive_suspend_command():
                    self.save()
                    vali_loss = self.vali(vali_data, vali_loader, criterion)
                    early_stopping(vali_loss, self.model, path)
                    hfai.set_whole_life_state(self.steps)
                    time.sleep(5)
                    hfai.go_suspend()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            if self.rank == 0:
                print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(epoch + 1, train_loss, vali_loss, test_loss))
                self.save()

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Rank: {}, Early stopping".format(self.rank))
                break

            adjust_learning_rate(self.model_optim, epoch+1, self.args)
            
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        else:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float()
        # encoder - decoder

        if self.args.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:]

        return outputs, batch_y

    def load(self):
        self.model.load_state_dict(torch.load(os.path.join(self.save_step_path, 'model.pt')))
        self.model_optim.load_state_dict(torch.load(os.path.join(self.save_step_path, 'optimizer.pt')))
        self.steps = torch.load(os.path.join(self.save_step_path, 'other'))
        torch.cuda.empty_cache()

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_step_path, 'model.pt'))
        torch.save(self.model_optim.state_dict(), os.path.join(self.save_step_path, 'optimizer.pt'))
        torch.save(self.steps, os.path.join(self.save_step_path, 'other'))
        torch.cuda.empty_cache()


