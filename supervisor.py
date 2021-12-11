import os
import time

import numpy as np
import torch
from lib import utils
from experiments import dataloader

from model.clcnn import CLCRNModel, CLCSTNModel
from model.baselines.recurrent import RNNModel
from model.baselines.attention import ATTModel
from model.loss import masked_mae_loss, masked_mse_loss, masked_mape_loss
from tqdm import tqdm
from pathlib import Path
torch.set_num_threads(4)

def exists(val):
    return val is not None

class Supervisor:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self._device = torch.device("cuda:{}".format(kwargs.get('gpu')) if torch.cuda.is_available() else "cpu") 

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        # logging.
        self._experiment_name = self._train_kwargs.get('experiment_name')
        self._log_dir = self._get_log_dir(self, kwargs)

        self._model_name = self._model_kwargs.get('model_name')

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        
        self._data = dataloader.load_dataset(**self._data_kwargs)

        self.standard_scaler = self._data['scaler']
        self.sparse_idx = torch.from_numpy(self._data['kernel_info']['sparse_idx']).long().to(self._device)
        self.location_info = torch.from_numpy(self._data['kernel_info']['MLP_inputs']).float().to(self._device)
        self.geodesic = torch.from_numpy(self._data['kernel_info']['geodesic']).float().to(self._device)
        self.angle_ratio = torch.from_numpy(self._data['kernel_info']['angle_ratio']).float().to(self._device)

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False)
            )
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder
        # setup model
        if self._model_name == 'CLCRN':
            model = CLCRNModel(
                self.location_info, 
                self.sparse_idx, 
                self.geodesic, 
                self.angle_ratio, 
                logger=self._logger, 
                **self._model_kwargs
                )
        elif self._model_name == 'CLCSTN':
            model = CLCSTNModel(
                self.location_info, 
                self.sparse_idx, 
                self.geodesic, 
                self.angle_ratio, 
                logger=self._logger, 
                **self._model_kwargs
            )
        elif self._model_name in ['DCRNN', 'GConvGRU', 'AGCRN', 'TGCN']:
            model = RNNModel(
                sparse_idx=self.sparse_idx, 
                conv_method=self._model_name,
                logger=self._logger,
                **self._model_kwargs
                )
        elif self._model_name in ['ASTGCN', 'MSTGCN', 'STGCN']:
            model = ATTModel(
                sparse_idx=self.sparse_idx, 
                attention_method=self._model_name,
                logger=self._logger,
                **self._model_kwargs
            )
        else:
            print('The method is not provided.')
            exit()
        self.model = model.to(self._device)
        self._logger.info("Model created")
        
        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()

    @staticmethod
    def _get_log_dir(self, kwargs):
        log_dir = Path(kwargs['train'].get('log_dir'))/self._experiment_name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):

        model_path = Path(self._log_dir)/'saved_model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        config = dict(self._kwargs)
        config['model_state_dict'] = self.model.state_dict()
        config['epoch'] = epoch
        torch.save(config, model_path/('epo%d.tar' % epoch))
        self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch

    def load_model(self, epoch_num):

        self._setup_graph()
        model_path = Path(self._log_dir)/'saved_model'
        assert os.path.exists(model_path/('epo%d.tar' % epoch_num)), 'Weights at epoch %d not found' % epoch_num
        checkpoint = torch.load(model_path/('epo%d.tar' % epoch_num), map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.model = self.model.eval()

            val_iterator = self._data['val_loader']

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.model(x)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, dataset, batches_seen, epoch_num, load_model=False, steps=None):

        if load_model == True:
            self.load_model(epoch_num)

        with torch.no_grad():
            self.model = self.model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)]
            losses = []
            y_truths = []
            y_preds = []

            MAE_metric = masked_mae_loss
            MSE_metric = masked_mse_loss
            MAPE_metric = masked_mape_loss
            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.model(x)
                loss, y_true, y_pred = self._compute_loss(y, output)
                losses.append(loss.item())
                y_truths.append(y_true.cpu())
                y_preds.append(y_pred.cpu())

            mean_loss = np.mean(losses)
            y_preds = torch.cat(y_preds, dim=1)
            y_truths = torch.cat(y_truths, dim=1)

            loss_mae = MAE_metric(y_preds, y_truths).item()
            loss_mse = MSE_metric(y_preds, y_truths).item()
            loss_mape = MAPE_metric(y_preds, y_truths).item()
            dict_out = {'prediction': y_preds, 'truth': y_truths}
            dict_metrics = {}
            if exists(steps):
                for step in steps:
                    assert(step <= y_preds.shape[0]), ('the largest step is should smaller than prediction horizon!!!')
                    y_p = y_preds[:step, ...]
                    y_t = y_truths[:step, ...]
                    dict_metrics['mae_{}'.format(step)] = MAE_metric(y_p, y_t).item()
                    dict_metrics['rmse_{}'.format(step)] = MSE_metric(y_p, y_t).sqrt().item()
                    dict_metrics['mape_{}'.format(step)] = MAPE_metric(y_p, y_t).item()

            return loss_mae, loss_mse, loss_mape, dict_out, dict_metrics

    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                            gamma=lr_decay_ratio)

        self._logger.info('Start training ...')
        # this will fail if model is loaded with a changed batch_size
        num_batches = len(self._data['train_loader'])
        self._logger.info("num_batches:{}".format(num_batches))

        best_epoch=0
        batches_seen = num_batches * self._epoch_num
        # val_loss, val_loss_mse, val_loss_mape, _, __ = self.evaluate(dataset='val', batches_seen=batches_seen, epoch_num=0)
        for epo in range(self._epoch_num, epochs):
            
            epoch_num = epo + 1
            self.model = self.model.train()

            train_iterator = self._data['train_loader']
            losses = []

            start_time = time.time()
            progress_bar =  tqdm(train_iterator,unit="batch")

            for _, (x, y) in enumerate(progress_bar): 
                
                optimizer.zero_grad()

                x, y = self._prepare_data(x, y)
                output = self.model(x, y, batches_seen = batches_seen)
                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, eps=epsilon)

                loss, y_true, y_pred = self._compute_loss(y, output)
                
                progress_bar.set_postfix(training_loss=loss.item())
                self._logger.debug(loss.item())

                losses.append(loss.item())

                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                optimizer.step()
            
    
            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")

    
            val_loss, val_loss_mse, val_loss_mape, _, __ = self.evaluate(dataset='val', batches_seen=batches_seen, epoch_num=epoch_num)

            end_time = time.time()

            if (epoch_num % log_every) == 0:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), val_loss, lr_scheduler.get_last_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == 0:
                test_loss, val_loss_mse, val_loss_mape, _, __ = self.evaluate(dataset='test', batches_seen=batches_seen, epoch_num=epoch_num)
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), test_loss, lr_scheduler.get_last_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    best_epoch=epoch_num
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        return x.to(self._device), y.to(self._device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3).float()
        y = y.permute(1, 0, 2, 3).float()
        return x, y

    
    def _compute_loss(self, y_true, y_predicted):
        for out_dim in range(self.output_dim):
            y_true[...,out_dim] = self.standard_scaler[out_dim].inverse_transform(y_true[...,out_dim])
            y_predicted[...,out_dim] = self.standard_scaler[out_dim].inverse_transform(y_predicted[...,out_dim])
        return masked_mae_loss(y_predicted, y_true), y_true, y_predicted

    def _convert_scale(self, y_true, y_predicted):
        for out_dim in range(self.output_dim):
            y_true[...,out_dim] = self.standard_scaler[out_dim].inverse_transform(y_true[...,out_dim])
            y_predicted[...,out_dim] = self.standard_scaler[out_dim].inverse_transform(y_predicted[...,out_dim])
        return y_true, y_predicted
        
    def _prepare_x(self, x):
        x = x.permute(1, 0, 2, 3).float()
        return x.to(self._device)
    
    def _test_final_n_epoch(self, n=5, steps=[3, 6, 12]):
        model_path = Path(self._log_dir)/'saved_model'
        model_list = os.listdir(model_path)
        import re

        epoch_list = []
        for filename in model_list:
            epoch_list.append(int(re.search(r'\d+', filename).group()))

        epoch_list = np.sort(epoch_list)[-n:]
        for i in range(n):
            epoch_num = epoch_list[i]
            mean_score, mean_loss_mse, mean_loss_mape, _, dict_metrics = self.evaluate('test', 0, epoch_num, load_model=True, steps=steps)
            message = "Loaded the {}-th epoch.".format(epoch_num) + \
                " MAE : {}".format(mean_score), "RMSE : {}".format(np.sqrt(mean_loss_mse)), "MAPE : {}".format(mean_loss_mape)
            self._logger.info(message)
            message = "Metrics in different steps: {}".format(dict_metrics)
            self._logger.info(message)
            self._logger.handlers.clear()
    
    def _local_pattern(self, center_nodes, r=0.1, r_resolution=100, phi_resolution=360):
        assert self._model_name in ['CLCRN','CLCSTN'], 'the model does not provide the kernel visualization'
        with torch.no_grad():
            center_nodes = torch.from_numpy(np.array(center_nodes)).float().to(self._device)
            N = center_nodes.shape[0]
            angle_ratio = 1 / phi_resolution
            rs = np.linspace(0, r, r_resolution)
            phis = np.linspace(-np.pi, np.pi, phi_resolution)
            xs = torch.from_numpy(rs[:, None] * np.cos(phis)[None, :]).float().to(self._device).flatten() # r_res * phi_res
            ys = torch.from_numpy(rs[:, None] * np.sin(phis)[None, :]).float().to(self._device).flatten() # r_res * phi_res
            vs = torch.stack([xs, ys], dim=-1)[None, :, :].repeat(N, 1, 1)

            kernel = self.model.get_kernel()
            local_pattern = kernel.kernel_prattern(center_nodes, vs, angle_ratio)
        return local_pattern, center_nodes, rs, phis

    def _local_pattern_visual(self, center_nodes, r=0.1, r_resolution=180, phi_resolution=180):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        local_patterns, center_nodes, rs, phis = self._local_pattern(center_nodes, r, r_resolution, phi_resolution) 
        local_patterns = local_patterns.detach().cpu().numpy()
        rs_mesh, phis_mesh = np.meshgrid(rs, phis)
        vmin, vmax = 0, 0.02
        
        for i in range(center_nodes.shape[0]):
            local_pattern = local_patterns[i].reshape(r_resolution, phi_resolution)
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

            c = ax.pcolormesh(phis_mesh + 0.1 * np.sin(0.1*i*np.pi) * i, rs_mesh, local_pattern.T*np.exp(0.2*np.sin(0.03*i)), cmap='hot', vmin=vmin, vmax=vmax)
            # fig.colorbar(c)
            plt.plot(phis, rs, color='k', ls='none') 
            plt.grid()
            path = self._log_dir
            plt.savefig(path / 'local_kernel_center{}.png'.format(i))
    
    def _get_time_prediction(self):
        import copy
        _data_kwargs = copy.deepcopy(self._data_kwargs)
        _data_kwargs['dataset_dir']  = _data_kwargs['dataset_dir'][:-1] + '_visual'
        _data_kwargs['val_batch_size'] = 1
        _data = dataloader.load_dataset(**_data_kwargs)
        test_loader = _data['test_loader']
        y_preds = []
        y_trues = []
        with torch.no_grad():
            for _, (x, y) in enumerate(test_loader):
                x, y = self._prepare_data(x, y)
                output = self.model(x)
                loss, y_true, y_pred = self._compute_loss(y, output)
                y_preds.append(y_pred)
                y_trues.append(y_true)
            y_preds = torch.cat(y_preds, 0).squeeze(dim=1).cpu().numpy()
            y_trues = torch.cat(y_trues, 0).squeeze(dim=1).cpu().numpy()
            import pickle
            with open('{}.pkl'.format(self._model_name + self._data_kwargs['dataset_dir'][5:-1]), "wb") as f:
                save_data = {'y_preds': y_preds,
                            'y_trues': y_trues}
                pickle.dump(save_data, f, protocol = 4)

        