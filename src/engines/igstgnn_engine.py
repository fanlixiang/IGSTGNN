import numpy as np
import torch
from tqdm import tqdm

from src.base.engine import BaseEngine
from src.utils.metrics import compute_all_metrics, masked_mape, masked_rmse


class IGSTGNN_Engine(BaseEngine):
    """
    Incident-aware IGSTGNN engine.
    """
    def __init__(self, cl_step, warm_step, horizon, incident=False, **args):
        super(IGSTGNN_Engine, self).__init__(**args)
        self._cl_step = cl_step
        self._warm_step = warm_step
        self._horizon = horizon
        self._cl_len = 0
        self._incident = incident

    def _as_float_tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self._device)

    def _as_tensor(self, value):
        return torch.as_tensor(value, device=self._device)

    def _prepare_batch(self, batch):
        if isinstance(batch, dict):
            X = self._as_float_tensor(batch['x_data'])
            label = self._as_float_tensor(batch['y_data'])

            incident_data = None
            if self._incident:
                incident_data = {
                    'incident': self._as_float_tensor(batch['incident_features']),
                    'position': self._as_tensor(batch['incident_position']),
                    'distances': self._as_float_tensor(batch['incident_distances']),
                }

            sensor_data = None
            if 'sensor_data' in batch:
                sensor_data = {
                    key: self._as_tensor(value)
                    for key, value in batch['sensor_data'].items()
                }

            return X, label, incident_data, sensor_data

        X, label = batch
        X = self._as_float_tensor(X)
        label = self._as_float_tensor(label)
        return X, label, None, None

    def _predict(self, X, label, incident_data, sensor_data):
        if incident_data is not None:
            return self.model(
                X,
                label,
                incident_data=incident_data,
                sensor_data=sensor_data,
            )
        return self.model(X, label)

    def _mask_value(self, label):
        mask_value = torch.tensor(0.0, device=label.device)
        if label.min() < 1:
            mask_value = label.min()
        return mask_value

    def _update_curriculum_length(self):
        self._iter_cnt += 1
        if self._iter_cnt < self._warm_step:
            self._cl_len = self._horizon
        elif self._iter_cnt == self._warm_step:
            self._cl_len = 1
        elif self._cl_step > 0:
            if (self._iter_cnt - self._warm_step) % self._cl_step == 0 and self._cl_len < self._horizon:
                self._cl_len += 1

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()

        iterator = self._dataloader['train_loader'].get_iterator()
        total_batches = self._dataloader['train_loader'].num_batch
        progress_bar = tqdm(
            iterator,
            total=total_batches,
            desc='Training',
            unit='batch',
            leave=False,
            position=0,
            dynamic_ncols=True,
            colour='green',
        )

        try:
            for batch in progress_bar:
                self._optimizer.zero_grad()

                X, label, incident_data, sensor_data = self._prepare_batch(batch)
                pred = self._predict(X, label, incident_data, sensor_data)
                pred, label = self._inverse_transform([pred, label])

                mask_value = self._mask_value(label)
                if self._iter_cnt == 0:
                    print('check mask value', mask_value)

                self._update_curriculum_length()
                pred = pred[:, :self._cl_len, :, :]
                label = label[:, :self._cl_len, :, :]

                loss = self._loss_fn(pred, label, mask_value)
                mape = masked_mape(pred, label, mask_value).item()
                rmse = masked_rmse(pred, label, mask_value).item()

                loss.backward()
                if self._clip_grad_value != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
                self._optimizer.step()

                train_loss.append(loss.item())
                train_mape.append(mape)
                train_rmse.append(rmse)

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mape': f'{mape:.4f}',
                    'rmse': f'{rmse:.4f}',
                })
        finally:
            progress_bar.close()

        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)

    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for batch in self._dataloader[mode + '_loader'].get_iterator():
                X, label, incident_data, sensor_data = self._prepare_batch(batch)
                pred = self._predict(X, label, incident_data, sensor_data)
                pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        mask_value = torch.tensor(0.0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        if mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))
