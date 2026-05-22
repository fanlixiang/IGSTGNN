import csv
import os
import pickle

import numpy as np
import torch


class IncidentDataLoader(object):
    def __init__(self, samples, incident, bs, logger, input_dim=None, sensor_info=None):
        self.samples = list(samples)
        self.incident = incident
        self.size = len(self.samples)
        self.bs = bs
        self.num_batch = int(np.ceil(self.size / self.bs)) if self.size > 0 else 0
        self.current_ind = 0
        self.input_dim = input_dim
        self.sensor_info = sensor_info

        if self.size == 0:
            raise ValueError('No samples were loaded.')

        first_sample = self.samples[0]
        self._validate_sample(first_sample)
        self.seq_len = first_sample['x_data'].shape[0]
        self.horizon = first_sample['y_data'].shape[0]
        self.num_nodes = first_sample['x_data'].shape[1]

        logger.info(f'Samples: {self.size}, Batches: {self.num_batch}')
        logger.info(
            f'Shape: x=({self.seq_len},{self.num_nodes},{self.input_dim}), '
            f'y=({self.horizon},{self.num_nodes},1)'
        )

    def _validate_sample(self, sample):
        required = ['x_data', 'y_data']
        if self.incident:
            required.extend([
                'incident_features',
                'incident_position',
                'incident_distances',
            ])
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f'Sample is missing required keys: {missing}')

    def shuffle(self):
        indices = np.random.permutation(self.size)
        self.samples = [self.samples[i] for i in indices]

    def _stack_xy(self, batch_samples):
        x = np.stack([
            np.asarray(sample['x_data'], dtype=np.float32)[..., :self.input_dim]
            for sample in batch_samples
        ], axis=0)
        y = np.stack([
            np.asarray(sample['y_data'], dtype=np.float32)[..., :1]
            for sample in batch_samples
        ], axis=0)
        return x, y

    def _incident_features_to_array(self, features):
        if isinstance(features, dict):
            return np.asarray([
                features.get('Incident Time', 0.0),
                features.get('Description', 0),
                features.get('Type', 0),
                features.get('Holiday', 0),
            ], dtype=np.float32)
        return np.asarray(features, dtype=np.float32)

    def _sensor_batch(self, batch_size):
        if self.sensor_info is None:
            return None
        return {
            key: torch.as_tensor(value).unsqueeze(0).expand(batch_size, -1)
            for key, value in self.sensor_info.items()
            if key in ['sensor_type', 'surface', 'roadway_use', 'road_width', 'speed_limit']
        }

    def _build_batch(self, batch_samples):
        x, y = self._stack_xy(batch_samples)
        if not self.incident:
            return x, y

        batch_size = len(batch_samples)
        batch = {
            'x_data': x,
            'y_data': y,
            'incident_features': np.stack([
                self._incident_features_to_array(sample['incident_features'])
                for sample in batch_samples
            ], axis=0),
            'incident_position': np.asarray([
                sample['incident_position'] for sample in batch_samples
            ], dtype=np.int64),
            'incident_distances': np.stack([
                np.asarray(sample['incident_distances'], dtype=np.float32)
                for sample in batch_samples
            ], axis=0),
        }

        sensor_data = self._sensor_batch(batch_size)
        if sensor_data is not None:
            batch['sensor_data'] = sensor_data

        return batch

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                batch_samples = self.samples[start_ind:end_ind]
                self.current_ind += 1
                yield self._build_batch(batch_samples)

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    def transform(self, data):
        mean = self.mean.to(data.device) if torch.is_tensor(data) else self.mean
        std = self.std.to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = self.mean.to(data.device) if torch.is_tensor(data) else self.mean
        std = self.std.to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


def _required_file(data_path, filename):
    path = os.path.join(data_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Required dataset file not found: {path}')
    return path


def _clean_text(value):
    if value is None:
        return 'Unknown'
    value = str(value).strip()
    return value if value else 'Unknown'


def _float_value(value, default):
    try:
        if value is None or str(value).strip() == '':
            return default
        return float(value)
    except ValueError:
        return default


def _first_column(row, candidates, default=None):
    for name in candidates:
        if name in row:
            return row[name]
    return default


def _encode_categories(values):
    mapping = {}
    encoded = []
    for value in values:
        key = _clean_text(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        encoded.append(mapping[key])
    return np.asarray(encoded, dtype=np.int64), mapping


def _load_sensor_info(data_path, node_num, args, logger):
    if not getattr(args, 'use_sensor_info', False):
        args.sensor_type_size = 1
        args.surface_size = 1
        args.roadway_use_size = 1
        return None

    sensor_file = _required_file(data_path, 'sensors.csv')
    with open(sensor_file, 'r', newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    if len(rows) != node_num:
        raise ValueError(
            f'sensors.csv has {len(rows)} rows, but dataset {args.dataset} expects {node_num} nodes.'
        )

    sensor_type_values = [_first_column(row, ['Sensor Type', 'Type'], 'Unknown') for row in rows]
    surface_values = [_first_column(row, ['Surface'], 'Unknown') for row in rows]
    roadway_use_values = [_first_column(row, ['Roadway Use'], 'Unknown') for row in rows]
    road_width_values = [
        _float_value(_first_column(row, ['Road Width'], 0.0), 0.0)
        for row in rows
    ]
    speed_limit_values = [
        _float_value(_first_column(row, ['Design Speed Limit', 'Speed Limit'], 50.0), 50.0)
        for row in rows
    ]

    sensor_type, sensor_type_mapping = _encode_categories(sensor_type_values)
    surface, surface_mapping = _encode_categories(surface_values)
    roadway_use, roadway_use_mapping = _encode_categories(roadway_use_values)

    args.sensor_type_size = max(len(sensor_type_mapping), 1)
    args.surface_size = max(len(surface_mapping), 1)
    args.roadway_use_size = max(len(roadway_use_mapping), 1)

    logger.info(
        'Loaded sensor metadata: '
        f'{len(rows)} rows, '
        f'{args.sensor_type_size} sensor types, '
        f'{args.surface_size} surfaces, '
        f'{args.roadway_use_size} roadway uses'
    )

    return {
        'sensor_type': sensor_type,
        'surface': surface,
        'roadway_use': roadway_use,
        'road_width': np.asarray(road_width_values, dtype=np.float32),
        'speed_limit': np.asarray(speed_limit_values, dtype=np.float32),
    }


def load_dataset(data_path, args, logger):
    dataloader = {}
    node_num = getattr(args, 'node_num', 521)
    sensor_info = _load_sensor_info(data_path, node_num, args, logger)

    for split in ['train', 'val', 'test']:
        file_path = _required_file(data_path, f'incident_{split}.npy')
        samples = np.load(file_path, allow_pickle=True)
        if len(samples) == 0:
            raise ValueError(f'No samples in {file_path}')

        first_sample = samples[0]
        x_shape = first_sample['x_data'].shape
        y_shape = first_sample['y_data'].shape
        logger.info(f'{split}: samples={len(samples)}, x={x_shape}, y={y_shape}')

        dataloader[split + '_loader'] = IncidentDataLoader(
            samples=samples,
            incident=args.incident,
            bs=args.bs,
            logger=logger,
            input_dim=args.input_dim,
            sensor_info=sensor_info,
        )

    stats_file = _required_file(data_path, 'incident_stats.npz')
    stats = np.load(stats_file, allow_pickle=True)
    logger.info(f"Stats: mean={stats['mean']}, std={stats['std']}")
    scaler = StandardScaler(mean=stats['mean'], std=stats['std'])

    return dataloader, scaler


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)


def get_dataset_info(dataset):
    base_dir = os.path.join(os.getcwd(), 'data')
    dataset_info = {
        'Alameda': [
            os.path.join(base_dir, 'xtraffic', 'Alameda'),
            os.path.join(base_dir, 'xtraffic', 'Alameda', 'adj_matrix.npy'),
            521,
        ],
        'Contra_Costa': [
            os.path.join(base_dir, 'xtraffic', 'Contra_Costa'),
            os.path.join(base_dir, 'xtraffic', 'Contra_Costa', 'adj_matrix.npy'),
            496,
        ],
        'Orange': [
            os.path.join(base_dir, 'xtraffic', 'Orange'),
            os.path.join(base_dir, 'xtraffic', 'Orange', 'adj_matrix.npy'),
            990,
        ],
    }
    if dataset not in dataset_info:
        supported = ', '.join(sorted(dataset_info))
        raise ValueError(f'Unknown dataset: {dataset}. Supported datasets: {supported}')
    return dataset_info[dataset]
