import argparse

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset', type=str, default='Alameda')
    parser.add_argument('--model_name', type=str, default='igstgnn')
    parser.add_argument('--seed', type=int, default=2025)

    parser.add_argument('--bs', type=int, default=64)
    # seq_len denotes input history length, horizon denotes output future length
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)

    parser.add_argument('--mode', type=str, default='train')
    # parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    
    parser.add_argument('--incident', action='store_true', default=False)
    parser.add_argument('--use_sensor_info', action='store_true', default=False)
    return parser
