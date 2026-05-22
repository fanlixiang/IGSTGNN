#!/bin/bash

python experiments/IGSTGNN/main.py --dataset Alameda --model_name igstgnn --seed 2025 --bs 48 --incident --device cuda:0 --use_sensor_info
