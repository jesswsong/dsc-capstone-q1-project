#!/usr/bin/env python

# Import libraries
import sys
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll

from source import create_dataset, forward_q_x, visualize_forward, train, visualize_output

def main(targets):
    with open('config.json') as f:
        params = json.load(f)

    data_params = params['data']
    data, dataset, s_color = create_dataset(**data_params)
    q_x = forward_q_x(dataset)
        
    if 'visualize' in targets:
        visualize_forward(dataset, s_color)
    
    if 'random' in targets:
        print('with random z')
        model = train(params['batch_size'], params['num_epoch'], dataset, s_color, q_x, random_z=True)
        visualize_output(model, data, dataset, q_x, s_color, random_z=True)
    else:
        print('without random z')
        model = train(params['batch_size'], params['num_epoch'], dataset, s_color, q_x)
        visualize_output(model, data, dataset, q_x, s_color)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
