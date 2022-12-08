import argparse
import time
import torch
import configs
import tensor_utils as utils
from population import Population
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main(args):
    start_time = time.time()
    torch.manual_seed(args.random_seed) 
    
    utils.makedirs(args.dataset)  
    
    print(args.super_ratio)
    
    pop = Population(args)
    pop.evolve_net(start_time)

    # run on single model
    # num_epochs = 200
    # actions = ['gcn', 'mean', 'softplus', 16, 8, 'gcn', 'max', 'relu', 16, 32, 'sum_pool'] 
    # params = [0.05, 0.05, 5e-4, 5e-4]
    # pop.single_model_run(num_epochs, actions, params)
    
    
if __name__ == "__main__":
    args = configs.build_args('GeneticGNN')
    main(args)