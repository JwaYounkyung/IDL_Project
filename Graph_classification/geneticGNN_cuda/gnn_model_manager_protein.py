import numpy as np
import os.path as osp
import time
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.data import DataLoader
from gnn import GraphClassfier

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

def evaluate(model, dataloader):
    output = []
    labels = []
    for data in dataloader:
        logits = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        logits = F.log_softmax(logits, 1)
        output.extend(logits)
        labels.extend(data.y.to(device))
    output = torch.stack(output)
    labels = torch.stack(labels)   
    
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices == labels)
    # return correct.item() * 1.0 / mask.sum().item()
    return correct.item() * 1.0 / len(labels)

class GNNModelManager(object):
    
    def __init__(self, args):
        
        self.args = args
        self.loss_fn = torch.nn.functional.nll_loss
        
    
    def load_data(self, dataset='PROTEINS'):
        
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
        # dataset = Planetoid(path, dataset)#, T.NormalizeFeatures())
        dataset = TUDataset(path, dataset)
        
        
#         print(np.sum(np.array(data.val_mask), 0))
        
#         data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
#         data.train_mask[:-1000] = 1
#         data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
#         data.val_mask[data.num_nodes - 1000: data.num_nodes - 500] = 1
#         data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
#         data.test_mask[data.num_nodes - 500:] = 1
        
        self.data = dataset
        data = dataset[0]
        
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Is undirected: {data.is_undirected()}')
        
        # self.args.num_class = data.y.max().item() + 1
        self.args.num_class = self.data.num_classes
        self.args.in_feats = self.data.num_features
        
        
        # make dataloader
        dataset = dataset.shuffle()
        ix1 = int(len(dataset) * 0.6)
        ix2 = int(len(dataset) * 0.8)
        train_dataset = dataset[:ix1]
        valid_dataset = dataset[ix1:ix2]
        test_dataset = dataset[ix2:]
        
        self.trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.validloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
        self.testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
    
    def load_param(self):
        # don't share param
        pass    

    def update_args(self, args):
        self.args = args

    def save_param(self, model, update_all=False):
        pass

    def shuffle_data(self, full_data=True):
        device = torch.device('cuda' if self.args.cuda else 'cpu')
        if full_data:
            self.data = fix_size_split(self.data, self.data.num_nodes - 1000, 500, 500)
        else:
            self.data = fix_size_split(self.data, 1000, 500, 500)
        self.data.to(device)
            
        
    def build_gnn(self, actions, drop_outs):
        
        model = GraphClassfier(self.args.num_gnn_layers,
                         actions, self.args.in_feats, self.args.num_class, 
                         drop_outs=drop_outs, multi_label=False,
                         batch_normal=False, residual=False)

        return model
        
    # train from scratch
    def evaluate(self, actions=None, format="two"):
        actions = process_action(actions, format, self.args)
        print("train action:", actions)

        # create model
        model = self.build_gnn(actions)
        model.to(device)

        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        try:
            model, val_acc, test_acc = self.run_model(model, optimizer, self.loss_fn, self.data, self.epochs,
                                                      cuda=self.args.cuda, return_best=True,
                                                      half_stop_score=max(self.reward_manager.get_top_average() * 0.7,
                                                                          0.4))
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
                test_acc = 0
            else:
                raise e
        return val_acc, test_acc
        
    # train from scratch
    def train(self, actions=None, params=None):
        # change the last gnn dimension to num_class
        # actions[-1] = self.args.num_class
        print('==================================\ncurrent training actions={}, params={}'.format(actions, params))
        
        # create gnn model
        learning_rate = params[-2]
        weight_decay = params[-1]
        drop_outs = params[:-2]
        
        gnn_model = self.build_gnn(actions, drop_outs)
        gnn_model.to(device)
        print(gnn_model)
        
        # define optimizer
        optimizer = torch.optim.Adam(gnn_model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        
        # run model to get accuracy
        model, val_acc, test_acc = self.run_model(gnn_model, 
                                        optimizer, 
                                        self.loss_fn, 
                                        self.trainloader,
                                        self.validloader,
                                        self.testloader, 
                                        self.args.epochs,
                                        show_info=False)

        return val_acc, test_acc
        
    @staticmethod
    def run_model(model, optimizer, loss_fn, trainloader, validloader, testloader, epochs, early_stop=5, 
                  return_best=False, cuda=True, need_early_stop=False, show_info=False):
        dur = []
        begin_time = time.time()
        best_performance = 0
        min_val_loss = float("inf")
        min_train_loss = float("inf")
        model_val_acc = 0
        model_test_acc = 0
        best_epoch = 0
#         print("Number of train datas:", data.train_mask.sum())
        for epoch in range(1, epochs + 1):
            
            model.train()
#             print(data.edge_index.shape, data.x.shape, data.y.shape)
            train_loss = 0
            for data in trainloader:
                logits = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
                logits = F.log_softmax(logits, 1)
            
                loss = loss_fn(logits, data.y.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(trainloader)

            # evaluate
            model.eval()          
            
            # train_acc = evaluate(logits.to(device), data.y.to(device), data.train_mask.to(device))
            # val_acc = evaluate(logits.to(device), data.y.to(device), data.val_mask.to(device))
            # test_acc = evaluate(logits.to(device), data.y.to(device), data.test_mask.to(device))
            
            train_acc = evaluate(model, trainloader)            
            val_acc = evaluate(model, validloader)
            test_acc = evaluate(model, testloader)


            val_loss = 0
            for data in validloader:
                logits = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
                logits = F.log_softmax(logits, 1)
            
                loss = loss_fn(logits, data.y.to(device))
                val_loss += loss.item()
            val_loss /= len(validloader)
            
            # loss = loss_fn(logits[data.val_mask].to(device), data.y[data.val_mask].to(device))
            # val_loss = loss.item()
            if val_loss < min_val_loss:  # and train_loss < min_train_loss
                min_val_loss = val_loss
                min_train_loss = train_loss
                model_val_acc = val_acc
                model_test_acc = test_acc
                best_epoch = epoch
                if test_acc > best_performance:
                    best_performance = test_acc
            if show_info:
                time_used = time.time() - begin_time
                print(
                    "Epoch {:05d} | Loss {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f} | time {}".format(
                        epoch, val_loss, train_acc, val_acc, test_acc, time_used))

#                 print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
        print("val_score:{:.4f}, test_score:{:.4f}, best_epoch:{}".format(model_val_acc, model_test_acc, best_epoch), '\n')
        if return_best:
            return model, model_val_acc, best_performance
        else:
            return model, model_val_acc, model_test_acc
        

    @staticmethod
    def prepare_data(data, cuda=True):
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        mask = torch.ByteTensor(data.train_mask)
        test_mask = torch.ByteTensor(data.test_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        n_edges = data.graph.number_of_edges()
        # create DGL graph
        g = DGLGraph(data.graph)
        # add self loop
        g.add_edges(g.nodes(), g.nodes())
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0

        if cuda:
            features = features.to(device)
            labels = labels.to(device)
            norm = norm.to(device)
        g.ndata['norm'] = norm.unsqueeze(1)
        return features, g, labels, mask, val_mask, test_mask, n_edges        