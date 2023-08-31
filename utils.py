import sys
import math
import copy
import torch
import random
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
from torch.utils.data import Dataset

class Complete(object):
    def __call__(self, data):
        if data.x is None:
            if hasattr(data, 'adj'):
                data.x = data.adj.sum(1).view(-1, 1)
            else:
                adj = to_scipy_sparse_matrix(data.edge_index).sum(1)
                data.x = torch.FloatTensor(adj.sum(1)).view(-1, 1)
        return data

class RemoveEdgeAttr(object):
    def __call__(self, data):
        if data.edge_attr is not None:
            data.edge_attr = None
        if data.x is None:
            if hasattr(data, 'adj'):
                data.x = data.adj.sum(1).view(-1, 1)
            else:
                adj = to_scipy_sparse_matrix(data.edge_index).sum(1)
                data.x = torch.FloatTensor(adj.sum(1)).view(-1, 1)

        data.x = data.x.float()
        return data

class ConcatPos(object):
    def __call__(self, data):
        if data.edge_attr is not None:
            data.edge_attr = None
        data.x = torch.cat([data.x, data.pos], dim=1)
        data.pos = None
        return data

class SparseTensorDataset(Dataset):
    def __init__(self, data):
        self.data  = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def accuracy(output, labels):
	preds = output.max(1)[1].type_as(labels)
	correct = preds.eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)

def accuracy_binary(output, labels):
	output = (output > 0.5).float() * 1
	correct = output.type_as(labels).eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)

def f1(output, labels):
	preds = output.max(1)[1].type_as(labels)
	f1 = f1_score(labels, preds, average='weighted')
	return f1

def create_split(split_file, ngraph, dataset_split):
    ntraining, nvalidation, ntest = [math.floor(ngraph*x) for x in dataset_split]
    all_idx = [str(x) for x in list(range(ngraph))]
    with open(split_file, 'w', encoding='utf-8') as fout:
        random.shuffle(all_idx)
        training_idx = all_idx[:ntraining]
        val_idx = all_idx[ntraining:ntraining+nvalidation]
        test_idx = all_idx[ntraining+nvalidation:]
        fout.write(' '.join(training_idx)+'\t')
        fout.write(' '.join(val_idx)+'\t')
        fout.write(' '.join(test_idx)+'\n')
        training_idx = [int(x) for x in training_idx]
        val_idx = [int(x) for x in val_idx]
        test_idx = [int(x) for x in test_idx]
        split = [training_idx, val_idx, test_idx]
    return split

def load_split(split_file):
    with open(split_file, 'r', encoding='utf-8') as fin:
        for i in fin:
            training_idx, val_idx, test_idx = i.strip().split('\t')
            training_idx = [int(x) for x in training_idx.split(' ')]
            val_idx = [int(x) for x in val_idx.split(' ')]
            test_idx = [int(x) for x in test_idx.split(' ')]
            split = [training_idx, val_idx, test_idx]
    return split

def to_torch_coo_tensor(
    edge_index,
    size,
    edge_attr = None,
):

    size = (size, size)

    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)
    size += edge_attr.size()[1:]
    out = torch.sparse_coo_tensor(edge_index, edge_attr, size,
                                  device=edge_index.device)
    out = out.coalesce()
    return out

def batch_generator(dataset, all_training_idxs, current_idx, batch_size):
    selected_data = [copy.deepcopy(dataset[i]) for i in all_training_idxs[current_idx:current_idx+batch_size]]
    combined_edge_index = [i["edge_index"] for i in selected_data]
    combined_edge_weight = [i["edge_weight"] for i in selected_data]
    combined_x = [i["x"] for i in selected_data]
    combined_y = [i["y"] for i in selected_data]
    batch = []

    for i in range(len(selected_data)):
        batch += [i]*selected_data[i]["x"].shape[0]

    # combined_edge_index = torch.cat(combined_edge_index, dim=1)
    combined_edge_weight = torch.cat(combined_edge_weight)
    combined_x = torch.cat(combined_x)
    combined_y = torch.stack(combined_y, dim=0)
    batch = torch.LongTensor(batch)

    accumulated_idx = 0
    for i in range(len(combined_edge_index)):
        combined_edge_index[i] += accumulated_idx
        accumulated_idx += selected_data[i]["x"].shape[0]
    combined_edge_index = torch.cat(combined_edge_index, dim=1)

    return combined_edge_index, combined_edge_weight, combined_x, combined_y, batch

def avg_num_node(name):
    dataset2avg_num_node = {'MNIST':71, 'CIFAR10':118, 'DD':285, 'MUTAG':18,
                        'NCI1':30, 'NCI109':30,'PROTEINS':39,
                        'ogbg-molhiv':26, 'ogbg-molbbbp':24, 'ogbg-molbace':34}
    if name in dataset2avg_num_node:
        return dataset2avg_num_node[name]
    else:
        print("Unknown avg_num_node of the dataset {}".format(name))
        sys.exit()
