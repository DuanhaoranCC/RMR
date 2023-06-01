import torch
import numpy as np
import torch_geometric.transforms as T
import scipy.sparse as sp
from torch_geometric.datasets import IMDB, DBLP, OGB_MAG
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree, sort_edge_index, is_undirected, to_undirected
from torch_geometric.transforms import AddMetaPaths
# from data import renamed_load
import pickle
import json
from collections import Counter
import networkx as nx
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from transformers import BertModel, BertTokenizer


# def transfer(gloveFile, word2vecFile):
#     glove2word2vec(gloveFile, word2vecFile)


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def train_test_split(labels, seed, train_examples_per_class=None, val_examples_per_class=None,
                     test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(seed=0)
    train_indices, val_indices, test_indices = get_train_val_test_split(
        random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size,
        val_size, test_size)

    # print('number of training: {}'.format(len(train_indices)))
    # print('number of validation: {}'.format(len(val_indices)))
    # print('number of testing: {}'.format(len(test_indices)))

    train_mask = np.zeros((labels.shape[0], 1), dtype=int)
    train_mask[train_indices, 0] = 1
    train_mask = np.squeeze(train_mask, 1)
    val_mask = np.zeros((labels.shape[0], 1), dtype=int)
    val_mask[val_indices, 0] = 1
    val_mask = np.squeeze(val_mask, 1)
    test_mask = np.zeros((labels.shape[0], 1), dtype=int)
    test_mask[test_indices, 0] = 1
    test_mask = np.squeeze(test_mask, 1)
    mask = {}
    mask['train'] = train_mask
    mask['val'] = val_mask
    mask['test'] = test_mask
    return mask


def preprocess_sp_features(features):
    features = features.tocoo()
    row = torch.from_numpy(features.row)
    col = torch.from_numpy(features.col)
    e = torch.stack((row, col))
    v = torch.from_numpy(features.data)
    x = torch.sparse_coo_tensor(e, v, features.shape).to_dense()
    x.div_(x.sum(dim=-1, keepdim=True).clamp_(min=1.))
    return x


def preprocess_th_features(features):
    x = features.to_dense()
    x.div_(x.sum(dim=-1, keepdim=True).clamp_(min=1.))
    return x


def nei_to_edge_index(nei, reverse=False):
    edge_indexes = []

    for src, dst in enumerate(nei):
        src = torch.tensor([src], dtype=dst.dtype, device=dst.device)
        src = src.repeat(dst.shape[0])
        if reverse:
            edge_index = torch.stack((dst, src))
        else:
            edge_index = torch.stack((src, dst))

        edge_indexes.append(edge_index)

    return torch.cat(edge_indexes, dim=1)


def sp_feat_to_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def sp_adj_to_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return indices


def make_sparse_eye(N):
    e = torch.arange(N, dtype=torch.long)
    e = torch.stack([e, e])
    o = torch.ones(N, dtype=torch.float32)
    return torch.sparse_coo_tensor(e, o, size=(N, N))


def make_sparse_tensor(x):
    row, col = torch.where(x == 1)
    e = torch.stack([row, col])
    o = torch.ones(e.shape[1], dtype=torch.float32)
    return torch.sparse_coo_tensor(e, o, size=x.shape)


def load_acm():
    path = "./data/acm/"
    ratio = [1, 5, 10, 20]
    label = np.load(path + "labels.npy").astype('int32')
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    feat_p = sp.load_npz(path + "p_feat.npz").astype("float32")
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_s = make_sparse_eye(60)
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    pos = sp.load_npz(path + "pos.npz")

    # train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    # test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    # val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.LongTensor(label)
    nei_a = nei_to_edge_index([torch.LongTensor(i) for i in nei_a])
    nei_s = nei_to_edge_index([torch.LongTensor(i) for i in nei_s])
    feat_p = preprocess_sp_features(feat_p)
    feat_a = preprocess_sp_features(feat_a)
    feat_s = preprocess_th_features(feat_s)
    pap = sp_adj_to_tensor(pap)
    psp = sp_adj_to_tensor(psp)
    pos = sp_adj_to_tensor(pos)

    # train = [torch.LongTensor(i) for i in train]
    # val = [torch.LongTensor(i) for i in val]
    # test = [torch.LongTensor(i) for i in test]

    data = HeteroData()
    # mask = torch.tensor([False] * feat_p.shape[0])

    data['p'].x = feat_p
    data['a'].x = feat_a
    data['s'].x = feat_s
    data['p'].y = label
    data['num_nodes_dict'] = {
        'p': feat_p.shape[0],
        'a': feat_a.shape[0],
        's': 60
    }
    # for r, tr, va, te in zip(ratio, train, val, test):
    #     train_mask_l = f"{r}_train_mask"
    #     train_mask = mask.clone()
    #     train_mask[tr] = True
    #
    #     val_mask_l = f"{r}_val_mask"
    #     val_mask = mask.clone()
    #     val_mask[va] = True
    #
    #     test_mask_l = f"{r}_test_mask"
    #     test_mask = mask.clone()
    #     test_mask[te] = True
    #
    #     data['p'][train_mask_l] = train_mask
    #     data['p'][val_mask_l] = val_mask
    #     data['p'][test_mask_l] = test_mask
    for r in ratio:
        mask = train_test_split(
            data['p'].y.detach().cpu().numpy(), seed=np.random.randint(0, 35456, size=1),
            train_examples_per_class=r,
            val_size=1000, test_size=None)
        train_mask_l = f"{r}_train_mask"
        train_mask = mask['train'].astype(bool)
        val_mask_l = f"{r}_val_mask"
        val_mask = mask['val'].astype(bool)

        test_mask_l = f"{r}_test_mask"
        test_mask = mask['test'].astype(bool)

        data['p'][train_mask_l] = train_mask
        data['p'][val_mask_l] = val_mask
        data['p'][test_mask_l] = test_mask

    data[('a', 'p')].edge_index = nei_a.flip([0])
    data[('s', 'p')].edge_index = nei_s.flip([0])
    data[('p', 'a')].edge_index = nei_a
    data[('p', 's')].edge_index = nei_s
    # data[('p', 'a', 'p')].edge_index = pap
    # data[('p', 's', 'p')].edge_index = psp
    # data[('p', 'pos', 'p')].edge_index = pos

    metapath_dict = {
        ('p', 'a', 'p'): None,
        ('p', 's', 'p'): None
    }

    schema_dict = {
        ('a', 'p'): None,
        ('s', 'p'): None
    }
    data['mp'] = {
        ('p', 'a'): None,
        ('p', 's'): None
    }
    data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    data['main_node'] = 'p'
    data['use_nodes'] = ('p', 'a', 's')

    return data


def load_aminer():
    ratio = [20, 40, 60]
    path = "./data/aminer_small/"

    label = np.load(path + "labels.npy").astype('int32')
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_r = np.load(path + "nei_r.npy", allow_pickle=True)
    # Because none of P, A or R has features, we assign one-hot encodings to all of them.
    feat_p = make_sparse_eye(6564)
    feat_a = make_sparse_eye(13329)
    feat_r = make_sparse_eye(35890)
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.LongTensor(label)
    nei_a = nei_to_edge_index([torch.LongTensor(i) for i in nei_a])
    nei_r = nei_to_edge_index([torch.LongTensor(i) for i in nei_r])
    feat_p = preprocess_th_features(feat_p)
    feat_a = preprocess_th_features(feat_a)
    feat_r = preprocess_th_features(feat_r)
    pap = sp_adj_to_tensor(pap)
    prp = sp_adj_to_tensor(prp)
    pos = sp_adj_to_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    data = HeteroData()
    mask = torch.tensor([False] * feat_p.shape[0])

    data['p'].x = feat_p
    data['a'].x = feat_a
    data['r'].x = feat_r
    data['p'].y = label

    for r, tr, va, te in zip(ratio, train, val, test):
        train_mask_l = f"{r}_train_mask"
        train_mask = mask.clone()
        train_mask[tr] = True

        val_mask_l = f"{r}_val_mask"
        val_mask = mask.clone()
        val_mask[va] = True

        test_mask_l = f"{r}_test_mask"
        test_mask = mask.clone()
        test_mask[te] = True

        data['p'][train_mask_l] = train_mask
        data['p'][val_mask_l] = val_mask
        data['p'][test_mask_l] = test_mask

    data[('a', 'p')].edge_index = nei_a.flip([0])
    data[('r', 'p')].edge_index = nei_r.flip([0])
    data[('p', 'a')].edge_index = nei_a
    data[('p', 'r')].edge_index = nei_r
    # data[('p', 'a', 'p')].edge_index = pap
    # data[('p', 'r', 'p')].edge_index = prp
    # data[('p', 'pos', 'p')].edge_index = pos

    # metapath_dict = {
    #     ('p', 'a', 'p'): None,
    #     ('p', 'r', 'p'): None
    # }

    schema_dict = {
        ('a', 'p'): None,
        ('r', 'p'): None
    }
    data['mp'] = {
        ('p', 'a'): None,
        ('p', 'r'): None
    }
    # data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    data['main_node'] = 'p'
    data['use_nodes'] = ('p', 'a', 'r')

    return data


def load_freebase():
    ratio = [1, 5, 10, 20]
    path = "./data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_w = np.load(path + "nei_w.npy", allow_pickle=True)

    feat_m = torch.eye(3492)
    feat_d = torch.eye(2502)
    feat_a = torch.eye(33401)
    feat_w = torch.eye(4459)

    label = torch.LongTensor(label)
    nei_d = nei_to_edge_index([torch.LongTensor(i) for i in nei_d])
    nei_a = nei_to_edge_index([torch.LongTensor(i) for i in nei_a])
    nei_w = nei_to_edge_index([torch.LongTensor(i) for i in nei_w])

    data = HeteroData()

    data['m'].x = feat_m
    data['d'].x = feat_d
    data['a'].x = feat_a
    data['w'].x = feat_w
    data['m'].y = label

    # train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    # test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    # val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    # train = [torch.LongTensor(i) for i in train]
    # val = [torch.LongTensor(i) for i in val]
    # test = [torch.LongTensor(i) for i in test]
    # mask = torch.tensor([False] * feat_m.shape[0])
    # for r, tr, va, te in zip(ratio, train, val, test):
    #     train_mask_l = f"{r}_train_mask"
    #     train_mask = mask.clone()
    #     train_mask[tr] = True
    #
    #     val_mask_l = f"{r}_val_mask"
    #     val_mask = mask.clone()
    #     val_mask[va] = True
    #
    #     test_mask_l = f"{r}_test_mask"
    #     test_mask = mask.clone()
    #     test_mask[te] = True
    #
    #     data['m'][train_mask_l] = train_mask
    #     data['m'][val_mask_l] = val_mask
    #     data['m'][test_mask_l] = test_mask

    data[('d', 'm')].edge_index = nei_d.flip([0])
    data[('m', 'd')].edge_index = nei_d
    data[('a', 'm')].edge_index = nei_a.flip([0])
    data[('m', 'a')].edge_index = nei_a
    data[('w', 'm')].edge_index = nei_w.flip([0])
    data[('m', 'w')].edge_index = nei_w
    for r in ratio:
        mask = train_test_split(
            data['m'].y.detach().cpu().numpy(), seed=np.random.randint(0, 35456, size=1),
            train_examples_per_class=r,
            val_size=1000, test_size=None)
        train_mask_l = f"{r}_train_mask"
        train_mask = mask['train'].astype(bool)
        val_mask_l = f"{r}_val_mask"
        val_mask = mask['val'].astype(bool)

        test_mask_l = f"{r}_test_mask"
        test_mask = mask['test'].astype(bool)

        data['m'][train_mask_l] = train_mask
        data['m'][val_mask_l] = val_mask
        data['m'][test_mask_l] = test_mask

    schema_dict = {
        ('a', 'm'): None,
        ('d', 'm'): None,
        ('w', 'm'): None,
    }
    data['mp'] = {
        ('m', 'a'): None,
        ('m', 'd'): None,
        ('m', 'w'): None
    }
    data['schema_dict1'] = {
        ('a', 'm'): None,
        ('d', 'm'): None
    }
    data['schema_dict2'] = {
        ('a', 'm'): None,
        ('w', 'm'): None,
    }
    data['schema_dict3'] = {
        ('d', 'm'): None,
        ('w', 'm'): None,
    }
    data['schema_dict'] = schema_dict
    data['main_node'] = 'm'
    data['use_nodes'] = ('m', 'a', 'd', 'w')

    return data


def load_imdb():
    ratio = [1, 5, 10, 20]
    data = IMDB(root='./data/imdb/')[0]
    metapaths = [[("movie", "director"), ("director", "movie")],
                 [("movie", "actor"), ("actor", "movie")]]
    data = AddMetaPaths(metapaths)(data)

    # metapath_dict = {
    #     ('movie', 'metapath_0', 'movie'): None,
    #     ('movie', 'metapath_1', 'movie'): None
    # }
    # data['num_nodes_dict'] = {
    #     'movie': 4278,
    #     'actor': 5257,
    #     'director': 2081
    # }
    schema_dict = {
        ('actor', 'movie'): None,
        ('director', 'movie'): None
    }

    data['mp'] = {
        ('movie', 'actor'): None,
        ('movie', 'director'): None
    }

    # data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    data['main_node'] = 'movie'
    data['use_nodes'] = ('movie', 'actor', 'director')

    for r in ratio:
        mask = train_test_split(
            data['movie'].y.detach().cpu().numpy(), seed=np.random.randint(0, 35456, size=1),
            train_examples_per_class=r,
            val_size=1000, test_size=None)
        train_mask_l = f"{r}_train_mask"
        train_mask = mask['train'].astype(bool)
        val_mask_l = f"{r}_val_mask"
        val_mask = mask['val'].astype(bool)

        test_mask_l = f"{r}_test_mask"
        test_mask = mask['test'].astype(bool)

        data['movie'][train_mask_l] = train_mask
        data['movie'][val_mask_l] = val_mask
        data['movie'][test_mask_l] = test_mask
    return data


# def load_dblp():
#     ratio = [20, 40, 60]
#     data = DBLP(root='./data/dblp2/')[0]
#     metapaths = [[("author", "paper"), ("paper", "author")],
#                  [("author", "paper"), ("paper", "conference"), ("conference", "paper"), ("paper", "author")],
#                  [("author", "paper"), ("paper", "term"), ("term", "paper"), ("paper", "author")]]
#     data = AddMetaPaths(metapaths)(data)
#
#     metapath_dict = {
#         ('author', 'metapath_0', 'author'): None,
#         ('author', 'metapath_1', 'author'): None,
#         ('author', 'metapath_2', 'author'): None
#     }
#
#     schema_dict = {
#         ('paper', 'author'): None,
#         ('paper', 'term'): None,
#         ('paper', 'conference'): None
#     }
#
#     data['metapath_dict'] = metapath_dict
#     data['schema_dict'] = schema_dict
#     data['main_node'] = 'author'
#     data['use_nodes'] = ('author', 'paper', 'conference', 'term')
#
#     for r in ratio:
#         mask = train_test_split(
#             data['movie'].y.detach().cpu().numpy(), seed=np.random.randint(0, 35456, size=1),
#             train_examples_per_class=r,
#             val_size=1000, test_size=None)
#         train_mask_l = f"{r}_train_mask"
#         train_mask = mask['train'].astype(bool)
#         val_mask_l = f"{r}_val_mask"
#         val_mask = mask['val'].astype(bool)
#
#         test_mask_l = f"{r}_test_mask"
#         test_mask = mask['test'].astype(bool)
#
#         data['movie'][train_mask_l] = train_mask
#         data['movie'][val_mask_l] = val_mask
#         data['movie'][test_mask_l] = test_mask
#     return data

def load_dblp():
    path = "./data/dblp/"
    ratio = [20, 40, 60]
    label = np.load(path + "labels.npy").astype('int32')
    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.load_npz(path + "p_feat.npz").astype("float32")
    feat_t = np.load(path + "t_feat.npz").astype("float32")
    pc = np.genfromtxt(path + "pc.txt")
    pt = np.genfromtxt(path + "pt.txt")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.LongTensor(label)
    nei_p = nei_to_edge_index([torch.LongTensor(i) for i in nei_p], True)
    feat_p = preprocess_sp_features(feat_p)
    feat_a = preprocess_sp_features(feat_a)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    data = HeteroData()
    mask = torch.tensor([False] * feat_a.shape[0])
    data['a'].x = feat_a
    data['a'].y = label

    for r, tr, va, te in zip(ratio, train, val, test):
        train_mask_l = f"{r}_train_mask"
        train_mask = mask.clone()
        train_mask[tr] = True

        val_mask_l = f"{r}_val_mask"
        val_mask = mask.clone()
        val_mask[va] = True

        test_mask_l = f"{r}_test_mask"
        test_mask = mask.clone()
        test_mask[te] = True

        data['a'][train_mask_l] = train_mask
        data['a'][val_mask_l] = val_mask
        data['a'][test_mask_l] = test_mask

    data['p'].x = feat_p
    data['t'].x = torch.FloatTensor(feat_t)
    data['c'].x = make_sparse_eye(20)
    data[('p', 't')].edge_index = torch.tensor(pt, dtype=torch.long).t().contiguous()
    data[('p', 'c')].edge_index = torch.tensor(pc, dtype=torch.long).t().contiguous()
    data[('p', 'a')].edge_index = nei_p

    data[('t', 'p')].edge_index = data[('p', 't')].edge_index.flip(0)
    data[('c', 'p')].edge_index = data[('p', 'c')].edge_index.flip(0)
    data[('a', 'p')].edge_index = data[('p', 'a')].edge_index.flip(0)

    # metapath_dict = {
    #     ('a', 'p', 'a'): None,
    #     ('a', 'pcp', 'a'): None,
    #     ('a', 'ptp', 'a'): None
    # }

    schema_dict = {
        ('c', 'p'): None,
        ('p', 'a'): None,
        ('t', 'p'): None,
    }
    data['mp'] = {
        ('p', 'a'): None,
        ('p', 't'): None,
        ('p', 'c'): None,
    }
    # data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    data['main_node'] = 'a'
    data['use_nodes'] = ('p', 'a', 't', 'c')
    data['relation1'] = (['a', 'p', 't'],
                         [('p', 'to', 't'),
                          ('p', 'to', 'a'),
                          ('t', 'to', 'p'),
                          ('a', 'to', 'p')])
    data['relation2'] = (['a', 'p', 't', 'c'],
                         [('p', 'to', 't'),
                          ('p', 'to', 'c'),
                          ('p', 'to', 'a'),
                          ('t', 'to', 'p'),
                          ('c', 'to', 'p'),
                          ('a', 'to', 'p')])

    return data


def load_yelp():
    path = "./data/yelp/"
    ratio = [20, 40, 60]
    data = HeteroData()
    label = np.loadtxt(path + "true_cluster.txt").astype("int64")
    feat = np.loadtxt(path + "attributes.txt").astype("float32")
    RB = np.loadtxt(path + "RB.txt").astype("int64")
    RK = np.loadtxt(path + "RK.txt").astype("int64")
    RU = np.loadtxt(path + "RU.txt").astype("int64")
    BRKRB = np.loadtxt(path + "BRKRB.txt").astype("int64")
    BRURB = np.loadtxt(path + "BRURB.txt").astype("int64")
    data[('R', 'B')].edge_index = torch.from_numpy(RB).t().contiguous()
    data[('R', 'K')].edge_index = torch.from_numpy(RK).t().contiguous()
    data[('R', 'U')].edge_index = torch.from_numpy(RU).t().contiguous()
    data[('B', 'BRKRB', 'B')].edge_index = torch.from_numpy(BRKRB).t().contiguous()
    data[('B', 'BRURB', 'B')].edge_index = torch.from_numpy(BRURB).t().contiguous()
    data['B'].y = torch.from_numpy(label)
    data['B'].x = torch.from_numpy(feat)
    data['R'].x = torch.eye(33360)
    data['U'].x = torch.eye(1286)
    data['K'].x = torch.eye(82)
    schema_dict = {
        ('R', 'B'): None
    }
    metapath_dict = {
        ('B', 'BRKRB', 'B'): None,
        ('B', 'BRURB', 'B'): None
    }
    data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    data['main_node'] = 'B'
    data['use_nodes'] = ('B', 'R')

    for r in ratio:
        mask = train_test_split(
            data['B'].y.detach().cpu().numpy(), seed=np.random.randint(0, 35456, size=1),
            train_examples_per_class=r,
            val_size=1000, test_size=None)
        train_mask_l = f"{r}_train_mask"
        train_mask = mask['train'].astype(bool)
        val_mask_l = f"{r}_val_mask"
        val_mask = mask['val'].astype(bool)

        test_mask_l = f"{r}_test_mask"
        test_mask = mask['test'].astype(bool)

        data['B'][train_mask_l] = train_mask
        data['B'][val_mask_l] = val_mask
        data['B'][test_mask_l] = test_mask
    return data


def load_Aminer_Large():
    path = "./data/Aminer/"
    ratio = [1, 5, 10, 20]
    data = HeteroData()
    label = np.loadtxt(path + "paper_label.txt").astype("int64")

    # for i in range(127623):
    #     if i not in label[:, 0]:
    #         print(i)
    PA = np.loadtxt(path + "paper_author.txt").astype("int64")
    PC = np.loadtxt(path + "paper_conference.txt").astype("int64")
    PR = np.loadtxt(path + "paper_type.txt").astype("int64")

    # for i in range(127623):
    #     if i not in label[:, 0]:
    #         # print(i)
    #         # np.nonzero(np.delete(PA, np.nonzero(PA[:, 0] == i), axis=0)[:, 0] == i)
    #         # label = np.delete(label, np.nonzero(label[:, 0] == i), axis=0)
    #         PA = np.delete(PA, np.nonzero(PA[:, 0] == i), axis=0)
    #         PC = np.delete(PC, np.nonzero(PC[:, 0] == i), axis=0)
    #         PR = np.delete(PR, np.nonzero(PR[:, 0] == i), axis=0)
    data[('P', 'A')].edge_index = torch.from_numpy(PA).t().contiguous()
    data[('A', 'P')].edge_index = torch.from_numpy(PA).t().contiguous()[[1, 0]]
    data[('P', 'C')].edge_index = torch.from_numpy(PC).t().contiguous()
    data[('C', 'P')].edge_index = torch.from_numpy(PC).t().contiguous()[[1, 0]]
    data[('P', 'R')].edge_index = torch.from_numpy(PR).t().contiguous()
    data[('R', 'P')].edge_index = torch.from_numpy(PR).t().contiguous()[[1, 0]]
    data['P'].y = torch.from_numpy(label[:, -1])
    data['P'].x = make_sparse_eye(127623)
    data['A'].x = make_sparse_eye(164473)
    data['R'].x = make_sparse_eye(147251)
    data['C'].x = make_sparse_eye(101)
    schema_dict = {
        ('A', 'P'): None,
        ('R', 'P'): None,
        ('C', 'P'): None
    }
    # metapath_dict = {
    #     ('P', 'PAP', 'P'): None,
    #     ('P', 'PRP', 'P'): None
    # }
    # metapaths = [[("P", "A"), ("A", "P")],
    #              [("P", "R"), ("R", "P")],
    #              [("P", "C"), ("C", "P")]]
    # data = AddMetaPaths(metapaths)(data)
    # data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    data['schema_dict1'] = {
        ('A', 'P'): None,
        ('R', 'P'): None
    }
    data['schema_dict2'] = {
        ('A', 'P'): None,
        ('C', 'P'): None
    }
    data['schema_dict3'] = {
        ('R', 'P'): None,
        ('C', 'P'): None
    }
    data['main_node'] = 'P'
    data['use_nodes'] = ('P', 'A', 'R', 'C')
    data['label'] = torch.from_numpy(label)
    data['mp'] = {
        ('P', 'A'): None,
        ('P', 'R'): None,
        ('P', 'C'): None
    }
    data['num_nodes_dict'] = {
        'P': 127623,
        'R': 147251,
        'C': 101,
        'A': 164473
    }
    for r in ratio:
        mask = train_test_split(
            data['P'].y.detach().cpu().numpy(), seed=np.random.randint(0, 35456, size=1),
            train_examples_per_class=r,
            val_size=1000, test_size=None)
        train_mask_l = f"{r}_train_mask"
        train_mask = mask['train'].astype(bool)
        val_mask_l = f"{r}_val_mask"
        val_mask = mask['val'].astype(bool)

        test_mask_l = f"{r}_test_mask"
        test_mask = mask['test'].astype(bool)

        data['P'][train_mask_l] = train_mask
        data['P'][val_mask_l] = val_mask
        data['P'][test_mask_l] = test_mask
    return data


def load_mag():
    ratio = [1, 5, 10, 20]
    # transform = T.ToUndirected(merge=True)
    dataset = OGB_MAG(root='./data/MAG')
    data = dataset[0]
    data['author'].x = make_sparse_eye(1134649)
    data['R'].x = make_sparse_eye(736389)
    data['field_of_study'].x = make_sparse_eye(59965)
    schema_dict = {
        ('author', 'writes', 'paper'): None,
        ('field_of_study', 'rev_has_topic', 'paper'): None,
        ('R', 'rev_cite', 'paper'): None
    }
    # data[('paper', 'cite', 'R')].edge_index = data[('paper', 'cites', 'paper')].edge_index
    data['mp'] = {
        # ('institution', 'rev_affiliated_with', 'author'): None,
        ('paper', 'has_topic', 'field_of_study'): None,
        ('paper', 'rev_writes', 'author'): None,
        ('paper', 'cite', 'R'): None,
    }
    # data[('institution', 'rev_affiliated_with', 'author')].edge_index = \
    #     data[('author', 'affiliated_with', 'institution')].edge_index[[1, 0]]
    # data[('paper', 'rev_cites', 'paper')].edge_index = data[('paper', 'cites', 'paper')].edge_index[[1, 0]]
    # data[('paper', 'rev_writes', 'author')].edge_index = \
    #     data[('author', 'writes', 'paper')].edge_index[[1, 0]]
    data[('R', 'rev_cite', 'paper')].edge_index = data[('paper', 'cites', 'paper')].edge_index
    data[('field_of_study', 'rev_has_topic', 'paper')].edge_index = \
        data[('paper', 'has_topic', 'field_of_study')].edge_index[[1, 0]]
    data['schema_dict'] = schema_dict
    data['schema_dict1'] = {
        ('author', 'writes', 'paper'): None,
        ('field_of_study', 'rev_has_topic', 'paper'): None
    }
    data['schema_dict2'] = {
        ('author', 'writes', 'paper'): None,
        ('R', 'rev_cite', 'paper'): None
    }
    data['schema_dict3'] = {
        ('field_of_study', 'rev_has_topic', 'paper'): None,
        ('R', 'rev_cite', 'paper'): None
    }
    data['main_node'] = 'paper'
    data['use_nodes'] = ('paper', 'field_of_study', 'author', 'R')
    del data['institution']
    del data['affiliated_with']
    del data['rev_affiliated_with']
    del data['cites']
    # for r in ratio:
    #     mask = train_test_split(
    #         data['paper'].y.detach().cpu().numpy(), seed=np.random.randint(0, 35456, size=1),
    #         train_examples_per_class=r,
    #         val_size=1000, test_size=None)
    #     train_mask_l = f"{r}_train_mask"
    #     train_mask = mask['train'].astype(bool)
    #     val_mask_l = f"{r}_val_mask"
    #     val_mask = mask['val'].astype(bool)
    #
    #     test_mask_l = f"{r}_test_mask"
    #     test_mask = mask['test'].astype(bool)
    #
    #     data['paper'][train_mask_l] = train_mask
    #     data['paper'][val_mask_l] = val_mask
    #     data['paper'][test_mask_l] = test_mask
    return data


# def load_cs():
#     # graph = renamed_load(open("../hgt/graph_CS.pk", "rb"))
#     # ntype = "paper"
#     # feature = np.array(list(graph.node_feature[ntype]["emb"]))
#     # with open(f"{ntype}.npy", "wb") as f:
#     #     np.save(f, feature)
#     #
#     # edge_list = graph.edge_list
#     # edges = {}
#     #
#     # train_idx = []
#     # train_label = []
#     # val_idx = []
#     # val_label = []
#     # test_idx = []
#     # test_label = []
#     #
#     # # loop over all journal papers to create train, val, test split
#     # cand_list = list(graph.edge_list["venue"]["paper"]["PV_Journal"].keys())
#     #
#     # for paper_id in graph.edge_list["paper"]["venue"]["rev_PV_Journal"]:
#     #     for venue_id in graph.edge_list["paper"]["venue"]["rev_PV_Journal"][paper_id]:
#     #         _time = graph.edge_list["paper"]["venue"]["rev_PV_Journal"][paper_id][venue_id]
#     #         assert _time is not None
#     #         if _time < 2015:
#     #             if paper_id not in train_idx:
#     #                 train_idx.append(paper_id)
#     #                 train_label.append(cand_list.index(venue_id))
#     #         elif _time <= 2016:
#     #             if paper_id not in val_idx:
#     #                 val_idx.append(paper_id)
#     #                 val_label.append(cand_list.index(venue_id))
#     #         else:
#     #             if paper_id not in test_idx:
#     #                 test_idx.append(paper_id)
#     #                 test_label.append(cand_list.index(venue_id))
#     #
#     # num_papers = len(graph.node_feature["paper"])
#     # labels = np.zeros(num_papers, dtype=np.long) - 1  # init to -1
#     # labels[train_idx] = train_label
#     # labels[val_idx] = val_label
#     # labels[test_idx] = test_label
#     #
#     # node_types = set()
#     #
#     # for dtype in edge_list:
#     #     for stype in edge_list[dtype]:
#     #         for rel in edge_list[dtype][stype]:
#     #             if rel != "PV_Journal" and not rel.startswith("rev_"):
#     #                 print(stype, dtype, rel)
#     #                 src_list = []
#     #                 dst_list = []
#     #                 for dst in edge_list[dtype][stype][rel]:
#     #                     for src in edge_list[dtype][stype][rel][dst]:
#     #                         src_list.append(src)
#     #                         dst_list.append(dst)
#     #                 src_list = np.array(src_list)
#     #                 dst_list = np.array(dst_list)
#     #                 edges[(stype, rel, dtype)] = (src_list, dst_list)
#     #                 node_types.add(stype)
#     #                 node_types.add(dtype)
#     #
#     # n_classes = len(cand_list)
#     #
#     # graph = {
#     #     "edges": edges,
#     #     "labels": labels,
#     #     "split": [train_idx, val_idx, test_idx],
#     #     "n_classes": n_classes,
#     # }
#     # with open(f"graph_venue.npy", "wb") as f:
#     #     np.save(f, graph)
#     # pickle.dump(graph, open("graph_venue.pk", "wb"))
#     with open("./data/OAG/paper.npy", "rb") as f:
#         paper_feat = torch.from_numpy(np.load(f)).float()
#     with open("./data/OAG/graph_venue.pk", "rb") as f:
#         dataset = pickle.load(f)
#     data = HeteroData()
#     data[('paper', 'PV_Conference', 'venue')].edge_index = torch.cat(
#         [torch.from_numpy(dataset['edges'][('paper', 'PV_Conference', 'venue')][0]).unsqueeze(0),
#          torch.from_numpy(dataset['edges'][('paper', 'PV_Conference', 'venue')][1]).unsqueeze(0)])
#     data[('venue', 'Conference', 'paper')].edge_index = torch.cat(
#         [torch.from_numpy(dataset['edges'][('paper', 'PV_Conference', 'venue')][0]).unsqueeze(0),
#          torch.from_numpy(dataset['edges'][('paper', 'PV_Conference', 'venue')][1]).unsqueeze(0)])[[1, 0]]
#     # data[('paper', 'PP_cite', 'paper')].edge_index = torch.cat(
#     #     [torch.from_numpy(dataset['edges'][('paper', 'PP_cite', 'paper')][0]).unsqueeze(0),
#     #      torch.from_numpy(dataset['edges'][('paper', 'PP_cite', 'paper')][1]).unsqueeze(0)])
#     data[('paper', 'cite', 'paper')].edge_index = torch.cat(
#         [torch.from_numpy(dataset['edges'][('paper', 'PP_cite', 'paper')][0]).unsqueeze(0),
#          torch.from_numpy(dataset['edges'][('paper', 'PP_cite', 'paper')][1]).unsqueeze(0)])[[1, 0]]
#     data[('author', 'AP_write_last', 'paper')].edge_index = torch.cat(
#         [torch.from_numpy(dataset['edges'][('author', 'AP_write_last', 'paper')][0]).unsqueeze(0),
#          torch.from_numpy(dataset['edges'][('author', 'AP_write_last', 'paper')][1]).unsqueeze(0)])
#     data[('paper', 'write_last', 'author')].edge_index = torch.cat(
#         [torch.from_numpy(dataset['edges'][('author', 'AP_write_last', 'paper')][0]).unsqueeze(0),
#          torch.from_numpy(dataset['edges'][('author', 'AP_write_last', 'paper')][1]).unsqueeze(0)])[[1, 0]]
#     data[('paper', 'PF_in_L0', 'field')].edge_index = torch.cat(
#         [torch.from_numpy(dataset['edges'][('paper', 'PF_in_L0', 'field')][0]).unsqueeze(0),
#          torch.from_numpy(dataset['edges'][('paper', 'PF_in_L0', 'field')][1]).unsqueeze(0)])
#     data[('field', 'in_L0', 'paper')].edge_index = torch.cat(
#         [torch.from_numpy(dataset['edges'][('paper', 'PF_in_L0', 'field')][0]).unsqueeze(0),
#          torch.from_numpy(dataset['edges'][('paper', 'PF_in_L0', 'field')][1]).unsqueeze(0)])[[1, 0]]
#     data['paper'].x = paper_feat
#     data['venue'].x = make_sparse_eye(6933)
#     data['author'].x = make_sparse_eye(510189)
#     data['field'].x = make_sparse_eye(45438)
#     data['paper'].y = torch.from_numpy(dataset['labels'] + 1)
#     print(Counter(dataset['labels']))
#     schema_dict = {
#         ('venue', 'Conference', 'paper'): None,
#         ('author', 'AP_write_last', 'paper'): None,
#         ('field', 'in_L0', 'paper'): None,
#         ('paper', 'cite', 'paper'): None
#     }
#
#     data['mp'] = {
#         ('paper', 'PV_Conference', 'venue'): None,
#         ('paper', 'write_last', 'author'): None,
#         ('paper', 'PF_in_L0', 'field'): None
#     }
#     data['schema_dict'] = schema_dict
#     data['schema_dict1'] = {
#         ('venue', 'Conference', 'paper'): None,
#         ('author', 'AP_write_last', 'paper'): None,
#         ('field', 'in_L0', 'paper'): None,
#     }
#     data['schema_dict2'] = {
#         ('author', 'AP_write_last', 'paper'): None,
#         ('field', 'in_L0', 'paper'): None,
#         ('paper', 'cite', 'paper'): None
#     }
#     data['schema_dict3'] = {
#         ('venue', 'Conference', 'paper'): None,
#         ('field', 'in_L0', 'paper'): None,
#         ('paper', 'cite', 'paper'): None
#     }
#     data['schema_dict4'] = {
#         ('venue', 'Conference', 'paper'): None,
#         ('author', 'AP_write_last', 'paper'): None,
#         ('paper', 'cite', 'paper'): None
#     }
#     data['main_node'] = 'paper'
#     data['use_nodes'] = ('paper', 'field', 'author', 'venue')
#     # ratio = [1]
#     # for r in ratio:
#     #     mask = train_test_split(
#     #         data['paper'].y.detach().cpu().numpy(), seed=np.random.randint(0, 35456, size=1),
#     #         train_examples_per_class=r,
#     #         val_size=1000, test_size=None)
#     #     train_mask_l = f"{r}_train_mask"
#     #     train_mask = mask['train'].astype(bool)
#     #     val_mask_l = f"{r}_val_mask"
#     #     val_mask = mask['val'].astype(bool)
#     #
#     #     test_mask_l = f"{r}_test_mask"
#     #     test_mask = mask['test'].astype(bool)
#     #
#     #     data['paper'][train_mask_l] = train_mask
#     #     data['paper'][val_mask_l] = val_mask
#     #     data['paper'][test_mask_l] = test_mask
#
#     train_nid, val_nid, test_nid = dataset["split"]
#     mask = torch.tensor([False] * paper_feat.shape[0])
#     train_mask = mask.clone()
#     train_mask[train_nid] = True
#
#     val_mask = mask.clone()
#     val_mask[val_nid] = True
#
#     test_mask = mask.clone()
#     test_mask[test_nid] = True
#
#     data['paper']['train_mask'] = train_mask
#     data['paper']['val_mask'] = val_mask
#     data['paper']['test_mask'] = test_mask
#     return data

def load_cite():
    """
    An example of paper:
    #*Some constructions of the join of fuzzy subgroups and certain lattices of fuzzy subgroups with sup property
    #@Naseem Ajmal,Aparna Jain
    #t2009
    #cInformation Sciences: an International Journal
    #index498474
    #%81430
    #%211532
    #%225669
    #%296731
    #%317844
    #%417229
    #%583404
    #%582809
    #%596177
    #!In this paper, some new lattices of fuzzy substructures are constructed. For a given fuzzy set @m in a group G, a fuzzy subgroup S(@m) generated by @m is defined which helps to establish that the set L"s of all fuzzy subgroups with sup property constitutes a lattice. Consequently, many other sublattices of the lattice L of all fuzzy subgroups of G like L"s"""

    class Data_Reformat:
        def __init__(self, text_file='./data/outputacm.txt'):
            self.f = open(text_file, encoding='utf-8')
            self.num_paper = None
            # Create databases here
            self.papers = []  # list of list of strings with paper information
            self.authorship = []
            self.citation = []
            # self.paper_info = []
            self.abstract = []
            self.author_paper = []
            self.paper_conf = []
            self.year = []
            # self.labels = []
            self.model = KeyedVectors.load_word2vec_format('./data/glove.6B.100ff.txt', binary=False)
            with open("./data/author_type.pk", "rb") as f:
                self.author = pickle.load(f)
            with open("./data/confer_type.pk", "rb") as f:
                self.conf = pickle.load(f)
            with open("./data/d_s.pk", "rb") as f:
                self.d_s = pickle.load(f)

        def separate_paper(self):
            line = self.f.readline()  # include newline
            line_index = 1
            single_paper = []

            while line:
                line = line.rstrip()  # strip trailing spaces and newline

                if line_index == 1:
                    self.num_paper = int(line)
                else:
                    if line != "":
                        single_paper.append(line)
                    else:
                        self.papers.append(single_paper)
                        single_paper = []

                line = self.f.readline()
                line_index += 1

        def fill_table(self):
            assert self.papers != [], f"You forget to run separate_papers(): {self.papers}"
            assert self.num_paper == len(self.papers)
            for paper in self.papers:
                assert paper[0][0:2] == "#*", f"oh no! title doesn't start with #*: {paper[0][0:2]}"
                assert paper[1][0:2] == "#@", f"on no! author doesn't start with #@:{paper[1][0:2]}"
                assert paper[2][0:2] == "#t"
                assert paper[3][0:2] == "#c"
                assert paper[4][0:6] == "#index"
                title = paper[0][2:]
                index = int(paper[4][6:])
                # print(index)
                # word_vectors = [self.model[word] for word in title.split() if word in self.model]
                # if np.array(word_vectors).size == 0:
                #     continue
                # self.abstract.append(np.mean(word_vectors, axis=0))
                author_list = paper[1][2:].split(",")
                author_list = [x.strip() for x in author_list]

                author_list_filtered = filter(lambda x: x != "", author_list)
                author_list_filtered = filter(lambda x: x != "Jr.", author_list_filtered)
                author_list_filtered = filter(lambda x: x != "III", author_list_filtered)
                author_list_filtered = filter(lambda x: x != "II", author_list_filtered)
                year = int(paper[2][2:])
                conference = paper[3][2:]  # might be empty string ""
                self.year.append(year)
                # if conference not in self.conf:
                #     self.conf[conference] = len(self.conf.keys())
                if conference != "" and self.conf[conference] - 1 not in np.array(self.d_s[-8000:])[0]:
                    self.paper_conf.append([index, self.conf[conference] - 1])
                # for a in author_list_filtered:
                #     if a not in self.author:
                #         self.author[a] = len(self.author.keys())
                # self.paper_info.append([index, title, year, conference])

                for author in author_list_filtered:
                    # self.authorship.append([index, author])
                    self.author_paper.append([self.author[author], index])
                # transfer('../data/glove.6B.100d.txt', '../data/glove.6B.100ff.txt')
                # if np.array(self.abstract).shape[0] == 39579:
                #     print("KKKKKKKKK")
                paper_info_length = len(paper)
                if paper_info_length > 5:
                    for i in range(5, paper_info_length):
                        if paper[i][0:2] == "#!":  # abstract
                            self.abstract.append([index, paper[i][2:]])
                            # word_vectors1 = [self.model[word] for word in paper[i][2:].split() if word in self.model]
                            # if np.array(word_vectors1).size != 0:
                            #     self.abstract.append([np.mean(word_vectors1, axis=0)])
                            # else:
                            #     self.abstract.append([np.mean(word_vectors, axis=0)])
                            # print(np.array(self.abstract).shape)
                        elif paper[i][0:2] == "#%":  # citation
                            self.citation.append([index, int(paper[i][2:])])
                        else:
                            # assert self.authorship != []
                            assert self.citation != []
                            # assert self.paper_info != []
                            assert self.abstract != []
                # else:
                #     self.abstract.append([np.mean(word_vectors, axis=0)])
                #     print(np.array(self.abstract).shape)
            data = HeteroData()
            data['P'].x = make_sparse_eye(629814)
            data['R'].x = make_sparse_eye(629814)
            data[('A', 'write', 'P')].edge_index = torch.from_numpy(np.array(self.author_paper)).t().contiguous()
            data[('P', 'cite', 'R')].edge_index = torch.from_numpy(np.array(self.citation)).t().contiguous()
            data['P'].y = torch.from_numpy(np.array(self.paper_conf))
            data['A'].x = make_sparse_eye(595737)
            data[('P', 'rev_write', 'A')].edge_index = data[('A', 'write', 'P')].edge_index[[1, 0]]
            data[('R', 'rev_cite', 'P')].edge_index = data[('P', 'cite', 'R')].edge_index[[1, 0]]
            label = []
            label_dict = {}
            for index, i in enumerate(self.d_s[:100]):
                label_dict[i[0]] = index
            for index, i in enumerate(data['P'].y[:, 1]):
                if i.numpy() in np.array(self.d_s[:100])[:, 0]:
                    label.append([data['P'].y.numpy()[index][0], label_dict[i.numpy().tolist()]])

            data['P'].y = torch.from_numpy(np.array(label))
            # d = Counter(data['P'].y[:, 1].cpu().numpy())
            # d_s = sorted(d.items(), key=lambda x: x[1], reverse=True)
            ratio = [1, 5, 10, 20]
            for r in ratio:
                mask = train_test_split(
                    data['P'].y[:, 1].cpu().numpy(), seed=np.random.randint(0, 35456, size=1),
                    train_examples_per_class=r,
                    val_size=200, test_size=None)
                train_mask_l = f"{r}_train_mask"
                train_mask = mask['train'].astype(bool)
                val_mask_l = f"{r}_val_mask"
                val_mask = mask['val'].astype(bool)

                test_mask_l = f"{r}_test_mask"
                test_mask = mask['test'].astype(bool)

                data['P'][train_mask_l] = train_mask
                data['P'][val_mask_l] = val_mask
                data['P'][test_mask_l] = test_mask
            schema_dict = {
                ('A', 'write', 'P'): None,
                ('R', 'rev_cite', 'P'): None,
            }
            data['mp'] = {
                ('P', 'rev_write', 'A'): None,
                ('P', 'cite', 'R'): None,
            }
            data['schema_dict'] = schema_dict

            data['main_node'] = 'P'
            data['use_nodes'] = ('P', 'A', 'R')
            print(data)
            pickle.dump(data, open("data.pk", "wb"))

            return data

    # acm_reformater = Data_Reformat()
    # acm_reformater.separate_paper()
    # data = acm_reformater.fill_table()
    with open("./data/cite.pk", "rb") as f:
        data = pickle.load(f)
    data['P'].x = torch.squeeze(data['P'].x)
    ratio = [1, 5, 10, 20]
    for r in ratio:
        mask = train_test_split(
            data[data.main_node].y[:, 1].long().cpu().numpy(), seed=np.random.randint(0, 35456, size=1),
            train_examples_per_class=r,
            val_size=500, test_size=None)
        train_mask_l = f"{r}_train_mask"
        train_mask = mask['train'].astype(bool)
        val_mask_l = f"{r}_val_mask"
        val_mask = mask['val'].astype(bool)

        test_mask_l = f"{r}_test_mask"
        test_mask = mask['test'].astype(bool)

        data['P'][train_mask_l] = train_mask
        data['P'][val_mask_l] = val_mask
        data['P'][test_mask_l] = test_mask
    return data


# path = "../data/PGB/pgb_0.jsonl"
# file = open(path, 'r', encoding='utf-8')
# model = KeyedVectors.load_word2vec_format('../data/glove.6B.100ff.txt', binary=False)
# term = {}
# data = []
# cite = {}
# author = {}
# conf = {}
# for i in file.readlines():
#     da = json.loads(i)
#     data.append(da)
#     # if da["mesh"] != []:
#     #     for ter in da["mesh"]:
#     #         term[ter['term']] = len(term.keys())
#     if da["mesh"] != [] and da["mesh"][0]['term'] not in term:
#         term[da["mesh"][0]['term']] = len(term.keys())
#     if da["outbound_citations"] != []:
#         for ci in da["outbound_citations"]:
#             cite[int(ci)] = len(cite.keys())
#     for au in da["authors"]:
#         if au["middle"] != []:
#             author[au["first"].strip() + au["middle"][0].strip() + au["last"].strip()] = len(author.keys())
#         else:
#             author[au["first"].strip() + au["last"].strip()] = len(author.keys())
#     conf[da["journal"]] = len(conf.keys())
#
# ll = [85, 6, 15, 148, 100, 64, 28]
#
#
# def get_keys_by_value(dictionary, value):
#     keys = []
#     for k, v in dictionary.items():
#         if v == value:
#             keys.append(k)
#     return keys
#
#
# paper_ref = []
# paper_conf = []
# paper_label = []
# paper_author = []
# abstract = []
# year = []
# label = {}
# for index, d in enumerate(ll):
#     label[d] = index
# index = 0
# # with open("./data.pk", "rb") as f:
# #     data1 = pickle.load(f)
# for ind, paper in enumerate(data):
#     if paper["abstract"] == None:
#         ab = paper["title"]
#     else:
#         ab = paper["abstract"]
#     word_vectors = [model[word] for word in ab.split() if word in model]
#     print(np.array(abstract).shape)
#     if paper["authors"] == [] or paper["mesh"] == [] \
#             or paper["year"] == None or paper["title"] == None or \
#             paper["abstract"] == None or word_vectors == []:
#         continue
#     abstract.append([np.mean(word_vectors, axis=0)])
#     for ci in paper["outbound_citations"]:
#         paper_ref.append([index, cite[int(ci)]])
#     # l = []
#     # for ter in paper["mesh"]:
#     #     l.append(term[ter['term']])
#     # label = np.zeros(26740)
#     # label[l] = 1
#     # paper_label.append(label)
#     if term[paper["mesh"][0]['term']] in ll:
#         paper_label.append([index, label[term[paper["mesh"][0]['term']]]])
#         year.append([index, paper["year"]])
#     paper_label.append([index, term[paper["mesh"][0]['term']]])
#     paper_conf.append([index, conf[paper["journal"]]])
#     for auth in paper["authors"]:
#         if auth["middle"] != []:
#             paper_author.append(
#                 [index, author[auth["first"].strip() + auth["middle"][0].strip() + auth["last"].strip()]])
#         else:
#             paper_author.append([index, author[auth["first"].strip() + auth["last"].strip()]])
#     index += 1
# d = Counter(np.array(paper_label)[:, 1])
# d_s = sorted(d.items(), key=lambda x: x[1], reverse=True)
# # ll = []
# # for i in d_s[:10]:
# #     ll.append(i[0])
# data = HeteroData()
# data['P'].x = torch.from_numpy(np.array(abstract))
# # data['P'].x = data1['P'].x.squeeze()
# data['A'].x = make_sparse_eye(1034249)
# data['R'].x = make_sparse_eye(1559581)
# data['C'].x = make_sparse_eye(17443)
# data[('P', 'A')].edge_index = torch.from_numpy(np.array(paper_author)).t().contiguous().long()
# data[('P', 'R')].edge_index = torch.from_numpy(np.array(paper_ref)).t().contiguous().long()
# data[('P', 'C')].edge_index = torch.from_numpy(np.array(paper_conf)).t().contiguous().long()
# data[('A', 'P')].edge_index = data[('P', 'A')].edge_index[[1, 0]]
# data[('R', 'P')].edge_index = data[('P', 'R')].edge_index[[1, 0]]
# data[('C', 'P')].edge_index = data[('P', 'C')].edge_index[[1, 0]]
# data['P'].y = torch.from_numpy(np.array(paper_label))
# data['year'] = torch.from_numpy(np.array(year))
# n = data['P'].y.size(0)
# mask = torch.zeros(n)
# mask[torch.nonzero(data['year'][:, 1] <= 2017)] = 1
# print(torch.count_nonzero(mask == 1))
# data['P']["train"] = mask.bool()
# mask = torch.zeros(n)
# mask[torch.nonzero(data['year'][:, 1] == 2018)] = 1
# print(torch.count_nonzero(mask == 1))
# data['P']["val"] = mask.bool()
# mask = torch.zeros(n)
# mask[torch.nonzero(data['year'][:, 1] == 2019)] = 1
# print(torch.count_nonzero(mask == 1))
# data['P']["test"] = mask.bool()
# schema_dict = {
#     ('A', 'P'): None,
#     ('R', 'P'): None,
#     ('C', 'P'): None
# }
# data['schema_dict'] = schema_dict
# data['schema_dict1'] = {
#     ('A', 'P'): None,
#     ('R', 'P'): None
# }
# data['schema_dict2'] = {
#     ('A', 'P'): None,
#     ('C', 'P'): None
# }
# data['schema_dict3'] = {
#     ('R', 'P'): None,
#     ('C', 'P'): None
# }
# data['main_node'] = 'P'
# data['use_nodes'] = ('P', 'A', 'R', 'C')
# data['mp'] = {
#     ('P', 'A'): None,
#     ('P', 'R'): None,
#     ('P', 'C'): None
# }
#
# pickle.dump(data, open("cite1.pk", "wb"))
