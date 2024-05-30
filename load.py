import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import IMDB
from torch_geometric.data import HeteroData
import pickle


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

    label = torch.LongTensor(label)
    nei_a = nei_to_edge_index([torch.LongTensor(i) for i in nei_a])
    nei_s = nei_to_edge_index([torch.LongTensor(i) for i in nei_s])
    feat_p = preprocess_sp_features(feat_p)
    feat_a = preprocess_sp_features(feat_a)
    feat_s = preprocess_th_features(feat_s)

    data = HeteroData()
    data['p'].x = feat_p
    data['a'].x = feat_a
    data['s'].x = feat_s

    data['a', 'p'].edge_index = nei_a.flip([0])
    data['s', 'p'].edge_index = nei_s.flip([0])
    data['p', 'a'].edge_index = nei_a
    data['p', 's'].edge_index = nei_s

    data['p'].y = label
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
    data['schema_dict1'] = {
        # ('a', 'p'): None,
        ('s', 'p'): None
    }
    data['schema_dict2'] = {
        ('a', 'p'): None,
        # ('s', 'p'): None
    }
    data['main_node'] = 'p'
    data['use_nodes'] = ('p', 'a', 's')

    return data


def load_imdb():
    ratio = [1, 5, 10, 20]
    data = IMDB(root='./data/imdb/')[0]

    schema_dict = {
        ('actor', 'movie'): None,
        ('director', 'movie'): None
    }

    data['mp'] = {
        ('movie', 'actor'): None,
        ('movie', 'director'): None
    }
    data['schema_dict1'] = {
        # ('actor', 'movie'): None,
        ('director', 'movie'): None
    }
    data['schema_dict2'] = {
        ('actor', 'movie'): None,
        # ('director', 'movie'): None
    }

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


def load_Aminer_Large():
    path = "./data/Aminer/"
    ratio = [1, 5, 10, 20]
    data = HeteroData()
    label = np.loadtxt(path + "paper_label.txt").astype("int64")
    PA = np.loadtxt(path + "paper_author.txt").astype("int64")
    PC = np.loadtxt(path + "paper_conference.txt").astype("int64")
    PR = np.loadtxt(path + "paper_type.txt").astype("int64")

    data[('P', 'A')].edge_index = torch.from_numpy(PA).t().contiguous()
    data[('A', 'P')].edge_index = torch.from_numpy(PA).t().contiguous()[[1, 0]]
    data[('P', 'C')].edge_index = torch.from_numpy(PC).t().contiguous()
    data[('C', 'P')].edge_index = torch.from_numpy(PC).t().contiguous()[[1, 0]]
    data[('P', 'R')].edge_index = torch.from_numpy(PR).t().contiguous()
    data[('R', 'P')].edge_index = torch.from_numpy(PR).t().contiguous()[[1, 0]]
    data['P'].y = torch.from_numpy(label[:, -1])
    ################################################################################
    data['P'].x = make_sparse_eye(127623)
    data['A'].x = make_sparse_eye(164473)
    data['R'].x = make_sparse_eye(147251)
    data['C'].x = make_sparse_eye(101)
    schema_dict = {
        ('A', 'P'): None,
        ('R', 'P'): None,
        ('C', 'P'): None
    }
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


def sample_edge_index(sample_size, edge_index):
    # [2, num_edges] -> [num_edges, 2]. index 1 is dst.
    num_nodes = int(edge_index[1].max() + 1)
    e = edge_index.clone().T
    buc = torch.zeros(num_nodes, dtype=torch.long, device=e.device)
    r = torch.randperm(e.shape[0], dtype=torch.long, device=e.device)
    e = e[r]

    edge_list = []
    for edge in e:
        _, dst = edge
        if buc[dst] < sample_size:
            edge_list.append(edge)
            buc[dst] += 1

    edge_index = torch.stack(edge_list)
    return edge_index.T
