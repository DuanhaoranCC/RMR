import numpy as np
import torch
import torch.nn.functional as F
from utils.load import load_acm, load_mag, load_freebase, make_sparse_eye, load_Aminer_Large, load_imdb, load_cite
from utils.params import set_params, acm_params, aminer_params, freebase_params, mag_params, imdb_params, cite_params
from utils.evaluate import evaluate
from model import Cross_View
from torch_geometric.loader import HGTLoader, NeighborLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from torch_geometric import seed_everything


# args = acm_params()
# args = aminer_params()
args = cite_params()
# args = imdb_params()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def main(space):
    seed_everything(0)
    if args.dataset == "acm":
        load_data = load_acm
    elif args.dataset == "aminer":
        load_data = load_Aminer_Large
    elif args.dataset == "imdb":
        load_data = load_imdb
    elif args.dataset == "cite":
        load_data = load_cite
    else:
        raise NotImplementedError
    space['alpha'] = 0.5
    data = load_data().to(device)

    model = Cross_View(data=data, hidden_dim=64, feat_drop=space['feat'], alpha=space['alpha'],
                       att_drop1=space['attr1'], att_drop2=space['attr2'], r1=space['r1'], r2=space['r2'],
                       r3=space['r3']
                       ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=space['lr'], weight_decay=space['w'])

    for epoch in range(int(space['epoch']) + 1):
        model.train()
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()

    model.eval()
    embeds = model.get_embed(data)
    for ratio in args.ratio:
        evaluate(
            embeds,
            ratio,
            data[data.main_node][f'{ratio}_train_mask'],
            data[data.main_node][f'{ratio}_val_mask'],
            data[data.main_node][f'{ratio}_test_mask'],
            data[data.main_node].y[:, 1].long(),
            device,
            data,
            0.01,
            0,
            args.dataset,
        )



if __name__ == '__main__':
    main({'attr1': 0.5, 'attr2': 0.0, 'epoch': 5000, 'feat': 0.0, 'lr': 1e-5,
          'r1': 0.7, 'r2': 0.5, 'r3': 0.1, 'w': 0.0, 'alpha': 0.5, 'acc': 0.3972})
