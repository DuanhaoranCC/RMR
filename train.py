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
    # data['P'].x = torch.squeeze(data['P'].x)
    # data['P'].x = make_sparse_eye(190792).to(device)
    # data['P'].x = make_sparse_eye(382019).to(device)
    # data['A'].x = make_sparse_eye(1846265).to(device)
    # data['C'].x = make_sparse_eye(20425).to(device)
    # data['R'].x = make_sparse_eye(2710499).to(device)
    # unique_labels, label_counts = np.unique(data[data.main_node].y[:, 1].long().cpu().numpy(), return_counts=True)
    #
    # plt.bar(unique_labels, label_counts)
    # plt.xlabel('Class')
    # plt.ylabel('Count')
    # plt.title('Label Distribution')
    # plt.show()
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

    # f1 = evaluate(
    #     embeds,
    #     20,
    #     data[data.main_node][f'{20}_train_mask'],
    #     data[data.main_node][f'{20}_val_mask'],
    #     data[data.main_node][f'{20}_test_mask'],
    #     data[data.main_node].y[:, 1].long(),
    #     device,
    #     data,
    #     0.01,
    #     0,
    #     args.dataset
    # )
    # f1 = evaluate(
    #     embeds,
    #     20,
    #     data[data.main_node].train,
    #     data[data.main_node].val,
    #     data[data.main_node].test,
    #     data[data.main_node].y[:, 1].long(),
    #     device,
    #     data,
    #     0.01,
    #     0,
    #     args.dataset
    # )


if __name__ == '__main__':
    # ACM
    # main({'alpha': 0.6, 'attr1': 0.0, 'attr2': 0.0, 'epoch': 200, 'feat': 0.8, 'lr': 0.005, 'r1': 0.9, 'r2': 0.3,
    #       'w': 0.0005, 'acc': 0.8867})
    # IMDB
    # main({'alpha': 0.5, 'attr1': 0.0, 'attr2': 0.3, 'epoch': 5500,
    #       'feat': 0.55, 'r1': 0.9, 'r2': 0.8, 'w': 5e-5, 'lr': 5e-5, 'acc': 0.5037})
    # Aminer
    # main({'alpha': 0.0, 'attr1': 0.85, 'attr2': 0.9, 'epoch': 400, 'feat': 0.0,
    #       'lr': 5e-5, 'r1': 0.75, 'r2': 0.85, 'r3': 0.0, 'r4': 0.9, 'r5': 0.5,
    #       'r6': 0.45, 'w': 0.001, 'acc': 0.725})
    # main({'alpha': 0.4, 'attr1': 0.0, 'attr2': 0.15, 'epoch': 4000, 'feat': 0.5,
    #       'lr': 5e-5, 'r1': 0.1, 'r2': 0.15, 'r3': 0.55, 'w': 1e-5, 'acc': 0.8853})
    # main({'alpha': 0.6, 'attr1': 0.25, 'attr2': 0.05, 'epoch': 3700,
    #       'feat': 0.0, 'lr': 5e-5, 'r1': 0.65, 'r2': 0.4, 'r3': 0.25,
    #       'w': 0.0005, 'acc': 0.122})
    # main({'alpha': 0.0, 'attr1': 0.9, 'attr2': 0.4, 'epoch': 4500,
    #       'feat': 0.6, 'lr': 5e-5, 'r1': 0.7, 'r2': 0.1, 'r3': 0.7,
    #       'w': 1e-3, 'acc': 0.4933})
    # s = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for i in s:
    #     main({'alpha': 0.0, 'attr1': 0.9, 'attr2': 0.3, 'epoch': 4500,
    #           'feat': 0.6, 'lr': 5e-5, 'r1': 0.7, 'r2': i, 'r3': 0.7,
    #           'w': 0.001, 'acc': 0.4998})
    # main({'attr1': 0.2, 'attr2': 0.15, 'epoch': 200, 'feat': 0.2,
    #       'lr': 1e-4, 'r1': 0.4, 'r2': 0.1, 'r3': 0.2, 'w': 1e-5})
    # main({'alpha': 0.5, 'attr1': 0.0, 'attr2': 0.15, 'epoch': 6000, 'feat': 0.5,
    #       'lr': 1e-5, 'r1': 0.3, 'r2': 0.3, 'r3': 0.3, 'w': 1e-5, 'acc': 0.8853})
    # s = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for i in s:
    #     main({'attr1': i, 'attr2': 0.7, 'epoch': 3000, 'feat': 0.2,
    #           'lr': 1e-5, 'r1': 0.5, 'r2': 0.6, 'r3': 0.8, 'w': 0.0, 'acc': 0.0764})
    main({'attr1': 0.5, 'attr2': 0.0, 'epoch': 5000, 'feat': 0.0, 'lr': 1e-5,
          'r1': 0.7, 'r2': 0.5, 'r3': 0.1, 'w': 0.0, 'alpha': 0.5, 'acc': 0.3972})
    # Macro-F1_mean: 0.0645 var: 0.0093  Micro-F1_mean: 0.1003 var: 0.0059 auc 0.5357 var: 0.0220
    # s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for i in s:
    # main({'attr1': 0.3, 'attr2': 0.0, 'epoch': 4000, 'feat': 0.0, 'lr': 1e-5,
    #       'r1': 0.1, 'r2': 0.1, 'r3': 0.4, 'w': 0.001, 'alpha': 0.5, 'acc': 0.32933})
# main({'attr1': 0.4, 'attr2': 0.7, 'epoch': 10000, 'feat': 0.2,
#       'lr': 1e-5, 'r1': 0.5, 'r2': 0.6, 'r3': 0.8, 'w': 0.0, 'acc': 0.0764})
