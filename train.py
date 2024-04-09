import numpy as np
import torch
import torch.nn.functional as F
from utils.load import load_acm, load_mag, load_freebase, make_sparse_eye, load_Aminer_Large, load_imdb, load_cite
from utils.params import set_params, acm_params, aminer_params, freebase_params, mag_params, imdb_params, cite_params
from utils.evaluate import evaluate
from model import Cross_View
from torch_geometric import seed_everything


# args = acm_params()
args = aminer_params()
# args = cite_params()
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
            data[data.main_node].y,
            device,
            data,
            0.01,
            0,
            args.dataset,
        )


if __name__ == '__main__':
    # Tune hyperparameters:
    # s = [0.04, 0.08, 0.1, 0.12, 0.16, 0.2]
    # for i in s:
    #     main({'alpha': 0.4, 'attr1': 0.0, 'attr2': 0.15, 'epoch': 4000, 'feat': 0.5,
    #           'lr': 5e-5, 'r1': 0.1, 'r2': i, 'r3': 0.55, 'w': 1e-5, 'acc': 0.8853})
    main({'alpha': 0.5, 'attr1': 0.0, 'attr2': 0.15, 'epoch': 6000, 'feat': 0.5,
          'lr': 1e-5, 'r1': 0.3, 'r2': 0.3, 'r3': 0.3, 'w': 1e-5, 'acc': 0.8853})
