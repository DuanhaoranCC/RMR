import torch
from load import load_acm, load_Aminer_Large, load_imdb
from params import acm_params, aminer_params, imdb_params
from evaluate import evaluate
from model import Cross_View
from torch_geometric import seed_everything
import warnings

warnings.filterwarnings("ignore")

# args = acm_params()
args = aminer_params()
# args = imdb_params()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

if args.dataset == "acm":
    load_data = load_acm
elif args.dataset == "aminer":
    load_data = load_Aminer_Large
elif args.dataset == "imdb":
    load_data = load_imdb
elif args.dataset == "cite":
    load_data = load_cite

data = load_data()
data = data.to(device)
print(data)


def main(args):
    seed_everything(0)
    model = Cross_View(data=data, hidden_dim=64, feat_drop=args.feat,
                       att_drop1=args.attr1, att_drop2=args.attr2, r1=args.r1,
                       r2=args.r2,
                       r3=args.r3,
                       ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w)
    for epoch in range(1, args.epoch + 1):
        model.train()
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()
        print("Epoch:{}, Loss:{}".format(epoch, loss))

    model.eval()
    embeds = model.get_embed(data).cpu()
    for ratio in args.ratio:
        evaluate(
            embeds,
            data[data.main_node][f'{ratio}_train_mask'],
            data[data.main_node][f'{ratio}_val_mask'],
            data[data.main_node][f'{ratio}_test_mask'],
            data[data.main_node].y.long(),
            device,
            data,
            0.01,
            0,
            args.dataset,
        )


if __name__ == '__main__':
    # ACM
    # main({'attr1': 0.0, 'attr2': 0.0, 'epoch': 200, 'feat': 0.8, 'lr': 0.005, 'r1': 0.9, 'r2': 0.3, 'r3': 0.0,
    #       'w': 0.0005})
    # IMDB
    # main({'attr1': 0.0, 'attr2': 0.3, 'epoch': 5500,
    #       'feat': 0.55, 'r1': 0.9, 'r2': 0.8, 'r3': 0.0, 'w': 5e-5, 'lr': 5e-5, 'acc': 0.5037})
    # Aminer
    # main({'alpha': 0.4, 'attr1': 0.0, 'attr2': 0.15, 'epoch': 4000, 'feat': 0.5,
    #       'lr': 5e-5, 'r1': 0.1, 'r2': 0.15, 'r3': 0.55, 'w': 1e-5, 'acc': 0.8853})

    main(args)
