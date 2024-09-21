import torch
import torch.nn.functional as F
from load import load_acm, load_Aminer_Large, load_imdb, load_pubmed
from params import acm_params, aminer_params, imdb_params, pubmed_params
from evaluate import evaluate
from model import Cross_View
from torch_geometric import seed_everything
import warnings

warnings.filterwarnings("ignore")

# args = acm_params()
args = aminer_params()
# args = imdb_params()
# args = pubmed_params()
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
elif args.dataset == "pubmed":
    load_data = load_pubmed

data = load_data()
data = data.to(device)
print(data)


def main(space):
    seed_everything(0)
    model = Cross_View(data=data, hidden_dim=64, feat_drop=space['feat'],
                       att_drop1=space['attr1'], att_drop2=space['attr2'], r1=space['r1'],
                       r2=space['r2'],
                       r3=space['r3'],
                       ).to(device)
    # total_params = sum(p.numel() for p in model.parameters())
    # print("Number of parameter: %.2fM" % (total_params / 1e6))

    model.eval()
    embeds = model.get_embed(data).cpu()
    if args.dataset == 'pubmed':
        f1 = evaluate(
            embeds,
            data[data.main_node].train.cpu(),
            data[data.main_node].val.cpu(),
            data[data.main_node].test.cpu(),
            data[data.main_node].y.long().cpu(),
            device,
            data,
            0.01,
            0,
            args.dataset
        )
        print(f1)
    else:
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

    optimizer = torch.optim.Adam(model.parameters(), lr=space['lr'], weight_decay=space['w'])
    for epoch in range(1, int(space['epoch']) + 1):

        model.train()
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()
        print("Epoch:{}, Loss:{}".format(epoch, loss))
        # if epoch % 1000 == 0:
        #     print("Epoch:{}, Loss:{}".format(epoch, loss))

    # torch.save(model.state_dict(), f"DBLP_pretrain.pth")
    model.eval()
    # model.load_state_dict(torch.load(f"DBLP_pretrain.pth"))
    embeds = model.get_embed(data).cpu()

    if args.dataset == 'pubmed':
        f1 = evaluate(
            embeds,
            data[data.main_node].train.cpu(),
            data[data.main_node].val.cpu(),
            data[data.main_node].test.cpu(),
            data[data.main_node].y.long().cpu(),
            device,
            data,
            0.01,
            0,
            args.dataset
        )
        print(f1)
    else:
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
    # PubMed
    # main({'attr1': 0.0, 'attr2': 0.6, 'epoch': 3000, 'feat': 0.3, 'lr': 1e-5,
    #       'r1': 0.2, 'r2': 0.1, 'r3': 0.1, 'w': 0.0, 'acc': 0.3286})
    main(args)
