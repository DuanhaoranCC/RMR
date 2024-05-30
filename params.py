import argparse


def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[1, 5, 10, 20])
    parser.add_argument('--use_cuda', default=True, action="store_true")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0.0)

    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--w', type=float, default=0.0005)
    parser.add_argument('--epoch', type=int, default=200)

    # model-specific parameters
    parser.add_argument('--attr1', type=float, default=0.0)
    parser.add_argument('--attr2', type=float, default=0.0)
    parser.add_argument('--feat', type=float, default=0.8)
    parser.add_argument('--r1', type=float, default=0.9)
    parser.add_argument('--r2', type=float, default=0.3)
    parser.add_argument('--r3', type=float, default=0.0)

    args, _ = parser.parse_known_args()
    return args


def aminer_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="aminer")
    parser.add_argument('--ratio', type=int, default=[1, 5, 10, 20])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0.0)

    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--w', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=4000)

    # model-specific parameters
    parser.add_argument('--attr1', type=float, default=0.0)
    parser.add_argument('--attr2', type=float, default=0.15)
    parser.add_argument('--feat', type=float, default=0.5)
    parser.add_argument('--r1', type=float, default=0.1)
    parser.add_argument('--r2', type=float, default=0.15)
    parser.add_argument('--r3', type=float, default=0.55)

    args, _ = parser.parse_known_args()
    return args


def imdb_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--ratio', type=int, default=[1, 5, 10, 20])
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0.0)

    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--w', type=float, default=5e-5)
    parser.add_argument('--epoch', type=int, default=5500)

    # model-specific parameters
    parser.add_argument('--attr1', type=float, default=0.0)
    parser.add_argument('--attr2', type=float, default=0.3)
    parser.add_argument('--feat', type=float, default=0.55)
    parser.add_argument('--r1', type=float, default=0.9)
    parser.add_argument('--r2', type=float, default=0.8)
    parser.add_argument('--r3', type=float, default=0.0)

    args, _ = parser.parse_known_args()
    return args
