import numpy as np
import torch
from utils.logreg import LogReg
import torch.nn as nn
import torch.nn.functional as F

import functools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score


##################################################
# This section of code adapted from pcy1302/DMGI #
##################################################

def evaluate(embeds, ratio, train_mask, val_mask, test_mask, label, device, data, lr, wd, name):
    num_features = embeds.shape[1]
    num_classes = label.max() + 1
    xent = nn.CrossEntropyLoss()
    if name == 'aminer':
        embeds = embeds[data.label[:, 0]]
    if name == 'cite':
        embeds = embeds[data[data.main_node].y[:, 0].long()]
    train_embs = embeds[train_mask]
    val_embs = embeds[val_mask]
    test_embs = embeds[test_mask]

    train_lbls = label[train_mask]
    val_lbls = label[val_mask]
    test_lbls = label[test_mask]

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(10):
        log = LogReg(num_features, num_classes).to(device)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []
        logits_list = []

        for i in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            preds = torch.argmax(logits, dim=1)

            train_acc = torch.sum(preds == train_lbls).float() / train_lbls.shape[0]
            # print(i, loss.item())
            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')
            # print(train_acc, val_acc, test_acc)
            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc

        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(
            roc_auc_score(
                y_true=F.one_hot(test_lbls).detach().cpu().numpy(),
                y_score=best_proba.detach().cpu().numpy(),
                multi_class='ovr'
            )
        )

    print("Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f} var: {:.4f}"
          .format(
            np.mean(macro_f1s),
            np.std(macro_f1s),
            np.mean(micro_f1s),
            np.std(micro_f1s),
            np.mean(auc_score_list),
            np.std(auc_score_list)
        )
    )

    return np.mean(macro_f1s)


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values)}
            # print_statistics(statistics, f.__name__)
            return statistics

        return wrapper

    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


@repeat(10)
def label_classification(embeddings, train_mask, val_mask, test_mask, label):
    X = embeddings.detach().cpu().numpy()
    Y = label.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    # if np.isinf(X).any() == True or np.isnan(X).any() == True:
    #     return {
    #         'F1Mi': 0,
    #         'F1Ma': 0,
    #         'Acc': 0
    #     }
    X = normalize(X, norm='l2')
    X_train = X[train_mask.cpu().numpy()]
    X_val = X[val_mask.cpu().numpy()]
    X_test = X[test_mask.cpu().numpy()]
    y_train = Y[train_mask.cpu().numpy()]
    y_val = Y[val_mask.cpu().numpy()]
    y_test = Y[test_mask.cpu().numpy()]
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_pred, multi_class='ovr')

    return {'F1I': micro, 'F1A': macro, 'AUC': auc}