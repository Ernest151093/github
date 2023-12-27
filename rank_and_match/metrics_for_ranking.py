from math import log2

from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int: #
    num_pairs = 0
    ys_true_desc, indices = ys_true.sort(descending=True)
    ys_pred_desc = ys_pred[indices]
    for i in range(len(ys_true_desc)):
        for j in range(i+1,len(ys_true_desc)):
            if ys_pred_desc[i]>ys_pred_desc[j] or (ys_true_desc[i]==ys_true_desc[j] and ys_pred_desc[i]<ys_pred_desc[j]):
                continue
            else:
                num_pairs += 1
    return num_pairs

def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'const':
        return float(y_value)
    elif gain_scheme == 'exp2':
        return float(2 ** y_value - 1)


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    dcg = 0
    ys_true_desc = ys_true[ys_pred.sort(descending=True).indices]
    for i in range(len(ys_true)):
        dcg += compute_gain(ys_true_desc[i], gain_scheme=gain_scheme) / log2(i + 2)
    return dcg


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    # допишите ваш код здесь
    idealdcg = 0
    ys_true_desc, indices = ys_true.sort(descending=True)
    for i in range(len(ys_true_desc)):
        idealdcg += compute_gain(ys_true_desc[i], gain_scheme=gain_scheme) / log2(i + 2)

    return dcg(ys_true, ys_pred, gain_scheme) / idealdcg


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    tp = 0
    fp = 0
    if sum(ys_true) == 0:
        return -1.0
    ys_pred_desc, indices = ys_pred.sort(descending=True)
    ys_true_desc = ys_true[indices]
    print(ys_pred_desc)
    print(ys_true_desc)
    for i in range(len(ys_pred_desc[:k])):
        if ys_true_desc[i] == 1 and ys_pred_desc[i] >= 0.5:
            tp += 1
        elif ys_true_desc[i] == 0 and ys_pred_desc[i] >= 0.5:
            fp += 1

    precision_k = tp / (tp + fp)
    return precision_k


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    ys_pred_desc, indices = ys_pred.sort(descending=True)
    ys_true_desc = ys_true[indices]
    rank = 1
    i = 0
    print(ys_pred_desc)
    print(ys_true_desc)
    #for i in range(len(ys_true_desc)):
    while ys_true_desc[i] != 1:
        rank+=1
        i += 1
    print('rank:', rank)
    if len(ys_true.shape) == 1:
        mrr = 1/rank
    return mrr


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15) -> float:
    ys_pred_desc, indices = ys_pred.sort(descending=True)
    p_rel = ys_true[indices]
    print(ys_pred_desc)
    print(p_rel)
    p_look = 1
    pfound = p_look * p_rel[0]
    print(pfound)
    for i in range(1, len(p_rel)):
        print(i)
        p_look = p_look * (1 - p_rel[i - 1]) * (1 - p_break)
        print(p_look)
        pfound += p_look * p_rel[i]
        print(pfound)

    return float(pfound)


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    if sum(ys_true) == 0:
        return -1.0
    ys_pred_desc, indices = ys_pred.sort(descending=True)
    ys_true_desc = ys_true[indices]
    print(ys_pred_desc)
    print(ys_true_desc)
    ap = 0
    rel_doc = 0
    ret_doc = 0
    for i in range(len(ys_true_desc)):
        if ys_true_desc[i] == 1:
            rel_doc += 1
            ret_doc += 1
            precision = rel_doc/ret_doc
            ap += precision
        else:
            ret_doc += 1
            precision = rel_doc/ret_doc
            ap += (0 * precision)
    print(rel_doc)
    ap = ap/(rel_doc)
    return ap
