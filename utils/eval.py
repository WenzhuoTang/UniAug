import torch

from ogb.linkproppred import Evaluator as LPEvaluator
from ogb.graphproppred import Evaluator as GPEvaluator
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
)
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
    multiclass_exact_match,
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_auroc,
)


def evaluate_hits(evaluator, y_pred_pos, y_pred_neg, k_list):
    results = {}
    for K in k_list:
        evaluator.K = K
        results[f'hits@{K}'] = evaluator.eval({
            'y_pred_pos': y_pred_pos,
            'y_pred_neg': y_pred_neg,
        })[f'hits@{K}']

    return results
        
def evaluate_mrr(evaluator, y_pred_pos, y_pred_neg):
    # if len(y_pred_pos) != len(y_pred_neg):
    #     return {}
    # y_pred_neg = y_pred_neg.view(-1, 1)
    # evaluator.eval_metric = 'mrr'
    # mrr_list = evaluator.eval({
    #     'y_pred_pos': y_pred_pos,
    #     'y_pred_neg': y_pred_neg,
    # })['mrr_list']
    
    # return {'mrr': mrr_list.mean().item()}

    y_pred_pos = y_pred_pos.view(-1, 1)
    # y_pred_neg = y_pred_neg.view(y_pred_pos.shape[0], -1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    results = {}
    results['mrr'] = (1 / ranking_list).mean().item()

    for k in [1,3,10,20,50,100]:
        results[f'mrr_hits{k}'] = (ranking_list <= k).float().mean().item()

    return results

def evaluate_auc(y_true, y_pred):
    # auc = roc_auc_score(y_true, y_pred)
    # ap = average_precision_score(y_true, y_pred)
    auc = binary_auroc(y_pred, y_true).item()
    ap = binary_average_precision(y_pred, y_true).item()

    return {'auc': auc, 'ap': ap}

def evaluate_link_prediction(y_pred_pos, y_pred_neg, evaluator=None, hits=True, mrr=True, auc=False):
    if evaluator is None:
        evaluator = LPEvaluator(name='ogbl-collab')

    results = {}
    if hits:
        k_list = [1, 3, 10, 20, 50, 100]
        results.update(evaluate_hits(evaluator, y_pred_pos, y_pred_neg, k_list))

    if mrr:
        results.update(evaluate_mrr(evaluator, y_pred_pos, y_pred_neg))
   
    if auc:
        y_pred = torch.cat([y_pred_pos, y_pred_neg])
        y_true = torch.cat([torch.ones(y_pred_pos.size(0), dtype=int), 
                            torch.zeros(y_pred_neg.size(0), dtype=int)])
        results.update(evaluate_auc(y_true.cpu(), y_pred.cpu()))

    return results

def evaluate_node_classification(y_true, y_pred):
    # acc = accuracy_score(y_true.argmax(1), y_pred.argmax(1))
    # macro_acc = balanced_accuracy_score(y_true.argmax(1), y_pred.argmax(1))
    # macro_f1 = f1_score(y_true.argmax(1), y_pred.argmax(1), average='macro')
    # macro_auc = roc_auc_score(y_true, y_pred, average='macro')
    num_classes = len(y_true.argmax(1).unique())
    acc = multiclass_exact_match(y_pred.argmax(1), y_true.argmax(1), num_classes=num_classes).item()
    macro_acc = multiclass_accuracy(y_pred.argmax(1), y_true.argmax(1), num_classes=num_classes, average='macro').item()
    macro_f1 = multiclass_f1_score(y_pred.argmax(1), y_true.argmax(1), num_classes=num_classes, average='macro').item()
    macro_auc = multiclass_auroc(y_pred, y_true.argmax(1), num_classes=num_classes, average='macro').item()

    return {'acc': acc, 'macro_acc': macro_acc, 'macro_f1': macro_f1, 'macro_auc': macro_auc}

def evaluate_graph_property_prediction(y_true, y_pred, evaluator=None, data_name=None):
    if evaluator is None: 
        evaluator = GPEvaluator(name=data_name)
    input_dict = {'y_true': y_true, 'y_pred': y_pred}

    return evaluator.eval(input_dict)