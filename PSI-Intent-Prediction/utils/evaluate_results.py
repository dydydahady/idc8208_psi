import json
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score, f1_score, precision_recall_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def evaluate_intent(groundtruth='', prediction='', args=None, writer=None, epoch=0):
    with open(groundtruth, 'r') as f:
        gt_intent = json.load(f)

    with open(prediction, 'r') as f:
        pred_intent = json.load(f)

    gt = []
    pred = []
    for vid in gt_intent.keys():
        for pid in gt_intent[vid].keys():
            for fid in gt_intent[vid][pid].keys():
                gt.append(gt_intent[vid][pid][fid]['intent'])
                pred.append(pred_intent[vid][pid][fid]['intent'])
    gt = np.array(gt)
    pred = np.array(pred)
    res = measure_intent_prediction(gt, pred, args, writer, epoch)
    print('Acc: ', res['Acc'])
    print('F1: ', res['F1'])
    print('mAcc: ', res['mAcc'])
    print('ConfusionMatrix: ', res['ConfusionMatrix'])

    # Plot Precision-Recall Curve
    plot_precision_recall_curve(gt, pred, writer, epoch)

    return res['F1']



def measure_intent_prediction(target, prediction, args, writer, epoch):
    print("Evaluating Intent ...")
    results = {
        'Acc': 0,
        'F1': 0,
        'mAcc': 0,
        'ConfusionMatrix': [[]],
    }

    bs = target.shape[0]
    lbl_target = target # bs
    lbl_pred = np.round(prediction) # bs, use 0.5 as threshold

    # hard label evaluation - acc, f1
    Acc = accuracy_score(lbl_target, lbl_pred) # calculate acc for all samples
    F1_score = f1_score(lbl_target, lbl_pred, average='macro')

    intent_matrix = confusion_matrix(lbl_target, lbl_pred)  # [2 x 2]
    intent_cls_acc = np.array(intent_matrix.diagonal() / intent_matrix.sum(axis=-1)) # 2
    intent_cls_mean_acc = intent_cls_acc.mean(axis=0)

    # results['MSE'] = MSE
    results['Acc'] = Acc
    results['F1'] = F1_score
    results['mAcc'] = intent_cls_mean_acc
    results['ConfusionMatrix'] = intent_matrix

    # Logging metrics to TensorBoard
    if writer is not None:
        writer.add_scalar('Accuracy/Val', Acc, epoch)
        writer.add_scalar('F1_Score/Val', F1_score, epoch)
        writer.add_scalar('Mean_Accuracy/Val', intent_cls_mean_acc, epoch)
        writer.add_figure('Confusion_Matrix/Val', plot_confusion_matrix(intent_matrix), epoch)

    return results

def plot_precision_recall_curve(target, prediction, writer, epoch):
    precision, recall, _ = precision_recall_curve(target, prediction)
    auc_score = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, marker='.', label=f'Precision-Recall curve (AUC = {auc_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid()

    # Save plot to TensorBoard
    if writer is not None:
        writer.add_figure('Precision_Recall_Curve', plt.gcf(), epoch)

    plt.show()


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(['Not Cross', 'Cross'])
    ax.set_yticklabels(['Not Cross', 'Cross'])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black")

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')

    return fig


if __name__ == '__main__':
    args = None
    # evaluate_intent('gt.json', 'pred.json', args)
    test_gt_file = './val_intent_gt.json'
    test_pred_file = './val_intent_prediction.json'
    score = evaluate_intent(test_gt_file, test_pred_file, args)
    print("Rankding score is : ", score)
