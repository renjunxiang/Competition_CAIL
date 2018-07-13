import numpy as np
from sklearn.metrics.scorer import f1_score


def predict2half(predictions):
    return np.where(predictions > 0.5, 1.0, 0.0)


def predict2top(predictions):
    one_hots = []
    for prediction in predictions:
        one_hot = np.where(prediction == prediction.max(), 1.0, 0.0)
        one_hots.append(one_hot)
    return np.array(one_hots)


def predict2both(predictions):
    one_hots = []
    for prediction in predictions:
        one_hot = np.where(prediction > 0.5, 1.0, 0.0)
        if one_hot.sum() == 0:
            one_hot = np.where(prediction == prediction.max(), 1.0, 0.0)
        one_hots.append(one_hot)
    return np.array(one_hots)


def f1_avg(y_pred, y_true):
    '''
    mission 1&2
    :param y_pred:
    :param y_true:
    :return:
    '''
    f1_micro = f1_score(y_pred=y_pred, y_true=y_true, pos_label=1, average='micro')
    f1_macro = f1_score(y_pred=y_pred, y_true=y_true, pos_label=1, average='macro')
    return (f1_micro + f1_macro) / 2


def distance_score(y_true, y_pred):
    '''
    mission 3
    :param y_true:
    :param y_pred:
    :return:
    '''
    result = 0
    n = len(y_true)
    for i in range(n):
        v = np.abs(np.log10(y_true[i][0] + 1) - np.log10(y_pred[i][0] + 1))
        if y_true[i][0] == 500:
            if y_pred[i][0] > 400:
                result += 1 / n
        elif y_true[i][0] == 400:
            if y_pred[i][0] <= 400 and y_pred[i][0] > 300:
                result += 1 / n
        else:
            if v <= 0.2:
                result += 1 / n
            elif v <= 0.4:
                result += 0.8 / n
            elif v <= 0.6:
                result += 0.6 / n
            elif v <= 0.8:
                result += 0.4 / n
            elif v <= 1.0:
                result += 0.2 / n
            else:
                pass
    return result


if __name__ == '__main__':
    print(f1_avg(y_pred=np.array([[0, 1], [1, 0]]),
                 y_true=np.array([[0, 1], [1, 1]])))
