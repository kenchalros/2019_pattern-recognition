import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def find_index_digit(y, digit):
    """digitで指定したラベルのインデックスをlistで返す。
    Args:
        y (ndarray): ラベル(ont-hot-label化したもの)
    Returns:
        index (int list): 指定したdigitに対するラベルのインデックス
    """
    index = ( np.where(y[:, digit] == 1)[0] ).tolist()
    return index

def print_mnist_img(x):
    """mnistデータを画像として表示する
    Args:
        x (ndarray): 画像データ
    Returns:
    """
    num_img = x.shape[0]
    # 画像の表示
    fig = plt.figure(figsize = (10, (num_img / 10) + 1 ))
    idx = 1
    for i in range(num_img):
        ax = fig.add_subplot((num_img / 10) + 1 , 10, idx, xticks = [], yticks = [])
        ax.imshow(x[i].reshape(28, 28), cmap='gray')
        idx += 1
    print("num: {}".format(idx-1))
    plt.show()

def print_mnist_img_digit(x, y, digit):
    """指定したdigitの画像を表示
    Args:
        x (ndarray): 画像データ
        y (ndarray): x のラベル
        digit (int): 0~9の数字
    Returns:

    """
    index = find_index_digit(y, digit)
    print_mnist_img(x[index])

def find_index_delta_digit(model, x, y, digit):
    """指定した digit において、誤った認識がなされた画像のindexをリストで返す
    Args：
        model (keras model): 予測を行うモデル
        x (ndarray): 画像データ
        y (ndarray): xに対応したラベル
        digit (int): 指定する数字
    Returns:
        delta_index (int list): 指定したdigitのうち、誤った認識結果となった画像のindex
    """
    index_digit = find_index_digit(y, digit)
    delta_index = []
    for i in index_digit:
        val = model.predict(x[i:i+1]).argmax()
        if np.where(y[i] == 1) != np.array(val):
            delta_index.append(i)
    return delta_index

def plot_history(history):
    # 精度の履歴をプロット
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()

def print_cmx(y_true, y_pred):
    """混同行列を表示
    使い方の例
    ------------------------------------------------
    predict_classes = model.predict_classes(x_test)
    true_classes = np.argmax(y_test, axis=1)
    print_cmx(true_classes, predict_classes)
    ------------------------------------------------
    Args:
        y_true (ndarray):
            テストデータのラベル。one-hot表現の場合は元に戻しておくこと。
            eg. true_classes = np.argmax(y_test, axis=1)
        y_pred (ndarray):
            予測結果
    Returns:

    """
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cmx, annot=True, fmt='g' ,square = True, cmap="GnBu")
    plt.show()