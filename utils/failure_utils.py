import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np 


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    From https://stackoverflow.com/questions/60477129/at-least-one-label-specified-must-be-in-y-true-target-vector-is-numerical
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def majority_confusion(y_pred, y_true, labels): 
    '''
    Shows confusion matrix for majority class.
    '''
    maj = np.argmax(y_true, axis=0)
    maj_pred = np.argmax(y_pred, axis=0)
    cnf_matrix = confusion_matrix(maj, maj_pred)

    plot_confusion_matrix(cnf_matrix, classes=labels,normalize= False,  title='Confusion matrix')

    return cnf_matrix



def avg_mispred(y_pred:np.ndarray, y_true:np.ndarray, by_class=True):
    '''
    Returns average misclassification per label (per true majority class if by_class is True)
    '''

    diff = y_pred-y_true

    if by_class:
        maj = np.argmax(y_true, axis=1)
        classes = np.sort(np.unique(maj))
        result = np.array([np.mean(diff[maj==clss], axis=0) for clss in classes])
    else: 
        result = np.mean(diff, axis=0)

    return result 

