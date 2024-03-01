import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np 
from utils.inference_utils import perform_inference
from generics import Generics, Paths
from datasets.data_loader import CustomDataset
from datasets.data_loader_configs import BaseDataConfig
from models.CustomModel import CustomModel
from models.custom_model_configs import BaseModelConfig
from pandas import DataFrame


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
    y_pred: np.ndarray(n_samples, 6)
    y_true like y_pred
    labels: col labels 
    '''
    maj = np.argmax(y_true, axis=1)
    print(maj)
    maj = [labels[clss] for clss in maj]
    maj_pred = np.argmax(y_pred, axis=1)
    print(maj_pred)
    maj_pred = [labels[clss] for clss in maj_pred]
    cnf_matrix = confusion_matrix(maj, maj_pred, labels=labels)

    plot_confusion_matrix(cnf_matrix, classes=labels, normalize= False, title='Confusion matrix')

    return cnf_matrix



def avg_mispred(y_true:np.ndarray, y_pred:np.ndarray, by_class:bool):
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

def plot_avg_mispred(differences:np.ndarray, by_class=True): 
    x_ax = np.arange(0.5, len(Generics.LABEL_COLS)+0.5)
    if by_class: 
        fig, axes = plt.subplots(1, len(Generics.LABEL_COLS))
        axes = axes.flatten()
        for i, clss in enumerate(Generics.LABEL_COLS): 
            axes[i].bar(x=x_ax, height=differences[i])
            axes[i].set_xticks(ticks=x_ax, labels=Generics.LABEL_COLS, rotation=90)
            axes[i].set_title(clss)
    else: 
        plt.bar(x=x_ax, height=differences)
        plt.title(f'Avg misprediction')
        plt.xticks(ticks=x_ax, labels=Generics.LABEL_COLS)
    plt.suptitle('Avg misprediction per true majority class')
    # plt.tight_layout()
    plt.show()


def failure_analysis(dataset:CustomDataset, model:CustomModel=None, model_dir:str=None, submission_df:DataFrame=None): 
    """
    Perform inference on the val dataset using the trained model and log results to TensorBoard.
    Show confusion matrix for majority class and average difference to ground truth distribution. 

    Args:
        test_dataset (CustomDataset): The test dataset.
        model (torch.nn.Module): The base model on with which checkpoints were generated.
        model_dir (str): path to model checkpoint to test 
        submission_df (pd.DataFrame): can be passed instead of model info; assumes it is coming from a non shuffled validation set so the order is the same 

    Returns: 
        confusion matrix of majority classes (np.ndarray)
        avg differences between true and predicted per class (np.ndarray)
        avg differences between true and predicted across classes (np.ndarray)
    """
    dataset.load_data()
    y_true = dataset.main_df[Generics.LABEL_COLS]
    
    if submission_df is None: 
        y_pred = perform_inference(dataset, model, [model_dir]).values()
    else:
        y_pred = submission_df.drop(columns=['eeg_id'])

    cnf_mtrx = majority_confusion(y_pred=y_pred, y_true=y_true, labels=Generics.LABEL_COLS)

    diff_classes = avg_mispred(y_pred=y_pred, y_true=y_true, by_class=True)
    diff_general = avg_mispred(y_pred=y_pred, y_true=y_true, by_class=False)

    plot_avg_mispred(diff_classes, by_class=True)
    plot_avg_mispred(diff_general, by_class=False)

    return cnf_mtrx, diff_classes, diff_general


def run_failure_analysis(model_config:BaseModelConfig, data_loader_config:BaseDataConfig, model:CustomModel): 
   
    model_dirs = [f"{Paths.BEST_MODEL_CHECKPOINTS}/best_{model_config.MODEL}_{model_config.NAME}_{data_loader_config.NAME}_fold_{fold}.pth" for fold in range(model_config.FOLDS)]
    dataset = CustomDataset(config=data_loader_config, mode='val') # how to get the right val fold --> val split ratio is 0.2 but there are three folds for this config 

    for fold in range(model_config.FOLDS):
        failure_analysis(model_dir=model_dirs[fold], model=model, dataset=dataset)

