import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
from constants import *
import os


#Displays and saves model metrics, see example below
def display_and_save(framework:Frameworks, dataset_name:Datasets, model_name:str, predictions, y, class_names: list[str], exec_ms:float = 0,folder_name : str = 'saved'):

    from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,precision_score,recall_score,confusion_matrix,ConfusionMatrixDisplay
    y_pred = predictions.argmax(axis=1)  # Use the class with the highest probability.
    y_true = y.argmax(axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision_score = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred,average='macro')
    f1_score = f1_score(y_true, y_pred,average='macro')

    # compute the AUC score
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression( solver="liblinear" ).fit( predictions, y_true )
    y_score = clf.predict_proba( predictions )
    auc_roc = roc_auc_score( y_true, y_score, average='macro', multi_class='ovo' )

    #Print metrics
    print("\nTest metrics:")
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision macro: {precision_score:.4f}')
    print(f'Recall macro: {recall:.4f}')
    print(f'F1 Score macro: {f1_score:.4f}')
    print(f'AUC-ROC macro,ovr: {auc_roc:.4f}')

    cm = confusion_matrix(y_true, y_pred,normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot( xticks_rotation=-45 )
    plt.show()


    #Save metrics in json file
    metrics = {
        'exec_ms': exec_ms,
        'accuracy':accuracy,
        'precision_score_macro':precision_score,
        'recall_macro':recall,
        'f1_score_macro':f1_score,
        'confusion_matrix':cm.tolist(),
        'auc_roc_macro_ovr':auc_roc,
    }
    date : str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = os.path.join(
        folder_name,
        dataset_name.lower,
        f'{str(framework.lower)}_{model_name}_{date}.json')
    print(f'Saving metrics to {filename}')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(metrics, f)


#Example of use
def example():
    display_and_save(framework=Frameworks.PYG,
                     dataset_name=Datasets.PUBMED,
                     model_name=Models.GCN.lower,
                     predictions=np.array([[0.1,0.2,0.3],[0.8,0.1,0.1],[0.1,0.1,0.8]]),  #Output of the model
                     y=np.array([[0,0,1],[1,0,0],[0,1,0]]),  #True labels, one hot encoded
                     class_names=Datasets.PUBMED.mappings.values(),
                     exec_ms=1050)


class History:
    loss: list[float]
    acc: list[float]
    val_loss: list[float]
    val_acc: list[float]

    def __init__(self):
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

    def print_last(self):
        epoch = len(self.loss)
        print(
            f"Epoch {epoch}: loss={self.loss[-1]:.4f}, acc={self.acc[-1]:.4f}, val_loss={self.val_loss[-1]:.4f}, val_acc={self.val_acc[-1]:.4f}")


def plot_history(history: History, framework:Frameworks, dataset_name:Datasets, model_name:str, folder_name : str = 'saved'):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.loss, label="loss")
    ax[0].plot(history.val_loss, label="val_loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(history.acc, label="acc")
    ax[1].plot(history.val_acc, label="val_acc")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    plt.show()
    date: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    filename = os.path.join(folder_name, dataset_name.lower)
    if not os.path.exists(filename):
        os.makedirs(filename, exist_ok=True)
    filename = os.path.join(
        filename,
        f'{str(framework.lower)}_{model_name}_{date}-history.pdf')
    fig.savefig(filename, bbox_inches='tight')

