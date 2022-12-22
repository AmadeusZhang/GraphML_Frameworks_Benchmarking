import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
from enum import Enum
import os

class Framework(Enum):
    PYG = 'pyg'
    STELLARGRAPH = 'stellargraph'
    DGL = 'dgl'

class Datasets(Enum):
    CORA = 'cora'
    CITESEER = 'citeseer'
    PUBMED = 'pubmed'


#Displays and saves model metrics, see example below
def display_and_save(framework:Framework,dataset_name:Datasets, model_name:str, predictions, y, class_names: list[str], exec_ms:float):
    folder_name : str = 'metrics'#folder where the metrics will be saved

    from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,precision_score,recall_score,confusion_matrix,ConfusionMatrixDisplay
    y_pred = predictions.argmax(axis=1)  # Use the class with highest probability.
    y_true = y.argmax(axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision_score = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred,average='macro')
    f1_score = f1_score(y_true, y_pred,average='macro')
    #au_roc = roc_auc_score(out.detach().numpy()[data.test_mask], label_set,average='macro', multi_class='ovr')

    #Print metrics
    print("\nTest metrics:")
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision macro: {precision_score:.4f}')
    print(f'Recall macro: {recall:.4f}')
    print(f'F1 Score macro: {f1_score:.4f}')
    #print(f'AU-ROC macro,ovr: {au_roc:.4f}')

    cm = confusion_matrix(y_true, y_pred,normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.show()

    #Save metrics in json file
    metrics = {
        'exec_ms': exec_ms,
        'accuracy':accuracy,
        'precision_score_macro':precision_score,
        'recall_macro':recall,
        'f1_score_macro':f1_score,
        'confusion_matrix':cm.tolist()
    }
    date : str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(
        folder_name,
        str(dataset_name.value),
        f'{str(framework.value)}_{model_name}_{date}.json')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(metrics, f)

#Example of use
def example():
    display_and_save(framework=Framework.PYG,
                     dataset_name=Datasets.PUBMED,
                     model_name='GCN_example1',
                     predictions=np.array([[0.1,0.2,0.3],[0.8,0.1,0.1],[0.1,0.1,0.8]]),#Output of the model
                     y=np.array([[0,0,1],[1,0,0],[0,1,0]]),#True labels, one hot encoded
                     class_names=['A','B','C'],
                     exec_ms=1050)