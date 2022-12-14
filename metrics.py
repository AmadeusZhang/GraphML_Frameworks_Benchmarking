import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
from constants import *
import os




#Displays and saves model metrics, see example below
def display_and_save(framework:Frameworks, dataset_name:Datasets, model_name:str, predictions, y, class_names: list[str], exec_ms:float = 0,folder_name : str = 'metrics'):

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