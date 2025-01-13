import pandas as pd 
import numpy as np
import os 
import re
import string
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.corpus import stopwords
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model, model_selection
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score

# bert imports
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,WeightedRandomSampler
import torch.nn as nn
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F

# llama imports
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from copy import deepcopy
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import AutoModelForSequenceClassification
import torch
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import scipy

import random
import time





def evaluate_roc(probs, y_true, plot = False):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """

    if probs.ndim == 2:
        preds = probs[:, 1] #we changed this from preds = probs to fix an error in llama.train()
    else:
        preds = probs
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
       
    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = balanced_accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    

    if plot == True:
        # Plot ROC AUC
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    return roc_auc

def calibration(probs, y_true):
    # Get accuracy over the test set
    y_pred = np.where(probs >= 0.5, 1, 0)
    accuracy = balanced_accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')

    print(confusion_matrix(y_true, y_pred))
    
    abs_diff = abs(y_true - probs)
    positives = abs_diff[y_true == 1]
    negatives = abs_diff[y_true == 0]
    fig, axs = plt.subplots(1,3)
    axs[0].set_ylim([0, 1])
    axs[1].set_ylim([0, 1])
    axs[2].set_ylim([0, 1])
    axs[0].set_xlim([0, 1])
    axs[1].set_xlim([0, 1])
    axs[2].set_xlim([0, 1])
    #axs[0].set_title("overall")
    #axs[0].hist(abs_diff, weights=np.ones(len(abs_diff)) / len(abs_diff), bins = 20)
    #axs[0].set(xlabel = "|y_i - predicted probability|")
    axs[1].set_title("positives")
    axs[1].hist(positives, weights=np.ones(len(positives)) / len(positives), bins = 20)
    axs[1].set(xlabel = "|human code - prob.|")
    axs[2].set_title("negatives")
    axs[2].hist(negatives, weights=np.ones(len(negatives)) / len(negatives), bins = 20)
    axs[2].set(xlabel = "|human code - prob.|")
    
    
    
def get_auc_CV(model, X, y):
    """
    Return the average AUC score from cross-validation.
    """
    # Set KFold to shuffle data before the split
    kf = StratifiedKFold(5, shuffle=True, random_state=1)

    # Get AUC scores
    auc = cross_val_score(
        model, X, y, scoring="roc_auc", cv=kf)

    return auc.mean()


