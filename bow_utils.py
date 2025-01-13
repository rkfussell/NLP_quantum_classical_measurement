import pandas as pd 
import numpy as np
import os 
import re
import string
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
nltk.download('wordnet')
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
from data_utils import *
import random
import time

def get_train_x_bows(train):
    # Preprocess text
    X_train_preprocessed = [text_preprocessing(str(text)) for text in train.Sentences]
    #X_val_preprocessed = [text_preprocessing(str(text)) for text in val.Sentences]
    # Tokenize with binary encoding (not TF-IDF)
    vectorizer = TfidfVectorizer(ngram_range = (1,1), binary = True, use_idf = False, norm = None, tokenizer = my_tokenizer)
    
    #Create train and val inputs and outputs
    X_train = vectorizer.fit_transform(X_train_preprocessed)
    #X_val = vectorizer.transform(X_val_preprocessed)
    return vectorizer, X_train#, X_val
def get_val_x_bows(val, vectorizer):
    # Preprocess text
    X_val_preprocessed = [text_preprocessing(str(text)) for text in val.Sentences]
    #Create train and val inputs and outputs
    X_val = vectorizer.transform(X_val_preprocessed)
    return vectorizer, X_val

