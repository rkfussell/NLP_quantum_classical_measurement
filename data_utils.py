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

def code(df):
    df=df.rename(columns={'QuantitativeComparison':"QC"})
    df=df.rename(columns={'ProposedIteration':"PI"})
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['QC']=df['QC'].fillna(0)
    df['PI']=df['PI'].fillna(0)

    df=df.dropna()

    df = df[df.QC != 'r']
    df = df[df.PI != 'r']
    df = df[df.QC != 'R']
    df = df[df.PI != 'R']
    df = df[df.PI != 'a']
    df.loc[df['PI'] =='x', 'PI'] = 1
    df.loc[df['QC'] =='x', 'QC'] = 1
    df.loc[df['PI'] =='X', 'PI'] = 1
    df.loc[df['QC'] =='X', 'QC'] = 1
    df.loc[df['PI'] =='Lx', 'PI'] = 1
    df.loc[df['QC'] =='Lx', 'QC'] = 1


    #df.loc[df['QC']>1, 'QC'] = 1
    df.loc[df['PI'] =='L', 'PI'] = 0
    df.loc[df['QC'] =='L', 'QC'] = 0
    
    return df

#chunk by files
def dataframe(files, prompt = None, dataDir='xlsx'):
    df = pd.DataFrame({})
    for i, file in enumerate(files):
        if file[0]=='.':
            continue
        if i==0:
            df = pd.DataFrame(pd.read_excel('{}/{}'.format(dataDir,file)))
            df["filename"] = file
        else:
            tmp = pd.DataFrame(pd.read_excel('{}/{}'.format(dataDir,file)))
            tmp["filename"] = file
            df=pd.concat([tmp, df])
    df = code(df)
    df['Sentences'] = df['Sentences'].astype(str)
    if prompt != None:
        df['Sentences']=prompt+df['Sentences']+"[/INST]"
    return df


#Text preprocessing
def text_preprocessing(s):
    """
    - Lowercase the sentence
    - alter numbers
    - keep some punctuation, remove other punctuation
    - fix contractions?
    - Lemmatize?
    - Tokenize?
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()

    s = s.lower().split(" ")
    #fix contractions e.g. you're becomes you are
    s = [contractions.fix(word) for word in s]
    ##lemmatize
    s = [lemmatizer.lemmatize(word) for word in s]
    s = " ".join(s).lower()
    #Find numbers in their writing and replace with a tag
    s = re.sub('\d*\.+\d+', 'DEC', s)
    s = re.sub('\d+', 'INT', s)
    #keep certain punctuations, remove everything else
    s = re.sub(r'[^\w\s\?\'\(\)\-\+\:\*]', ' ', s)

    return s
    
    
### Bag of Words
def my_tokenizer(text):
    """
    Input to TfidfVectorizer
    """
    # create a space between special characters 
    text=re.sub("(\\W)"," \\1 ",text)

    # split based on whitespace
    return re.split("\\s+",text)


### BERT

def text_preprocessing_simple(text):
    """
        remove trailing whitespace
    """
    try:
        # Remove trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()
    except:
        print(text)
    return text
def weights(y_train):
    weight=len(y_train)/np.sum(y_train)
    weight2=len(y_train)/(np.sum(1-y_train))
    total=weight+weight2
    weight, weight2=weight/total, weight2/total
    return [weight, weight2]


def get_y(code, data):
    #e.g. data could be train or val
    y_data = np.array(data[code].values, dtype=int)
    return y_data



def getData(dataDir='xlsx', holdoutDir='holdout',ValCutoff=50):
    files=os.listdir(dataDir)
    random.shuffle(files)
    
    #chunk by files
    train=dataframe(files[ValCutoff:], prompt = None, dataDir = dataDir)
    val=dataframe(files[:ValCutoff], prompt = None, dataDir = dataDir)
    train['Sentences'] = train['Sentences'].astype(str)
    val['Sentences'] = val['Sentences'].astype(str)
    testFiles=os.listdir(holdoutDir)
    
    test=dataframe(testFiles, prompt=None, dataDir=holdoutDir)
    
    return train, val, test
    
    
