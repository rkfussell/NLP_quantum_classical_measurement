import pandas as pd 
import numpy as np
import os 
import matplotlib.pyplot as plt
from sklearn import linear_model #, model_selection
from transformers import BertTokenizer
import random
import time
import math
import torch

from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

from transformers import AutoModelForCausalLM

import gc  # garbage collect library
import utilities as ut
import bert_ut as bu 
import llama_ut as lu
import data_utils as du 
import bow_utils as bow 
import evals
import sys
import scipy 
from prompts import *
from scipy.special import softmax
from peft import get_peft_model, LoraConfig, TaskType

import pickle


def runTest(seed=0,code='QC',evalLlama=True, LlamaChatbot = False, dataDir='xlsx', holdoutDir='holdout', ValCutoff=50,savePath="savedData", prompt="G", hfToken="", saveModel=False, saveModelPath="" ):
    
    ut.set_seed((seed))
    train, val, test= du.getData(dataDir=dataDir, holdoutDir=holdoutDir,ValCutoff=ValCutoff)
    
    
    
    if torch.cuda.is_available():
        device='cuda'
    elif torch.backends.mps.is_available():
        device='mps'
    else:
        device='cpu'
    print("Device detected. Using device: {} ".format(device))
    
        
        
    batch_size = 16 
    

    llama_checkpoint = "meta-llama/Llama-2-7b-hf"
    
    
    df_full = pd.DataFrame()
    
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

    if not LlamaChatbot:
        #randomizes order of training set... 
        
        current_training_set = train.sample(n = len(train))
        
        
        #TRAIN
        #Prepare x and y for training set 
        y_train = du.get_y(code, current_training_set)
        #for bow:
        print("Optimizing Bag of Words") 
        
        vectorizer, X_train = bow.get_train_x_bows(current_training_set)
        #for bert:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        MAX_LEN_BERT = bu.get_max_len_bert(tokenizer, current_training_set, val, include_val=True)
        
        print("Training Bert")
        train_data, train_sampler, train_dataloader = bu.get_train_x_bert(code, y_train, current_training_set, batch_size, tokenizer, MAX_LEN_BERT, balanced = True)
        
        #Fit logistic regression model to data
        Log = linear_model.LogisticRegression(penalty = 'l2', random_state = 123, max_iter = 10000, class_weight = 'balanced')
        Log.fit(X_train,y_train)
        if not os.path.exists(saveModelPath):
            os.makedirs(saveModelPath)
        if saveModel:
            pickle.dump(Log, open(saveModelPath+"/BoW.sav", 'wb'))
            pickle.dump(vectorizer, open(saveModelPath+"/BoW_vectorizer.sav", 'wb'))
        
        epochs=3
        bert_classifier, optimizer, scheduler = bu.initialize_model(device, epochs=epochs, train_dataloader = train_dataloader)
        bu.train_BERT(device, bert_classifier, optimizer, scheduler, train_dataloader, val_dataloader=None, epochs=epochs, evaluation=False)
        print(saveModel)
        if saveModel:
            print("save model")
            torch.save(bert_classifier.state_dict(), saveModelPath+"/bert.pth")
        ##train LLaMA
        
        if evalLlama:
            print("Training Llama.  This may take a while.") 
        
            pos_weights, neg_weights, max_words, data, col_to_delete = lu.preprocessing_for_llama(code, current_training_set, val)
            
            MAX_LEN = 512 
            prompts=getPrompts()
            llama_tokenized_datasets, llama_data_collator, llama_tokenizer = lu.tokenize_for_llama(llama_checkpoint, data, col_to_delete, MAX_LEN, prompt=prompts[prompt], hfToken=hfToken)
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            llama_model = lu.set_llama_model(llama_checkpoint, hfToken)
            llama_trainer = lu.train_llama(llama_model, llama_tokenized_datasets, llama_data_collator, pos_weights, neg_weights)
            llama_model=llama_model.eval()
            if saveModel:
                llama_model.save_pretrained(saveModelPath+'/llamaAdapter')
                
            
        else: 
            print("Skipping Llama, set evalLlama = True to run.")
        
        
        
    #TEST
    #prepare val x and y for bag of words
    y_val = du.get_y(code, val)

    if not LlamaChatbot:
        print("Generating validation results for Bag of Words")
        vectorizer, X_val = bow.get_val_x_bows(val, vectorizer)
        # predict the labels on validation dataset: BoW
        predictions_Log = Log.predict(X_val)
        probs_Log = np.array(Log.predict_proba(X_val))[:,1]
        #for BERT
        print("Generating validation results for BERT")
        val_data, val_sampler, val_dataloader = bu.get_val_x_bert(code, y_val, val, batch_size, tokenizer, MAX_LEN_BERT)
        # predict the labels on validation dataset: BERT
        probs_BERT = np.array(bu.bert_predict(device, bert_classifier, val_dataloader))[:,1]
    
    
        
        predictions_BERT = np.where(probs_BERT >= 0.5, 1, 0)
        ##for LLaMA
        if evalLlama:
            print("evaluating llama on the test and validation set:  may take a while...") 
            
            llama_tokenized_datasets = lu.tokenize_for_llama_test(val, code, llama_tokenizer, col_to_delete, MAX_LEN)
            # predict the labels on validation dataset: LLaMA
            predictions_Llama_val=np.array(lu.llama_logits(llama_model, llama_tokenized_datasets, 1)[0])
            llama_tokenized_datasets = lu.tokenize_for_llama_test(test, code, llama_tokenizer, col_to_delete, MAX_LEN)
            # predict the labels on validation dataset: LLaMA
            predictions_Llama_test=np.array(lu.llama_logits(llama_model, llama_tokenized_datasets, 1)[0])
            
    y_test  = du.get_y(code, test)

    if not LlamaChatbot:
        #BoW
        print("Generating test results for Bag of Words")
        vectorizer, X_test = bow.get_val_x_bows(test, vectorizer)
        predictions_Log_test = Log.predict(X_test)
        probs_Log_test = np.array(Log.predict_proba(X_test))[:,1]
        #BERT
        print("Generating test results for BERT")
        test_data, test_sampler, test_dataloader = bu.get_val_x_bert(code, y_test, test, batch_size, tokenizer, MAX_LEN_BERT)
        # predict the labels on validation dataset: BERT
        probs_BERT_test = np.array(bu.bert_predict(device, bert_classifier, test_dataloader))[:,1]
        predictions_BERT_test = np.where(probs_BERT_test >= 0.5, 1, 0)
    elif LlamaChatbot:
        ### insert code to run the chatbot on the y_test data ###
        ut.set_seed((seed))
        modelpath="meta-llama/Meta-Llama-3-8B"
        
        model = AutoModelForCausalLM.from_pretrained(
            modelpath,    
            torch_dtype=torch.bfloat16,
            device_map="auto", token=hf_token,
            
            #attn_implementation="flash_attention_2",  # make sure to have flash-attn pip-installed
        )
        
        tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False, token='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') 
        
        # simple wrapper around model.generate() 
        def generate(prompt, max_new_tokens = 100): 
            prompt_tokenized=tokenizer(prompt, return_tensors="pt").to("cuda")
            output_tokenized = model.generate(
                **prompt_tokenized, 
                temperature=0.01,
                top_p= 0,
                max_new_tokens = max_new_tokens,
                pad_token_id = tokenizer.eos_token_id)[0]
            
            output_tokenized = output_tokenized[len(prompt_tokenized["input_ids"][0]):]
            output = tokenizer.decode(output_tokenized)
        
            return output
        if code == "QC":
            prompt='''Below is a json file that has sentences and a classification of whether or not the sentences demonstrates: 
    Comparison of Quantities - Students should apply data analysis tools quantitatively to make some sort of comparison (between data, best fit lines, predictions, etc.  
    Students use t' (t-prime) values and chi-squared values to compare quantities, however the heading of a table containing these does not count. Students may also compare the uncertainty in the mean. '''
            prompt+='''
                Input:
            {
              "text":'''+"Our t-prime value had an absolute value between 1 and 3, which indicates that it is possible that A and B are the same (that the period of the pendulum is unaffected by the amplitude of the string), but there is not enough evidence to clearly conclude they are indistinguishable."+'''"
            }
            Analysis:
            {
              "Quantatitive Comparison?": "Yes"
            }'''
            
            
            prompt+='''
                Input:
            {
              "text":'''+"To make our chi-squared value close to 1, using our standard deviation of the acceleration from our preliminary tests during last lab session as an estimate of dy, the uncertainty would need to be approximately:"+'''"
            }
            Analysis:
            {
              "Quantatitive Comparison?": "Yes"
            }'''
            
            prompt+='''
                Input:
            {
              "text":'''+"t’ = 10.09885"+'''"
            }
            Analysis:
            {
              "Quantatitive Comparison?": "Yes"
            }'''
            prompt+='''
                Input:
            {
              "text":'''+"The uncertainty is higher in B (20 degree swing) because we had an outlying data point."+'''"
            }
            Analysis:
            {
              "Quantatitive Comparison?": "Yes"
            }'''
        elif code == "PI":
            prompt='''Below is a json file that has sentences and a classification of whether or not the sentences demonstrates: 
Proposed Iteration - Students should be able to suggest additional rounds of experimentation and choose appropriate improvements. Could be based on experimental evidence.   
Must be a sentence in future or present tense that proposes an experimental choice (we could, we will, we are doing). Not something that was done in the past (e.g. this method improved our uncertainty) AND
Must have at least one word/phrase you can point to is synonymous with either more measurements (additional angles, new trials) or change/improve (new/different object, new/different method, revised/improved/adjusted/modified experiment, to reduce or account for uncertainty) or plan for next time/ plan for future'''
        prompt+='''
            Input:
        {
          "text":'''+"For more accurate results the use of a camera to time the oscillations would omit errors due to human reaction time. "+'''"
        }
        Analysis:
        {
          "Proposed Iteration?": "Yes"
        }'''
        
        
        prompt+='''
            Input:
        {
          "text":'''+"Continued experimentation"+'''"
        }
        Analysis:
        {
          "Proposed Iteration?": "No"
        }'''
        
        prompt+='''
            Input:
        {
          "text":'''+"Session 2 new experiment"+'''"
        }
        Analysis:
        {
          "Proposed Iteration?": "No"
        }'''
        prompt+='''
            Input:
        {
          "text":'''+"We could add more trials to improve our uncertainties"+'''"
        }
        Analysis:
        {
          "Proposed Iteration?": "Yes"
        }'''

        base = prompt
        predictions_chatbot = []
        for i in range(0,len(test)): #len(neg)):
        
            prompt=base+'''
            Input:  
            "text": ""'''+test.iloc[i]['Sentences']+'''"
            }
            Analysis:
            {
              "Quantatitive Comparison?": '''
            ans=(generate(prompt, 3) )
            if "Yes" in ans:
                predictions_chatbot.append(1)
            elif "No" in ans:
                predictions_chatbot.append(0)
            else:
                predictions_chatbot.append(0)
                print("Chatbot provided neither yes or no for this sentence:")
                print(test.iloc[i]['Sentences'])
        prefix="{}/{}/seed_{}".format(savePath, code, seed)
        try:
            os.mkdir(prefix)
        except:
            print("directory already exists") 
        np.save(prefix+'/chatbot_test.npy', predictions_chatbot)
    if not LlamaChatbot:
        #store data
        df = pd.DataFrame()
        y_val = du.get_y(code, val)
        df["Human_coding"] = y_val
        
        print("saving results to file: {}".format(savePath)) 
        prefix="{}/{}/seed_{}".format(savePath, code, seed)
        
        try:
            os.mkdir(prefix)
        except:
            print("directory already exists") 
        
        np.save(prefix+'/human_validation.npy', y_val)
        np.save(prefix+'/bert_probs_val.npy', probs_BERT)
        np.save(prefix+'/logreg_probs_val.npy', probs_Log)
        if evalLlama:
            np.save(prefix+'/llama_probs_val.npy', softmax(predictions_Llama_val, axis=1))
    
    
    np.save(prefix+'/human_test.npy', y_test)
    if not LlamaChatbot:
        np.save(prefix+'/bert_probs_test.npy', probs_BERT_test)
        np.save(prefix+'/logreg_probs_test.npy', probs_Log_test)
        if evalLlama:
            np.save(prefix+'/llama_probs_test.npy', softmax(predictions_Llama_test, axis=1))




import torch
from bert_ut import BertClassifier
from transformers import LlamaForSequenceClassification


def loadModel(model,path, hfToken=''):
    if model=='bow':
        model = pickle.load(open(path+'/BoW.sav', 'rb'))
        tokenizer = pickle.load(open(path+'/BoW_vectorizer.sav', 'rb'))
        return (model, tokenizer)
    elif model=='bert':
        model=BertClassifier()
        model.load_state_dict(state_dict=torch.load(path))
        model.eval()
    elif model=='llama':
        llama_checkpoint = "meta-llama/Llama-2-7b-hf"
        llama_model =  LlamaForSequenceClassification.from_pretrained(
            llama_checkpoint,
            device_map={"": 0},
            token=hfToken
        )
        llama_model.config.pad_token_id = llama_model.config.eos_token_id
        llama_model.load_adapter(path)
        model=llama_model
        model.eval()
    return model 
        


def single_trial_BoW(s, code, test_bank, vectorizer,bow_model, p_test, n):
    ut.set_seed(s)
    val = test_bank.sample(n = n)
    #prepare val x and y for bag of words
    y_val = du.get_y(code, val)
    vectorizer, X_val = bow.get_val_x_bows(val, vectorizer)
    # predict the labels on validation dataset: BoW
    predictions_Log = bow_model.predict(X_val)
    #store data
    E_C = predictions_Log.mean()
    E_H = y_val.mean()
    df_temp = pd.DataFrame({"E_C": [E_C], "E_H": [E_H], "p_test": [p_test]})
    return df_temp
def single_trial_BERT(s, code, test_bank, batch_size, tokenizer, max_len, device, bert_classifier, p_test, n):
    ut.set_seed(s)
    val = test_bank.sample(n = n)
    #prepare val x and y for bert
    y_val = du.get_y(code, val)
    val_data, val_sampler, val_dataloader = bu.get_val_x_bert(code, y_val, val, batch_size, tokenizer, max_len)
    # predict the labels on validation dataset: BERT
    probs_BERT = np.array(bu.bert_predict(device, bert_classifier, val_dataloader))[:,1]
    predictions_BERT = np.where(probs_BERT >= 0.5, 1, 0)
    #store data
    E_C = predictions_BERT.mean()
    E_H = y_val.mean()
    df_temp = pd.DataFrame({"E_C": [E_C], "E_H": [E_H], "p_test": [p_test]})
    return df_temp
def single_trial_LLaMA(s, code, test_bank, llama_model, llama_tokenizer, col_to_delete, max_len, p_test, n):
    ut.set_seed(s)
    val = test_bank.sample(n = n)
    #prepare val x and y for llama
    y_val = du.get_y(code, val)
    llama_tokenized_datasets = lu.tokenize_for_llama_test(val, code, llama_tokenizer, col_to_delete, max_len)
    # predict the labels on validation dataset: LLaMA
    #merged_model = llama_model.merge_and_unload()
    logits_LLaMA=np.array(lu.llama_logits(llama_model, llama_tokenized_datasets)[0])
    probs_LLaMA = np.array(scipy.special.softmax(logits_LLaMA, axis=1))[:,1]
    predictions_LLaMA = np.where(probs_LLaMA >= 0.5, 1, 0)
    #store data
    E_C = predictions_LLaMA.mean()
    E_H = y_val.mean()
    df_temp = pd.DataFrame({"E_C": [E_C], "E_H": [E_H], "p_test": [p_test]})
    return df_temp


def apply_to_uncoded(seed=0,code = "QC", evalLlama=True, holdoutDir='holdout', hf_token = "", N_bank = 200, N_trials = 20, n = 50, saveModelPath="", uncoded=True):
    
    
    from main import loadModel
    from main import getPrompts

    (bow_model, vectorizer) = loadModel('bow',saveModelPath)
    bert_model = loadModel('bert',saveModelPath+'/bert.pth', hf_token)
    llama_model = loadModel('llama',saveModelPath+'/llamaAdapter', hf_token)

    
    ut.set_seed(seed)

    testFiles=os.listdir(holdoutDir)
    test=du.dataframe(testFiles, prompt=None, dataDir=holdoutDir)
    test['Sentences'] = test['Sentences'].astype(str)
    
    if torch.cuda.is_available():
        device='cuda'
    elif torch.backends.mps.is_available():
        device='mps'
    else:
        device='cpu'
    print("Device detected. Using device: {} ".format(device))

    
    batch_size = 16 
    ### TODO CHANGE TO USING VAL DATA FOR THIS ??? ###
    #sort into positive and negative
    pos = test[test[code] == 1]
    neg = test[test[code] == 0]
    
    available_test_data_pos = pos[:N_bank] 
    available_test_data_neg = neg[:N_bank]
    available_test_data_pos = available_test_data_pos.sample(frac = 1) #shuffle
    available_test_data_neg = available_test_data_neg.sample(frac = 1)
    
    
    all_data_df = pd.read_excel("Data/all_data_df.xlsx")
    all_data_df = all_data_df[all_data_df["Sentences"].notnull()]
    all_data_df.insert(2,"target",0,True)

    #tokenizers
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    MAX_LEN_BERT = bu.get_max_len_bert(tokenizer, test, all_data_df, include_val=True)

    llama_checkpoint = "meta-llama/Llama-2-7b-hf"
    col_to_delete = ['Sentences']
    MAX_LEN = 512 
    prompts=getPrompts()

    llama_tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint, add_prefix_space=True, token=hf_token)
    llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    
    #llama_tokenized_datasets, llama_data_collator, llama_tokenizer = lu.tokenize_for_llama(llama_checkpoint, data, col_to_delete, MAX_LEN, prompt=prompts[prompt], hfToken=hfToken)

    df_out_BoW = pd.DataFrame({"E_C": [], "E_H": [], "p_test": []})
    df_out_BERT = pd.DataFrame({"E_C": [], "E_H": [], "p_test": []})
    df_out_LLaMA = pd.DataFrame({"E_C": [], "E_H": [], "p_test": []})
    
    p_tests = [0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9]
    for p_test in p_tests:
        N_pos_test = math.floor(N_bank*p_test)
        N_neg_test = N_bank - N_pos_test
        ut.set_seed((math.floor(p_test*50))) # set seed to something arbitrary for each p_test to avoid repeated values in the samples
        if (len(available_test_data_pos) >= N_pos_test) and (len(available_test_data_neg) >= N_neg_test):
            #make test bank
            #if N_pos_test >= len(available_test_data_pos) and N_neg_test >= len(available_test_data_neg):
            test_bank = pd.concat((available_test_data_pos.sample(n = N_pos_test), available_test_data_neg.sample(n = N_neg_test)))
            test_bank = test_bank.sample(frac = 1);
            test_bank.to_csv("testbank{}{}.csv".format(N_pos_test,code))
            #sample from test bank to get val
            for s in range(N_trials):
                df_temp_BoW = single_trial_BoW(s, code, test_bank,vectorizer, bow_model, p_test, n)
                df_out_BoW = pd.concat((df_out_BoW, df_temp_BoW))
                df_temp_BERT = single_trial_BERT(s, code, test_bank, batch_size, tokenizer, MAX_LEN_BERT, device, bert_model, p_test, n);
                df_out_BERT = pd.concat((df_out_BERT, df_temp_BERT));
                if evalLlama:
                    df_temp_LLaMA = single_trial_LLaMA(s, code, test_bank, llama_model, llama_tokenizer, col_to_delete, MAX_LEN, p_test, n);
                    df_out_LLaMA = pd.concat((df_out_LLaMA, df_temp_LLaMA));
            
    
    df_out_BoW.to_excel("df_out_BoW_" + code + ".xlsx")
    df_out_BERT.to_excel("df_out_BERT_" + code + ".xlsx")
    if evalLlama:
        df_out_LLaMA.to_excel("df_out_LLaMA_" + code + ".xlsx")
    
    #apply to uncoded data
    if uncoded:
        #Bag of words
        vectorizer, X_uncoded = bow.get_val_x_bows(all_data_df, vectorizer)
        
        # predict the labels on uncoded dataset: BoW
        predictions_Log = bow_model.predict(X_uncoded)
        probs_Log = np.array(bow_model.predict_proba(X_uncoded))[:,1]
        
        #Print the words with highest and lowest coefficient values in the model
        coefficients = bow_model.coef_[0]
        coefs_dict = {i: bow_model.coef_[0][i] for i in range(len(bow_model.coef_[0]))}
        sorted_keys = sorted(coefs_dict, key=coefs_dict.get)
        words_dict = dict((vectorizer.vocabulary_[word], word) for word in vectorizer.vocabulary_)
        words = []
        coefs = []
        for num in sorted_keys[:30]:
            words.append(words_dict[num])
            coefs.append(coefs_dict[num])
        for num in sorted_keys[-30:]:
            words.append(words_dict[num])
            coefs.append(coefs_dict[num])
        print("\n Bag of words: highest and lowest ranked words\n")
        print(words)
        print(coefs)
        
        #BERT
        y_uncoded_zeros = np.zeros(len(all_data_df))
        uncoded_data, uncoded_sampler, uncoded_dataloader = bu.get_val_x_bert(code, y_uncoded_zeros, all_data_df, batch_size, tokenizer, MAX_LEN_BERT)
    
        # predict the labels on validation dataset: BERT
        probs_BERT = np.array(bu.bert_predict(device, bert_model, uncoded_dataloader))[:,1]
        
        predictions_BERT = np.where(probs_BERT >= 0.5, 1, 0)
    
        #LLaMA
        llama_tokenized_datasets = lu.tokenize_for_llama_test(all_data_df, code, llama_tokenizer, col_to_delete, MAX_LEN)
        # predict the labels on validation dataset: LLaMA
        #merged_model = llama_model.merge_and_unload()
        uncoded_logits_LLaMA=np.array(lu.llama_logits(llama_model, llama_tokenized_datasets)[0])
        probs_LLaMA = np.array(scipy.special.softmax(uncoded_logits_LLaMA, axis=1))[:,1]
        predictions_LLaMA = np.where(probs_LLaMA >= 0.5, 1, 0)
        
        
        all_data_df["BERT_probs"] = probs_BERT
        all_data_df["LR_probs"] = probs_Log
        all_data_df["BERT_predictions"] = predictions_BERT
        all_data_df["LR_predictions"] = predictions_Log
        all_data_df["LLaMA_probs"] = probs_LLaMA
        all_data_df["LLaMA_predictions"] = predictions_LLaMA
        
        all_data_df.to_excel("All_lab_notes_machine_coded_bert_lr_" + code + ".xlsx")


    
