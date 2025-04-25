import numpy as np
import pandas as pd
import os
import sys, getopt
import configparser

###############################################################################################
# 1. GET DIRECTORIES FROM CONFIG
###############################################################################################
config_obj = configparser.ConfigParser()
config_obj.read('config.ini')
variables = config_obj["variables"]
date = variables["date"]
folds = config_obj["directories"]

# read -d variable for output date
opts, args = getopt.getopt(sys.argv[1:], "d:")
for opt,arg in opts:
    if opt == '-d':
        date = str(arg)

fold_BANCAL22_data = os.path.join(folds["data_input"],date)
fold_samples_output = os.path.join(folds["data_input"],date)

os.makedirs(fold_samples_output, exist_ok=True)

###############################################################################################
# 2. LOAD DATA
###############################################################################################

def load_data():
    data = pd.read_csv(os.path.join(fold_BANCAL22_data, 'samples_postburnin.csv'), delimiter="\t")
    return data


# def load_data():
#
#     data = np.loadtxt(os.path.join(fold_BANCAL22_data, 'samples_postburnin.csv'), skiprows=1)
#
#     return data

###############################################################################################
# 3. GENERATE RANDOM SAMPLE OF POSTERIOR ANELASTICITY MODELS AND FIND MAP
###############################################################################################

def find_summary_model():
    if os.path.exists(fold_BANCAL22_data+"/summary_model.txt"):
        print("mean model exists, skipping..")
        data_summary=1
        exists=1
    else:
        data=load_data()
        data_mean=data.mean(axis=0)
        data_std=data.std(axis=0)
        exists=0
        data_summary=pd.concat([data_mean, data_std], axis=1).T
    return data_summary, exists

def save_summary(data_summary):
    data_summary.to_csv(fold_BANCAL22_data+"/summary_model.txt", sep="\t", float_format="%20.15f", index=False)
    
def find_MAP_model():

    if os.path.exists(fold_BANCAL22_data+"/MAP_model.txt"):
        print("MAP model exists, skipping..")
        data_MAP=1
        exists=1
    else:
        data=load_data()
        idx=np.argmax(data['Posterior']) # find MAP model by sorting data by posterior probability and extracting the maximum
        data_MAP=data.iloc[idx] # get MAP model probability and parameters
        exists=0
    return data_MAP, exists
    
def save_MAP(data_MAP):
    data_MAP.to_csv(fold_BANCAL22_data+"/MAP_model.txt", sep="\t", header=False)
    
def generate_data_sample(sample_size=1000):
    if os.path.exists(fold_BANCAL22_data+"/data_sample.txt"):
        print("data sample exists, skipping..")
        data_sample=1
        exists=1
    else:
        data=load_data()
        data_sample = data.sample(n = sample_size)
        exists=0
    return data_sample, exists

def save_data_sample(data_sample):
    data_sample.to_csv(fold_BANCAL22_data+"/data_sample.txt", sep="\t")
    
data_sample, exists = generate_data_sample()

if exists < 1:
    save_data_sample(data_sample)

data_MAP, exists = find_MAP_model()
if exists < 1:
    save_MAP(data_MAP)

data_summary, exists = find_summary_model()
if exists < 1:
    save_summary(data_summary)