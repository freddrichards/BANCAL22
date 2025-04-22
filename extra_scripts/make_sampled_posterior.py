import numpy as np
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

    data = np.loadtxt(os.path.join(fold_BANCAL22_data, 'samples_postburnin.csv'), skiprows=1)

    return data

data = load_data()

###############################################################################################
# 3. GENERATE RANDOM SAMPLE OF POSTERIOR ANELASTICITY MODELS AND FIND MAP
###############################################################################################

def find_MAP():

    outfile = os.path.join(fold_samples_output, 'MAP_model.txt')
    if os.path.exists(outfile):
        return None
    idx_MAP = np.argmax(data[:,0])  # find MAP model by sorting data by posterior probability and extracting the maximum
    data_MAP = data.copy()[idx_MAP,np.arange(0,8,1)]  # get MAP model probability and parameters
    np.savetxt(outfile, data_MAP, delimiter='\t')  # save MAP model

    return data_MAP

def generate_sample():

    outfile = os.path.join(fold_samples_output, 'data_sample.txt')
    if os.path.exists(outfile):
        return None
    sample_size = 1000
    total_samples = np.shape(data)[0]
    idx_sample = np.random.default_rng().integers(low=0, high=total_samples, size=sample_size)
    data_sample = data.copy()[idx_sample,:]
    data_sample = data_sample[:,np.arange(0,8,1)]
    np.savetxt(outfile, data_sample, delimiter='\t')

    return data_sample

find_MAP()
generate_sample()