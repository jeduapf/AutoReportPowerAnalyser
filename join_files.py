import pandas as pd
import numpy as np
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from graphs import input_plot

global DIR
DIR = os.getcwd()

def generate_df(DIR = DIR):
    '''
        From all files in the current folder join all them in a single Pandas Dataframe
        
        Inputs:
            DIR: Directory where all files to be concatenated are. Usually the current directory.
        
        Outputs:
            df: Pandas dataframe containing the concatenation of all files in the current filder
        
    '''
    iteration_files = iter(os.listdir(DIR))
    
    # Join files in one plot
    file = next(iteration_files)
    flag = True

    while file:
        # If it is the first file just open it 
        if flag:
            if file.endswith(".csv"):
                flag = False
                df = pd.read_csv(os.path.join(DIR,file))
                
        # Else append new file data to the dataframe
        else:
            if file.endswith(".csv"):
                df_aux = pd.read_csv(os.path.join(DIR,file))
                df = pd.concat([df,df_aux])
                
        # Check if there is another file to append
        try:
            file = next(iteration_files)
        except:
            file = False
            
    return df

def main():
    
    # TODO: terminar de arrumar isso aqui 
    dir = input('Coloque o diretorio dos arquivos')
    if dir is None:
        dir = DIR
    
    df = generate_df(dir)       
    input_plot(df)
    
if __name__ == "__main__":
    main()