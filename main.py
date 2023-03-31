import pandas as pd
import numpy as np
import os
import plotly.express as px

global DIR
DIR = os.getcwd()

def get_fs(df, unit = 'milliseconds'):
    
    # Converting to UNIX time for better calculations
    time = (pd.to_datetime(df.TIME) - pd.Timestamp("1970-01-01")) // pd.Timedelta(1, unit)
    if unit == 'milliseconds':
        k = 10**3
    else:
        raise ValueError(f"Unit: {unit},  must be miliseconds for now")
    
    return 1/np.mean(np.diff(time)/k)

def main():
    
    for file in os.listdir(DIR): 
        
        # TODO: Retirar o '1'
        if file.endswith(" 1.csv"):
            
            df = pd.read_csv(os.path.join(DIR,file))
            # Add new column with only time
            df['time'] = pd.to_datetime(df.TIME).dt.strftime('%H:%M:%S')
            
            elementos = list(df.columns)
            grafico = input(f"\n\tEscolha dentre as possibilidades:\n\n{elementos}\n\n")
            if grafico not in elementos:
                raise ValueError("Valor escolhido nao existente!")
            
            fig = px.line(df, x='time', y=grafico)
            fig.show()
            
            print(get_fs(df))

if __name__ == "__main__":
    main()