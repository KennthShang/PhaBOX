import pandas as pd
import pickle as pkl
import numpy as np
import argparse
import subprocess
from collections import Counter


parser = argparse.ArgumentParser(description="""Main script of PhaSUIT.""")
parser.add_argument('--tool', help='Tool for analysis',  default = 'phamer')
parser.add_argument('--rootpth', help='rootpth of the user', default='user_0/')
parser.add_argument('--scripts', help='pth of the R script', default='plot_script/')
parser.add_argument('--out', help='output path of the user', default='out/')
parser.add_argument('--midfolder', help='mid folder for intermidiate files', default='midfolder/')
inputs = parser.parse_args()

rootpth = inputs.rootpth
outpth  = inputs.out
scriptpth = inputs.scripts
tool    = inputs.tool

df = pd.read_csv(f'{rootpth}/{outpth}/{tool}_prediction.csv')

cnt = Counter(df['Pred'].values)
category = []
num = []
for name, number in cnt.most_common():
    category.append(name)
    num.append(number)

df = pd.DataFrame({"Category":category, "value": num})
df.to_csv(f'{rootpth}/{outpth}/{tool}_figure.csv', index=False)

Rscipt_cmd = f'Rscript {scriptpth}/generate_pie.R {rootpth} {tool}'
_ = subprocess.check_call(Rscipt_cmd, shell=True)