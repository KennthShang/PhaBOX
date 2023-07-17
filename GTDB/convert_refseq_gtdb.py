import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--outfile', type=str, default = 'out/cherry_prediction.csv')
inputs = parser.parse_args()



# script for converting
pred_df = pd.read_csv(f'{inputs}')
refseq_df = pd.read_csv(f'refseq_prokaryote.csv')
gtdb_df = pd.read_csv(f'gtdb_prokaryote.csv')
find_name = []
new_pred = []
for name, pred in zip(pred_df['Accession'].values, pred_df['Pred'].values):
    try:
        accession = refseq_df[refseq_df['Species'] == pred]['Accession'].values[0]
        new_pred.append(gtdb_df[gtdb_df['Accession'] == accession])
        find_name.append(name)
    except:
        print(f"Missing for {pred}")

new_pred_df = pd.concat(new_pred)
new_pred_df['Accession'] = find_name
order = ['Accession', 'Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
new_pred_df = new_pred_df[order]
new_pred_df.to_csv('cherry_prediction_gtdb_ver.csv', index=False)