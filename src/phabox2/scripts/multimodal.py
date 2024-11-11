import os
import pandas as pd
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--node', help='cherry node file (tsv)',  default = 'cherry_network_nodes.tsv')
parser.add_argument('--edge', help='cherry edge file (tsv)',  default = 'cherry_network_edges.tsv')
parser.add_argument('--prediction', help='cherry prediction file (tsv)',  default = 'cherry_prediction.tsv')
parser.add_argument('--outpth', help='output path',  default = 'refined_out/')
inputs = parser.parse_args()

try:
    _ = os.mkdir(inputs.outpth)
except:
    print("Directory already exists")
    print('Please change the output path')
    exit()

df = pd.read_csv(f'{inputs.prediction}', sep='\t')
df = df[df['Host'] != '-']

ref_df = pd.read_csv(f'{inputs.node}', sep='\t')
ref_df = ref_df[(ref_df['Host'] != '-')&(ref_df['TYPE'] == 'Ref')]


Source = df['Accession'].tolist()+ref_df['Accession'].tolist()
Target = df['Host'].tolist()+ref_df['Host'].tolist()
Weight = df['CHERRYScore'].tolist()+[1]*len(ref_df['Accession'].tolist())

host_edges = pd.DataFrame({"Source": Source, "Target": Target, "Weight": Weight})


Accession = df['Host'].tolist() + df['Accession'].tolist() + list(set(ref_df['Host'].tolist()))
Host = df['Host'].tolist() + df['Host'].tolist() + list(set(ref_df['Host'].tolist()))
TYPE = ['Host'] * len(df['Host'].tolist()) + ['Query'] * len(df['Accession'].tolist()) + ['Host'] * len(set(ref_df['Host'].tolist()))


host_nodes = pd.DataFrame({"Accession": Accession, "Host": Host, "TYPE": TYPE})

edges_df = pd.concat((pd.read_csv(f'{inputs.edge}', sep='\t'), host_edges))
nodes_df = pd.concat((pd.read_csv(f'{inputs.node}', sep='\t'), host_nodes))

edges_df.to_csv(f'{inputs.outpth}/cherry_network_edges.tsv', sep='\t', index=False)
nodes_df.to_csv(f'{inputs.outpth}/cherry_network_nodes.tsv', sep='\t', index=False)