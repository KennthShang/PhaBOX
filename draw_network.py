import networkx as nx
import pandas as pd
import pickle as pkl
import numpy as np
import argparse
from scripts.ulity import *
from scripts.preprocessing import *


parser = argparse.ArgumentParser(description="""Main script of PhaSUIT.""")
parser.add_argument('--tool', help='Tool for analysis',  default = 'phagcn')
parser.add_argument('--rootpth', help='rootpth of the user', default='user_0/')
parser.add_argument('--out', help='output path of the user', default='out/')
parser.add_argument('--midfolder', help='mid folder for intermidiate files', default='midfolder/')
parser.add_argument('--dbdir', help='database directory',  default = 'database/')
inputs = parser.parse_args()


tool   = inputs.tool
midfolder = inputs.midfolder
rootpth   = inputs.rootpth
db_dir    = inputs.dbdir
out_dir   = inputs.out

check_path(os.path.join(rootpth, 'figures/'))

if tool == 'phagcn':
    idx2label = {0:"Ackermannviridae", 1:"Autographiviridae", 2:"Demerecviridae", 
                 3:"Drexlerviridae", 4:"Herelleviridae", 5:"Myoviridae", 6:"Podoviridae", 
                 7:"Siphoviridae"}
    network_df = pd.read_csv(f'{rootpth}/{midfolder}/phagcn_graph.csv')
    pred_df = pd.read_csv(f'{rootpth}/{out_dir}/phagcn_prediction.csv')
    label_df = pd.read_csv(f'{db_dir}/phagcn_taxonomic_label.csv')
    name2contig = {contig: acc for contig, acc in zip(pred_df['idx'].values, pred_df['Contig'])}

    all_contigs = pred_df['idx'].values
    number = 0
    for start in range(0, len(all_contigs), 50):
        source_node = []
        target_node = []
        for source in all_contigs[start:start+50]:
            for target in network_df[network_df['Source'] == source]['Target'].values[:10]:
                source_node.append(source)
                target_node.append(target)
            for target in network_df[network_df['Target'] == source]['Source'].values[:10]:
                source_node.append(source)
                target_node.append(target)

        tmp_edge_df = pd.DataFrame({"Source":source_node, "Target": target_node})
        all_neighbor = set(tmp_edge_df["Target"].values)
        check_node = {item: 1 for item in all_neighbor}
        
        for neighbor in all_neighbor:
            for target in network_df[network_df['Source'] == neighbor]['Target'].values[:10]:
                if 'PhaGCN' in target:
                    source_node.append(neighbor)
                    target_node.append(target)
                else:
                    try:
                        check_node[target]
                        source_node.append(neighbor)
                        target_node.append(target)
                    except:
                        continue
            for target in network_df[network_df['Target'] == neighbor]['Source'].values[:10]:
                if 'PhaGCN' in target:
                    source_node.append(neighbor)
                    target_node.append(target)
                else:
                    try:
                        check_node[target]
                        source_node.append(neighbor)
                        target_node.append(target)
                    except:
                        continue
        
        tmp_edge_df = pd.DataFrame({"Source":source_node, "Target": target_node})
        tmp_edge_df = tmp_edge_df.drop_duplicates()

        node_set = set(tmp_edge_df['Source'].values) | set(tmp_edge_df['Target'].values)
        node2id = {node:idx for idx, node in enumerate(node_set)}
        node2label = {}
        for node in node_set:
            try:
                label = pred_df[pred_df['idx'] == node]['Pred'].values[0]
            except:
                label = idx2label[label_df[label_df['contig_id'] == node]['family'].values[0]]
            node2label[node] = label

        Sources = [node2id[item] for item in tmp_edge_df['Source'].values]
        Targets = [node2id[item] for item in tmp_edge_df['Target'].values]
        Weight  = [1 for i in Sources]
        Type    = ['Undirected' for i in Sources]
        edge_df = pd.DataFrame({'Source': Sources, 'Target': Targets, 'Type': Type, 'Weight': Weight}) 
        Size    = [36 if "PhaGCN" in item else 1 for item in node2id.keys()]
        rename_node = [name2contig[node] if "PhaGCN" in node else node for node in node2id.values()]
        node_df = pd.DataFrame({"Id": rename_node, "Label": node2id.keys(), "Category": node2label.values(), "Size":Size})

        edge_df.to_csv(f'{rootpth}/figures/edge_{number}.csv', index=False)
        node_df.to_csv(f'{rootpth}/figures/node_{number}.csv', index=False)
        number += 1




if tool == 'cherry':
    network_df = pd.read_csv(f'{rootpth}/{midfolder}/cherry_graph.csv')
    pred_df = pd.read_csv(f'{rootpth}/{out_dir}/cherry_prediction.csv')
    bacteria_df = pd.read_csv(f'{db_dir}/cherry/prokaryote.csv')
    virus_df = pd.read_csv(f'{db_dir}/cherry/virus.csv')

    name2contig = {contig: acc for contig, acc in zip(pred_df['idx'].values, pred_df['Contig'])}


    all_contigs = pred_df['idx'].values
    number = 0
    for start in range(0, len(all_contigs), 50):
        source_node = []
        target_node = []
        for source in all_contigs[start:start+50]:
            for target in network_df[network_df['Source'] == source]['Target'].values[:10]:
                source_node.append(source)
                target_node.append(target)
            for target in network_df[network_df['Target'] == source]['Source'].values[:10]:
                source_node.append(source)
                target_node.append(target)

        tmp_edge_df = pd.DataFrame({"Source":source_node, "Target": target_node})
        all_neighbor = set(tmp_edge_df["Target"].values)
        check_node = {item: 1 for item in all_neighbor}
        
        for neighbor in all_neighbor:
            for target in network_df[network_df['Source'] == neighbor]['Target'].values[:10]:
                if 'PhaGCN' in target:
                    source_node.append(neighbor)
                    target_node.append(target)
                else:
                    try:
                        check_node[target]
                        source_node.append(neighbor)
                        target_node.append(target)
                    except:
                        continue
            for target in network_df[network_df['Target'] == neighbor]['Source'].values[:10]:
                if 'PhaGCN' in target:
                    source_node.append(neighbor)
                    target_node.append(target)
                else:
                    try:
                        check_node[target]
                        source_node.append(neighbor)
                        target_node.append(target)
                    except:
                        continue
        
        tmp_edge_df = pd.DataFrame({"Source":source_node, "Target": target_node})
        tmp_edge_df = tmp_edge_df.drop_duplicates()

        node_set = set(tmp_edge_df['Source'].values) | set(tmp_edge_df['Target'].values)
        node2id = {node:idx for idx, node in enumerate(node_set)}
        node2label = {}
        for node in node_set:
            try:
                label = pred_df[pred_df['idx'] == node]['Top_1_label'].values[0]
            except:
                try:
                    label = bacteria_df[bacteria_df['Accession'] == node]['Species'].values[0]
                except:
                    label = virus_df[virus_df['Accession'] == node]['Species'].values[0]
            node2label[node] = label

        Sources = [node2id[item] for item in tmp_edge_df['Source'].values]
        Targets = [node2id[item] for item in tmp_edge_df['Target'].values]
        Weight  = [1 for i in Sources]
        Type    = ['Undirected' for i in Sources]
        edge_df = pd.DataFrame({'Source': Sources, 'Target': Targets, 'Type': Type, 'Weight': Weight}) 
        Size    = [36 if "PhaGCN" in item else 1 for item in node2id.keys()]
        rename_node = [name2contig[node] if "PhaGCN" in node else node for node in node2id.values()]
        node_df = pd.DataFrame({"Id": rename_node, "Label": node2id.keys(), "Category": node2label.values(), "Size":Size})

        edge_df.to_csv(f'{rootpth}/figures/edge_{number}.csv', index=False)
        node_df.to_csv(f'{rootpth}/figures/node_{number}.csv', index=False)
        number += 1


