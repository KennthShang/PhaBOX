import networkx as nx
import pandas as pd
import pickle as pkl
import numpy as np
import argparse
import subprocess
from scripts.ulity import *
from scripts.preprocessing import *


def draw_network(tool, midfolder, htmlpth, rootpth, db_dir):

    check_path(os.path.join(htmlpth, ''))
    check_path(os.path.join(htmlpth, f'html_{tool}'))

    if tool == 'phagcn':
        idx2label = {0: 'Autographiviridae', 1: 'Straboviridae', 2: 'Herelleviridae', 3: 'Drexlerviridae', 4: 'Demerecviridae', 5: 'Peduoviridae', 6: 'Casjensviridae', 7: 'Schitoviridae', 8: 'Kyanoviridae', 9: 'Ackermannviridae', 10: 'Rountreeviridae', 11: 'Salasmaviridae', 12: 'Vilmaviridae', 13: 'Zierdtviridae', 14: 'Mesyanzhinovviridae', 15: 'Chaseviridae', 16: 'Zobellviridae', 17: 'Orlajensenviridae', 18: 'Guelinviridae', 19: 'Steigviridae', 20: 'Duneviridae', 21: 'Pachyviridae', 22: 'Winoviridae', 23: 'Assiduviridae', 24: 'Suoliviridae', 25: 'Naomviridae', 26: 'Intestiviridae', 27: 'Crevaviridae', 28: 'Pervagoviridae'}

        network_df = pd.read_csv(f'{rootpth}/{midfolder}/phagcn_graph.csv', names=['Source', 'Target'])
        pred_df = pd.read_csv(f'{rootpth}/{midfolder}/phagcn_prediction.csv')
        label_df = pd.read_csv(f'{db_dir}/phagcn_taxonomic_label.csv')
        name2contig = {contig: acc for contig, acc in zip(pred_df['idx'].values, pred_df['Accession'])}

        all_contigs = pred_df['idx'].values
        
        for start in range(0, len(all_contigs)):
            source_node = []
            target_node = []
            source = all_contigs[start]
            name = name2contig[source]
            for target in network_df[network_df['Source'] == source]['Target']:
                source_node.append(source)
                target_node.append(target)
            for target in network_df[network_df['Target'] == source]['Source']:
                source_node.append(source)
                target_node.append(target)

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
            rename_node = [name2contig[node] if "PhaGCN" in node else node for node in node2id.keys()]
            node_df = pd.DataFrame({"Id": rename_node, "Label": rename_node, "Category": node2label.values(), "Size":Size})

            edge_df.to_csv(f'{htmlpth}/edge_{name}.csv', index=False)
            node_df.to_csv(f'{htmlpth}/node_{name}.csv', index=False)
            draw_network_cmd = f'Rscript /home/www/web/app/PhaSUIT/plot_script/generate_network.R {htmlpth} {htmlpth}/edge_{name}.csv {htmlpth}/node_{name}.csv {name} {tool}'
            try:
                _ = subprocess.check_call(draw_network_cmd, shell=True)
            except:
                pass
            f_in = open(f'{htmlpth}/html_{tool}/Net_{name}.html').read()
            with open(f'{htmlpth}/html_{tool}/Net_{name}.html', 'w') as f_out:
                f_in = f_in.replace('---', '')
                f_in = f_in.replace('title: forceNetwork', '')
                f_in = f_in.replace('header-include: |-', '')
                f_in = f_in.replace("head: ''", '')
                f_in = f_in.replace('background-color: white', '')
                f_out.write(f_in)




    if tool == 'cherry':
        network_df = pd.read_csv(f'{rootpth}/{midfolder}/cherry_graph.csv')
        pred_df = pd.read_csv(f'{rootpth}/{midfolder}/cherry_prediction.csv')
        bacteria_df = pd.read_csv(f'{db_dir}/cherry/prokaryote.csv')
        virus_df = pd.read_csv(f'{db_dir}/cherry/virus.csv')

        name2contig = {contig: acc for contig, acc in zip(pred_df['idx'].values, pred_df['Accession'])}


        all_contigs = pred_df['idx'].values
        number = 0
        for start in range(0, len(all_contigs)):
            source_node = []
            target_node = []
            source = all_contigs[start]
            name = name2contig[source]
            for target in network_df[network_df['Source'] == source]['Target']:
                source_node.append(source)
                target_node.append(target)
            for target in network_df[network_df['Target'] == source]['Source']:
                source_node.append(source)
                target_node.append(target)

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
            rename_node = [name2contig[node] if "PhaGCN" in node else node.split('~')[0] for node in node2id.keys()]
            node_df = pd.DataFrame({"Id": rename_node, "Label": node2id.keys(), "Category": node2label.values(), "Size":Size})

            edge_df.to_csv(f'{htmlpth}/edge_{name}.csv', index=False)
            node_df.to_csv(f'{htmlpth}/node_{name}.csv', index=False)
            draw_network_cmd = f'Rscript /home/www/web/app/PhaSUIT/plot_script/generate_network.R {htmlpth} {htmlpth}/edge_{name}.csv {htmlpth}/node_{name}.csv {name} {tool}'
            try:
                _ = subprocess.check_call(draw_network_cmd, shell=True)
            except:
                pass
            f_in = open(f'{htmlpth}/html_{tool}/Net_{name}.html').read()
            with open(f'{htmlpth}/html_{tool}/Net_{name}.html', 'w') as f_out:
                f_in = f_in.replace('---', '')
                f_in = f_in.replace('title: forceNetwork', '')
                f_in = f_in.replace('header-include: |-', '')
                f_in = f_in.replace("head: ''", '')
                f_in = f_in.replace('background-color: white', '')
                f_out.write(f_in)


