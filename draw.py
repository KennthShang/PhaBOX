import networkx as nx
import pandas as pd
import pickle as pkl
import numpy as np
import argparse
import subprocess
from scripts.ulity import *
from scripts.preprocessing import *
import math
import os
import pandas as pd


def drop_network(tool, rootpth, midfolder, db_dir, outpth):
    if tool == 'phagcn':
        idx2label = {0: 'Autographiviridae', 1: 'Straboviridae', 2: 'Herelleviridae', 3: 'Drexlerviridae', 4: 'Demerecviridae', 5: 'Peduoviridae', 6: 'Casjensviridae', 7: 'Schitoviridae', 8: 'Kyanoviridae', 9: 'Ackermannviridae', 10: 'Rountreeviridae', 11: 'Salasmaviridae', 12: 'Vilmaviridae', 13: 'Zierdtviridae', 14: 'Mesyanzhinovviridae', 15: 'Chaseviridae', 16: 'Zobellviridae', 17: 'Orlajensenviridae', 18: 'Guelinviridae', 19: 'Steigviridae', 20: 'Duneviridae', 21: 'Pachyviridae', 22: 'Winoviridae', 23: 'Assiduviridae', 24: 'Suoliviridae', 25: 'Naomviridae', 26: 'Intestiviridae', 27: 'Crevaviridae', 28: 'Pervagoviridae'}

        network_df = pd.read_csv(f'{rootpth}/{midfolder}/phagcn_graph.csv')
        pred_df = pd.read_csv(f'{rootpth}/{midfolder}/phagcn_prediction.csv')
        label_df = pd.read_csv(f'{db_dir}/phagcn_taxonomic_label.csv')
        name2contig = {contig: acc for contig, acc in zip(pred_df['idx'].values, pred_df['Accession'])}

        node_set = list(set(network_df['Source'].values) | set(network_df['Target'].values))
        node2id = {item: idx for idx, item in enumerate(node_set)}
        node2label = {}
        for node in node2id.keys():
            try:
                label = pred_df[pred_df['idx'] == node]['Pred'].values[0]
            except:
                label = idx2label[label_df[label_df['contig_id'] == node]['family'].values[0]]
            node2label[node] = label

        Sources = [node2id[item] for item in network_df['Source'].values]
        Targets = [node2id[item] for item in network_df['Target'].values]
        Weight  = [1 for i in Sources]
        Type    = ['Undirected' for i in Sources]
        edge_df = pd.DataFrame({'Source': Sources, 'Target': Targets, 'Type': Type, 'Weight': Weight}) 
        rename_node = [name2contig[node] if "PhaGCN" in node else node.split('~')[0] for node in node2id.keys()]
        node_df = pd.DataFrame({"Id": node2id.values(), "Label": rename_node, "Category": node2label.values()})

        edge_df.to_csv(f'{rootpth}/{outpth}/phagcn_edge.csv', index=False)
        node_df.to_csv(f'{rootpth}/{outpth}/phagcn_node.csv', index=False)


    if tool == 'cherry':
        network_df = pd.read_csv(f'{rootpth}/{midfolder}/cherry_graph.csv')
        pred_df = pd.read_csv(f'{rootpth}/{midfolder}/cherry_prediction.csv')
        bacteria_df = pd.read_csv(f'{db_dir}/cherry/prokaryote.csv')
        virus_df = pd.read_csv(f'{db_dir}/cherry/virus.csv')

        name2contig = {contig: acc for contig, acc in zip(pred_df['idx'].values, pred_df['Accession'])}
        node_set = list(set(network_df['Source'].values) | set(network_df['Target'].values))
        node2label = {}
        for node in node_set:
            try:
                label = pred_df[pred_df['idx'] == node]['Top_1_label'].values[0]
            except:
                try:
                    label = bacteria_df[bacteria_df['Accession'] == node]['Species'].values[0]
                except:
                    try:
                        label = virus_df[virus_df['Accession'] == node]['Species'].values[0]
                    except:
                        continue
            node2label[node] = label

        node2id = {item: idx for idx, item in enumerate(node_set) if item in node2label}

        Sources = [node2id[item] for item in network_df['Source'].values if item in node2id]
        Targets = [node2id[item] for item in network_df['Target'].values if item in node2id]
        Weight  = [1 for i in Sources]
        Type    = ['Undirected' for i in Sources]
        edge_df = pd.DataFrame({'Source': Sources, 'Target': Targets, 'Type': Type, 'Weight': Weight}) 
        rename_node = [name2contig[node] if "PhaGCN" in node else node.split('~')[0] for node in node2id.keys()]
        node_df = pd.DataFrame({"Id": node2id.values(), "Label": rename_node, "Category": node2label.values()})

        edge_df.to_csv(f'{rootpth}/{outpth}/cherry_edge.csv', index=False)
        node_df.to_csv(f'{rootpth}/{outpth}/cherry_node.csv', index=False)








        


def draw_network(tool, midfolder, htmlpth, rootpth, db_dir):

    check_path(os.path.join(htmlpth, ''))
    check_path(os.path.join(htmlpth, f'html_{tool}'))

    if tool == 'phagcn':
        idx2label = {0: 'Autographiviridae', 1: 'Straboviridae', 2: 'Herelleviridae', 3: 'Drexlerviridae', 4: 'Demerecviridae', 5: 'Peduoviridae', 6: 'Casjensviridae', 7: 'Schitoviridae', 8: 'Kyanoviridae', 9: 'Ackermannviridae', 10: 'Rountreeviridae', 11: 'Salasmaviridae', 12: 'Vilmaviridae', 13: 'Zierdtviridae', 14: 'Mesyanzhinovviridae', 15: 'Chaseviridae', 16: 'Zobellviridae', 17: 'Orlajensenviridae', 18: 'Guelinviridae', 19: 'Steigviridae', 20: 'Duneviridae', 21: 'Pachyviridae', 22: 'Winoviridae', 23: 'Assiduviridae', 24: 'Suoliviridae', 25: 'Naomviridae', 26: 'Intestiviridae', 27: 'Crevaviridae', 28: 'Pervagoviridae'}

        network_df = pd.read_csv(f'{rootpth}/{midfolder}/phagcn_graph.csv')
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
            rename_node = [name2contig[node] if "PhaGCN" in node else node.split('~')[0] for node in node2id.keys()]
            node_df = pd.DataFrame({"Id": node2id.values(), "Label": rename_node, "Category": node2label.values(), "Size":Size})

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
            if pred_df[pred_df['idx'] == source]['Type'].values[0] == 'CRISPR':
                continue
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
            node_df = pd.DataFrame({"Id": node2id.values(), "Label": rename_node, "Category": node2label.values(), "Size":Size})

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
                f_in = f_in.replace('¶', '')
                f_out.write(f_in)


def draw_prophage(query, visualpath, figurepath):
    check_path(os.path.join(figurepath, ''))
    check_path(os.path.join(figurepath, 'html_prophage'))
    df_contig = pd.read_csv(os.path.join(visualpath, 'prophage/contigtable.csv'))
    length = df_contig[df_contig['Accession'] == query]['Length'].values[0]
    records = []
    df = pd.read_csv(os.path.join(visualpath, f'prophage/contigs/{query}_proteintable.csv'))
    for idx, start, end in zip(df['ID'], df['Pos_start'], df['Pos_end']):
        records.append([idx, start, end])

    interval1 = math.ceil(length*5/15)
    interval2 = math.ceil(length/15)
    f = open(os.path.join(figurepath, f'html_prophage/{query}_pic.html'), 'w')
    f.write(
        '<html><head><script src="angularplasmid.complete.min.js"></script></head><body>'
        '<style>@import url("http://fonts.googleapis.com/css?family=Lato:300,400,700|Donegal+One|Source+Code+Pro:400");'
        'body {font-family: "Lato";font-weight:400;}'
        '.boundary {stroke-dasharray:2,2;stroke-width:1px}'
        '.mdlabel {font-size:20px}'
        '.smlabel {font-size:8px}'
        '.black {fill:#000}'
        '.white {fill:#fff}'
        '.red {fill:rgb(192,64,64)}'
        '.purple {fill:rgb(192,64,192)}'
        '.blue {fill:rgb(64,128,256)}'
        '.green {fill:rgb(64,192,64)}'
        '.gold {fill:rgb(192,128,64)}</style>'
    )
    f.write(
        f'<div align="center"><plasmid sequencelength="{length}" plasmidheight="600" plasmidwidth="600">'
        f'<plasmidtrack trackstyle="fill:#ccc" width="5" radius="210"></plasmidtrack>'
        f'<plasmidtrack trackstyle="fill:rgba(225,225,225,0.5)" radius="200">'
        f'<tracklabel text="{query}" labelstyle="font-size:20px;font-weight:400"></tracklabel>'
        f'<tracklabel text="{length} bp" labelstyle="font-size:10px" vadjust="20"></tracklabel>'
        f'<trackscale interval="{interval1}" style="stroke:#999" direction="in" ticksize="3"></trackscale>'
        f'<trackscale interval="{interval1}" style=“stroke:#999” ticksize="3"></trackscale>'
        f'<trackscale interval="{interval2}" style="stroke:#f00" direction="in" showlabels="1" labelstyle="fill:#999;stroke:none;text-anchor:middle;alignment-baseline:middle;font-size:10px"></trackscale>'
    )

    height_array = [40, 60]
    color_array = ['fill:rgba(170,0,85,0.6)', 'fill:rgba(0,85,170,0.6)']
    font_color_array = ['red', 'blue']
    for idx, record in enumerate(records):
        acc = record[0]
        start = record[1]
        end = record[2]
        f.write(f'<trackmarker start="{start}" end="{end}" markerstyle={color_array[idx%2]}><markerlabel class="mdlabel {font_color_array[idx%2]}" text="{acc}" vadjust="{height_array[idx%2]}" style="font-size:10px;font-weight:800;"></markerlabel></trackmarker>')
        #f.write(f'<trackmarker start="{start}" end="{end}" markerstyle="fill:rgba(255,221,238,0.6)" wadjust="-5" vadjust="25"></trackmarker>')
        #f.write(f'<trackmarker start="{start}" markerstyle="stroke:rgba(128,64,64,0.8)" class="boundary" wadjust="20"></trackmarker>')
        #f.write(f'<trackmarker start="{end}" markerstyle="stroke:rgba(128,64,64,0.8)" class="boundary" wadjust="20"></trackmarker>')
        f.write(f'<trackmarker start="{(end+start)/2}" markerstyle="stroke:rgba(128,64,64,0.8)" class="boundary" wadjust="{height_array[idx%2]-20}"></trackmarker>')

    f.write(
        '</plasmidtrack></plasmid></div></body><html>'
    )



