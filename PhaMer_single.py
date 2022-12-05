#!/usr/bin/env python
import os
import shutil
import argparse
import subprocess
import random
import pandas as pd
import numpy as np
import pickle as pkl
import scipy as sp
import networkx as nx
import scipy.stats as stats
import scipy.sparse as sparse

from torch import nn
from torch import optim
from torch.nn import functional as F

from scripts.ulity import *
from scripts.preprocessing import *
from scripts.cnnscript import *
from shutil import which
from collections import Counter
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Blast.Applications import NcbiblastnCommandline
from models.phamer import Transformer
from models.CAPCNN import WCNN
from models.PhaGCN import GCN
from models import Cherry
from scipy.special import softmax
from scripts.data import load_data, preprocess_features, preprocess_adj, sample_mask






parser = argparse.ArgumentParser(description="""Main script of PhaSUIT.""")
parser.add_argument('--contigs', help='FASTA file of contigs',  default = 'inputs.fa')
parser.add_argument('--threads', help='number of threads to use', type=int, default=4)
parser.add_argument('--len', help='minimum length of contigs', type=int, default=3000)
parser.add_argument('--reject', help='threshold to reject prophage',  type=float, default = 0.2)
parser.add_argument('--rootpth', help='rootpth of the user', default='user_0/')
parser.add_argument('--out', help='output path of the user', default='out/')
parser.add_argument('--midfolder', help='mid folder for intermidiate files', default='midfolder/')
parser.add_argument('--visual', help='mid folder for intermidiate files', default='visual/')
parser.add_argument('--dbdir', help='database directory',  default = 'database/')
parser.add_argument('--parampth', help='path of parameters',  default = 'parameters/')
parser.add_argument('--proteins', help='FASTA file of predicted proteins (optional)')
parser.add_argument('--topk', help='Top k prediction',  type=int, default=1)
parser.add_argument('--html', help='mid folder for intermidiate files', default='state/')
parser.add_argument('--flag', help='for visualization', default=1)
inputs = parser.parse_args()


contigs   = inputs.contigs
midfolder = inputs.midfolder
rootpth   = inputs.rootpth
db_dir    = inputs.dbdir
out_dir   = inputs.out
parampth  = inputs.parampth
threads   = inputs.threads
visual    = inputs.visual
length    = inputs.len
html      = inputs.html

if not os.path.exists(db_dir):
    print(f'Database directory {db_dir} missing or unreadable')
    exit(1)

check_path(os.path.join(rootpth, out_dir))
check_path(os.path.join(rootpth, midfolder))
check_path(os.path.join(rootpth, visual))
check_path(os.path.join(rootpth, visual, 'xml'))
check_path(os.path.join(rootpth, visual, 'contigs'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == 'cpu':
    print("running with cpu")
    torch.set_num_threads(inputs.threads)

###############################################################
#######################  Filter length ########################
###############################################################

rec = []
ID2length = {}
for record in SeqIO.parse(os.path.join(rootpth, contigs), 'fasta'):
    ID2length[record.id] = len(record.seq)
    if len(record.seq) > inputs.len:
        rec.append(record)
        with open(f'{rootpth}/{visual}/contigs/{record.id}.txt', 'w') as file:
            file.write(str(record.seq))

if not rec:
    with open(f'{rootpth}/{visual}/non_alignment_flag.txt', 'w') as file_out:
        file_out.write('non_alignment_flag\n')
    with open(f'{rootpth}/{out_dir}/phamer_prediction.csv', 'w') as file_out:
        file_out.write("Accession,Pred,Score\n")
        for record in SeqIO.parse(os.path.join(rootpth, contigs), 'fasta'):
            file_out.write(f'{record.id},filtered,0\n')
    exit()

SeqIO.write(rec, f'{rootpth}/filtered_contigs.fa', 'fasta')


###############################################################
##########################  PhaMer ############################
###############################################################

translation(rootpth, os.path.join(rootpth, midfolder), 'filtered_contigs.fa', 'test_protein.fa', threads, inputs.proteins)
run_diamond(f'{db_dir}/phamer_database.dmnd', os.path.join(rootpth, midfolder), 'test_protein.fa', 'phamer', threads)
convert_xml(os.path.join(rootpth, midfolder), 'phamer')
if os.path.getsize(f'{rootpth}/{midfolder}/phamer_results.abc') == 0:
    with open(f'{rootpth}/{visual}/non_alignment_flag.txt', 'w') as file_out:
        file_out.write('non_alignment_flag\n')
    with open(f'{rootpth}/{out_dir}/phamer_prediction.csv', 'w') as file_out:
        file_out.write("Accession,Pred,Score\n")
        for record in SeqIO.parse(os.path.join(rootpth, contigs), 'fasta'):
            if len(record.seq) > inputs.len:
                file_out.write(f'{record.id},non-phage,0\n')
            else:
                file_out.write(f'{record.id},filtered,0\n')
    exit()
contig2sentence(db_dir, os.path.join(rootpth, midfolder), 'test_protein.fa', 'phamer')





pcs2idx = pkl.load(open(f'{rootpth}/{midfolder}/phamer_pc2wordsid.dict', 'rb'))
num_pcs = len(set(pcs2idx.keys()))


src_pad_idx = 0
src_vocab_size = num_pcs+1


model, optimizer, loss_func = reset_model(Transformer, src_vocab_size, device)
try:
    pretrained_dict=torch.load(f'{parampth}/transformer.pth', map_location=device)
    model.load_state_dict(pretrained_dict)
except:
    print('cannot find pre-trained model')
    exit(1)

sentence   = pkl.load(open(f'{rootpth}/{midfolder}/phamer_sentence.feat', 'rb'))
id2contig  = pkl.load(open(f'{rootpth}/{midfolder}/phamer_sentence_id2contig.dict', 'rb'))
proportion = pkl.load(open(f'{rootpth}/{midfolder}/phamer_sentence_proportion.feat', 'rb'))
contig2id  = {item: key for key, item in id2contig.items()}

all_pred = []
all_score = []
with torch.no_grad():
    _ = model.eval()
    for idx in range(0, len(sentence), 500):
        try:
            batch_x = sentence[idx: idx+500]
            weight  = proportion[idx: idx+500]
        except:
            batch_x = sentence[idx:]
            weight  = proportion[idx:]
        batch_x = return_tensor(batch_x, device).long()
        logit = model(batch_x)
        logit = torch.sigmoid(logit.squeeze(1))
        logit = reject_prophage(logit, weight, inputs.reject)
        pred  = ['phage' if item > 0.5 else 'non-phage' for item in logit]
        all_pred += pred
        all_score += [float('{:.3f}'.format(i)) for i in logit]


#FLAGS
if len(set(all_pred)) == 1 and all_pred[0] == 'non-phage':
    with open(f'{rootpth}/{out_dir}/phamer_prediction.csv', 'w') as file_out:
        file_out.write("Accession,Pred,Score\n")
        for record in SeqIO.parse(os.path.join(rootpth, contigs), 'fasta'):
            file_out.write(f'{record.id},non-phage,0\n')
    exit()   

### Add filtered label (Nov. 8th)
contigs_list = list(id2contig.values())
contigs_add = []
pred_rec = []
seq_dict = {}
phage_rec = []
for record in SeqIO.parse(f'{rootpth}/{contigs}', 'fasta'):
    seq_dict[record.id] = str(record.seq)
    if record.id not in contigs_list:
        if len(record.seq) < inputs.len:
            contigs_add.append(record.id)
            all_pred.append('filtered')
            all_score.append(0)
            continue
        contigs_add.append(record.id)
        all_pred.append('non-phage')
        all_score.append(0)



contigs_list += contigs_add
length_list = [ID2length[item] for item in contigs_list]

pred_csv = pd.DataFrame({"Accession":contigs_list, "Length":length_list, "Pred":all_pred, "Score":all_score})
pred_csv.to_csv(f'{rootpth}/{out_dir}/phamer_prediction.csv', index = False)
phage_list = pred_csv[pred_csv['Pred'] == 'phage']['Accession'].values

for record in SeqIO.parse(f'{rootpth}/{contigs}', 'fasta'):
    if record.id in phage_list:
        phage_rec.append(record)

SeqIO.write(phage_rec, f'{rootpth}/predicted_phage.fa', 'fasta')

if inputs.flag == 0:
    exit()

# For visulization

### Pie Chart (Nov. 8th)
cnt = Counter(pred_csv['Pred'].values)
pred_dict = {}
for key, value in zip(cnt.keys(), cnt.values()):
    pred_dict[key] = value

pkl.dump(pred_dict, open(f"{rootpth}/{visual}/phamer_pred.dict", 'wb'))


### BLASTP result (Nov. 8th)


phage_contig = pred_csv[pred_csv['Pred']=='phage']['Accession'].values
phage_contig_length = pred_csv[pred_csv['Pred']=='phage']['Length'].values
phamer_score = [f'{item:.3f}' for item in pred_csv[pred_csv['Pred']=='phage']['Score'].values]


# dump contigtable.csv (Nov. 13th)
all_Seq = [seq_dict[item] for item in phage_contig]
df = pd.DataFrame({"ID": [item+1 for item in range(len(phage_contig))], "Accession": phage_contig, "Length":phage_contig_length, "PhaMer": ['phage']*len(phage_contig), "PhaMer_score": phamer_score})
df.to_csv(f'{rootpth}/{visual}/contigtable.csv', index=False)


# recruit the predicted contigs (Nov. 13th)
blast_df = pd.read_csv(f"{rootpth}/{midfolder}/phamer_results.abc", sep=' ', names=['query', 'ref', 'evalue'])
protein2id = {protein:idx for idx, protein in enumerate(sorted(list(set(blast_df['query'].values))))}
id2protein = {idx:protein for idx, protein in enumerate(sorted(list(set(blast_df['query'].values))))}
contigs_list = [protein.rsplit("_", 1)[0] for protein in protein2id.keys()]
xml_files = parse_xml(protein2id, f'{rootpth}/{midfolder}', 'phamer')
protein2evalue = parse_evalue(blast_df, f'{rootpth}/{midfolder}', 'phamer')
protein2start, protein2end = parse_position(f'{rootpth}/{midfolder}')
position_start = [protein2start[item] for item in protein2id.keys()]
position_end = [protein2end[item] for item in protein2id.keys()]
evalue_list = [protein2evalue[item] for item in protein2id.keys()]
xml_df = pd.DataFrame({"Pos_start": position_start, "Pos_end": position_end, "Accession": contigs_list, "Protein_id": protein2id.keys(), "evalue": evalue_list, "xml":xml_files})



##### for phage-only proteintable #####
df_list = []
for item in phage_contig:
    df_list.append(xml_df[xml_df['Accession']==item])

df = pd.concat(df_list)

button_str = '<button type="button" class="btn btn-outline-primary btn-sm" data-bs-toggle="modal" data-bs-target="#exampleModalToggle2" data-bs-whatever="pengcheng">Visualize</button>'
# resorted the protein_id (Nov. 13th)
sorted_df_list = []
for contig in set(df['Accession'].values):
    tmp_df = df[df['Accession'] == contig].reset_index()
    proteins = tmp_df['Protein_id'].values
    ori_protein_idx = {item:idx for idx, item in enumerate(proteins)}
    sorted_idx = sorted([int(item.rsplit('_', 1)[1]) for item in proteins])
    new_protein_idx = [ori_protein_idx[f'{contig}_{item}'] for item in sorted_idx]
    tmp_df = tmp_df.loc[np.array(new_protein_idx)]
    ID = [item+1 for item in range(len(tmp_df))]
    tmp_df['ID'] = ID
    button_list = []
    # dump xml
    for protein, xml in zip(tmp_df['Protein_id'].values, tmp_df['xml'].values):
        with open(f'{rootpth}/{visual}/xml/{protein}_xml.txt', 'w') as file:
            file.write(xml)
        button_list.append(button_str.replace('pengcheng', protein))
    # dump single contig csv
    tmp_df = tmp_df.drop(columns=['xml'])
    tmp_df['button'] = button_list
    order = ['ID', 'Accession', 'Protein_id', 'Pos_start', 'Pos_end', 'evalue', 'button']
    tmp_df = tmp_df[order]
    tmp_df.to_csv(f'{rootpth}/{visual}/contigs/{contig}_proteintable.csv', index=False)
    sorted_df_list.append(tmp_df)


df = pd.concat(sorted_df_list)
df.to_csv(f'{rootpth}/{visual}/proteintable.csv', index=False) 


##### for phage-only results #####
rec = []
for record in SeqIO.parse(f'{rootpth}/{midfolder}/test_protein.fa', 'fasta'):
    try:
        protein2evalue[record.id]
        rec.append(record)
    except:
        pass 
SeqIO.write(rec, f'{rootpth}/{out_dir}/significant_proteins.fa', 'fasta')
os.system(f"cp {rootpth}/{midfolder}/phamer_results.tab {rootpth}/{out_dir}/blast_results.tab")
os.system(f"sed -i '1i\qseqid\tsseqid\tpident\tlength\tmismatch\tgapopen\tqstart\tqend\tsstart\tsend\tevalue' {rootpth}/{out_dir}/blast_results.tab")






