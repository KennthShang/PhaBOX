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
parser.add_argument('--threads', help='number of threads to use', type=int, default=8)
parser.add_argument('--len', help='minimum length of contigs', type=int, default=3000)
parser.add_argument('--reject', help='threshold to reject prophage',  type=float, default = 0.2)
parser.add_argument('--rootpth', help='rootpth of the user', default='user_0/')
parser.add_argument('--out', help='output path of the user', default='out/')
parser.add_argument('--midfolder', help='mid folder for intermidiate files', default='midfolder/')
parser.add_argument('--dbdir', help='database directory',  default = 'database/')
parser.add_argument('--parampth', help='path of parameters',  default = 'parameters/')
parser.add_argument('--proteins', help='FASTA file of predicted proteins (optional)')
parser.add_argument('--topk', help='Top k prediction',  type=int, default=1)
parser.add_argument('--visual', help='mid folder for intermidiate files', default='visual/')
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

if not os.path.isfile(os.path.join(rootpth, contigs)):
    exit()

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
for record in SeqIO.parse(os.path.join(rootpth, contigs), 'fasta'):
    if len(record.seq) > inputs.len:
        rec.append(record)
        with open(f'{rootpth}/{visual}/contigs/{record.id}.txt', 'w') as file:
            file.write(str(record.seq))

if not rec:
    with open(f'{rootpth}/{visual}/non_alignment_flag.txt', 'w') as file_out:
        file_out.write('non_alignment_flag\n')
    exit()
SeqIO.write(rec, f'{rootpth}/filtered_contigs.fa', 'fasta')


###############################################################
##########################  PhaTYP ############################
###############################################################

translation(rootpth, os.path.join(rootpth, midfolder), 'filtered_contigs.fa', 'test_protein.fa', threads, inputs.proteins)
run_diamond(f'{db_dir}/phamer_database.dmnd', os.path.join(rootpth, midfolder), 'test_protein.fa', 'phatyp', threads)
convert_xml(os.path.join(rootpth, midfolder), 'phatyp')
if os.path.getsize(f'{rootpth}/{midfolder}/phatyp_results.abc') == 0:
    with open(f'{rootpth}/{visual}/non_alignment_flag.txt', 'w') as file_out:
        file_out.write('non_alignment_flag\n')
    exit()

contig2sentence(db_dir, os.path.join(rootpth, midfolder), 'test_protein.fa', 'phatyp')
generate_bert_input(db_dir, os.path.join(rootpth, midfolder), os.path.join(rootpth, midfolder), 'phatyp')

id2contig  = pkl.load(open(f'{rootpth}/{midfolder}/phatyp_sentence_id2contig.dict', 'rb'))

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


bert_feat = pd.read_csv(f'{rootpth}/{midfolder}/phatyp_bert_feat.csv')
test  = pa.Table.from_pandas(bert_feat)
test  = datasets.Dataset(test)
data = datasets.DatasetDict({"test": test})

tokenizer = BertTokenizer.from_pretrained(f'{parampth}/bert_config', do_basic_tokenize=False)
tokenized_data= data.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = init_bert(f'{rootpth}/{midfolder}', bert_feat, os.path.join(parampth, "bert"), tokenizer, tokenized_data, data_collator)

with torch.no_grad():
    pred, label, metric = trainer.predict(tokenized_data["test"])


prediction_value = []
for item in pred:
    prediction_value.append(softmax(item))
prediction_value = np.array(prediction_value)



all_pred = []
all_score = []
for score in prediction_value:
    pred = np.argmax(score)
    if pred == 1:
        all_pred.append('temperate')
        all_score.append(score[1])
    else:
        all_pred.append('virulent')
        all_score.append(score[0])

### Add filtered label (Nov. 8th)
contigs_list = list(id2contig.values())
contigs_add = []
length_dict = {}
seq_dict = {}
for record in SeqIO.parse(f'{rootpth}/{contigs}', 'fasta'):
    seq_dict[record.id] = str(record.seq)
    length_dict[record.id] = len(record.seq)
    if record.id not in contigs_list:
        if len(record.seq) < inputs.len:
            contigs_add.append(record.id)
            all_pred.append('filtered')
            all_score.append(0)
            continue
        contigs_add.append(record.id)
        all_pred.append('unpredicted')
        all_score.append(0)

contigs_list += contigs_add
length_list = [length_dict[item] for item in contigs_list]

pred_csv = pd.DataFrame({"Accession":contigs_list, "Length":length_list, "Pred":all_pred, "Score":all_score})
pred_csv.to_csv(f'{rootpth}/{out_dir}/phatyp_prediction.csv', index = False)

### Pie Chart (Nov. 8th)
cnt = Counter(pred_csv['Pred'].values)
pred_dict = {}
for key, value in zip(cnt.keys(), cnt.values()):
    pred_dict[key] = value

pkl.dump(pred_dict, open(f"{rootpth}/{visual}/phatyp_pred.dict", 'wb'))



### BLASTP result (Nov. 8th)
phage_contig = [item for item in contigs_list if item != 'filtered' and item != 'unpredicted' ]
phage_contig_length = [pred_csv[pred_csv['Accession'] == item]['Length'].values[0] for item in phage_contig]
phage_contig_pred = [pred_csv[pred_csv['Accession'] == item]['Pred'].values[0] for item in phage_contig]
phage_contig_score = [pred_csv[pred_csv['Accession'] == item]['Score'].values[0] for item in phage_contig]
phage_contig_score = [f'{item:.3f}' for item in phage_contig_score]

# dump contigtable.csv (Nov. 13th)
all_Seq = [seq_dict[item] for item in phage_contig]
df = pd.DataFrame({"ID": [item+1 for item in range(len(phage_contig))], "Accession": phage_contig, "Length":phage_contig_length, "PhaTYP":phage_contig_pred, "PhaTYP_score":phage_contig_score})
df.to_csv(f'{rootpth}/{visual}/contigtable.csv', index=False)

blast_df = pd.read_csv(f"{rootpth}/{midfolder}/phatyp_results.abc", sep=' ', names=['query', 'ref', 'evalue'])
protein2id = {protein:idx for idx, protein in enumerate(sorted(list(set(blast_df['query'].values))))}
contigs_list = [protein.rsplit("_", 1)[0] for protein in protein2id.keys()]

xml_files = parse_xml(protein2id, f'{rootpth}/{midfolder}', 'phatyp')
protein2evalue = parse_evalue(blast_df, f'{rootpth}/{midfolder}', 'phatyp')
protein2start, protein2end = parse_position(f'{rootpth}/{midfolder}')
position_start = [protein2start[item] for item in protein2id.keys()]
position_end = [protein2end[item] for item in protein2id.keys()]



evalue_list = [protein2evalue[item] for item in protein2id.keys()]
df = pd.DataFrame({"Pos_start": position_start, "Pos_end": position_end, "Accession": contigs_list, "Protein_id": protein2id.keys(), "evalue": evalue_list, "xml":xml_files})
df_list = []
for item in phage_contig:
    df_list.append(df[df['Accession']==item])

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


rec = []
for record in SeqIO.parse(f'{rootpth}/{midfolder}/test_protein.fa', 'fasta'):
    try:
        protein2evalue[record.id]
        rec.append(record)
    except:
        pass 
SeqIO.write(rec, f'{rootpth}/{out_dir}/significant_proteins.fa', 'fasta')
os.system(f"cp {rootpth}/{midfolder}/phatyp_results.tab {rootpth}/{out_dir}/blast_results.tab")
os.system(f"sed -i '1i\qseqid\tsseqid\tpident\tlength\tmismatch\tgapopen\tqstart\tqend\tsstart\tsend\tevalue' {rootpth}/{out_dir}/blast_results.tab")



