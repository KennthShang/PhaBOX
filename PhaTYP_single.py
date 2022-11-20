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
parser.add_argument('--html', help='mid folder for intermidiate files', default='state/')
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

if not os.path.exists(db_dir):
    print(f'Database directory {db_dir} missing or unreadable')
    exit(1)

check_path(os.path.join(rootpth, out_dir))
check_path(os.path.join(rootpth, midfolder))
check_path(os.path.join(rootpth, visual))


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
SeqIO.write(rec, f'{rootpth}/filtered_contigs.fa', 'fasta')


###############################################################
##########################  PhaTYP ############################
###############################################################

translation(rootpth, os.path.join(rootpth, midfolder), 'filtered_contigs.fa', 'test_protein.fa', threads, inputs.proteins)
run_diamond(f'{db_dir}/phamer_database.dmnd', os.path.join(rootpth, midfolder), 'test_protein.fa', 'phatyp', threads)
convert_xml(os.path.join(rootpth, midfolder), 'phatyp')
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
