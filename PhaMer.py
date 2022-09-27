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
parser.add_argument('--contigs', help='FASTA file of contigs',  default = 'test_contigs.fa')
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
inputs = parser.parse_args()


contigs   = inputs.contigs
midfolder = inputs.midfolder
rootpth   = inputs.rootpth
db_dir    = inputs.dbdir
out_dir   = inputs.out
parampth  = inputs.parampth
threads   = inputs.threads

if not os.path.exists(db_dir):
    print(f'Database directory {db_dir} missing or unreadable')
    exit(1)

check_path(os.path.join(rootpth, out_dir))
check_path(os.path.join(rootpth, midfolder))


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
##########################  PhaMer ############################
###############################################################

translation(rootpth, os.path.join(rootpth, midfolder), 'filtered_contigs.fa', 'test_protein.fa', threads, inputs.proteins)
run_diamond(f'{db_dir}/phamer_database.dmnd', os.path.join(rootpth, midfolder), 'test_protein.fa', 'phamer', threads)
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


pred_csv = pd.DataFrame({"Contig":id2contig.values(), "Pred":all_pred, "Score":all_score})
pred_csv.to_csv(f'{rootpth}/{out_dir}/phamer_prediction.csv', index = False)