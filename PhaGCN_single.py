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
from draw import draw_network, drop_network
from scipy.special import softmax
from scripts.data import load_data, preprocess_features, preprocess_adj, sample_mask

from torch import nn
from torch import optim
from torch.nn import functional as F





parser = argparse.ArgumentParser(description="""Main script of PhaSUIT.""")
parser.add_argument('--contigs', help='FASTA file of contigs',  default = 'inputs.fa')
parser.add_argument('--threads', help='number of threads to use', type=int, default=8)
parser.add_argument('--len', help='minimum length of contigs', type=int, default=3000)
parser.add_argument('--reject', help='threshold to reject prophage',  type=float, default = 0.2)
parser.add_argument('--rootpth', help='rootpth of the user', default='user_0/')
parser.add_argument('--out', help='output path of the user', default='out/')
parser.add_argument('--midfolder', help='mid folder for intermidiate files', default='midfolder/')
parser.add_argument('--dbdir', help='database directory',  default = 'database/')
parser.add_argument('--scriptpth', help='path of parameters',  default = 'scripts/')
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
length    = inputs.len
scriptpth = inputs.scriptpth


if not os.path.isfile(contigs):
    exit()
if not os.path.exists(db_dir):
    print(f'Database directory {db_dir} missing or unreadable')
    exit(1)

check_path(os.path.join(rootpth, out_dir))
check_path(os.path.join(rootpth, midfolder))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("running with cpu")
    torch.set_num_threads(inputs.threads)



###############################################################
#######################  Filter length ########################
###############################################################



rec = []
ID2length = {}
for record in SeqIO.parse(contigs, 'fasta'):
    if len(record.seq) > inputs.len:
        rec.append(record)
        ID2length[record.id] = len(record.seq)

# FLAGS
if not rec:
    exit()
SeqIO.write(rec, f'{rootpth}/filtered_contigs.fa', 'fasta')

###############################################################
##################  Filter unknown family #####################
###############################################################

query_file = f"{rootpth}/filtered_contigs.fa"
db_virus_prefix = f"{db_dir}/unknown_db/db"
output_file = f"{rootpth}/{midfolder}/unknown_out.tab"
virus_call = NcbiblastnCommandline(query=query_file,db=db_virus_prefix,out=output_file,outfmt="6 qseqid sseqid evalue pident length qlen", evalue=1e-10,
                                 task='megablast',perc_identity=95,num_threads=threads)
virus_call()


check_unknown = {}
check_unknown_all = {}
check_unknown_all_score = {}
with open(output_file) as file_out:
    for line in file_out.readlines():
        parse = line.replace("\n", "").split("\t")
        virus = parse[0]
        target = parse[1]
        target = target.split('|')[1]
        ident  = float(parse[-3])
        length = float(parse[-2])
        qlen   = float(parse[-1])
        if length/qlen > 0.95 and ident > 0.95:
            check_unknown[virus] = target
        if virus not in check_unknown_all:
            ident  = float(parse[-3])/100
            ident  = float(f"{ident:.3f}")
            check_unknown_all[virus] = target
            check_unknown_all_score[virus] = ident

rec = []
for record in SeqIO.parse(f'{rootpth}/filtered_contigs.fa', 'fasta'):
    try:
        if check_unknown[record.id]:
            continue
    except:
        rec.append(record)

SeqIO.write(rec, f'{rootpth}/filtered_phagcn_contigs.fa', 'fasta')

if not rec:
    phagcn_exception_no_visual(rootpth, midfolder, out_dir, ID2length, inputs)

    exit()


###############################################################
########################## PhaGCN  ############################
###############################################################

translation(rootpth, os.path.join(rootpth, midfolder), 'filtered_phagcn_contigs.fa', 'test_protein.fa', threads, inputs.proteins)
nucl, protein = recruit_phage_file(rootpth, midfolder, None, 'filtered_contigs.fa')

SeqIO.write(nucl, f'{rootpth}/checked_phage_contigs.fa',"fasta")
SeqIO.write(protein, f'{rootpth}/checked_phage_protein.fa',"fasta")

single_pth   = os.path.join(rootpth, "CNN_temp/single")
cnninput_pth = os.path.join(rootpth, "CNN_temp/input")
phagcninput_pth = os.path.join(rootpth, midfolder, "phgcn/")
check_path(single_pth)
check_path(cnninput_pth)
check_path(phagcninput_pth)

contig2name = {}
with open(f"{rootpth}/{midfolder}/phagcn_name_list.csv",'w') as list_out:
    list_out.write("Contig,idx\n")
    for contig_id, record in enumerate(SeqIO.parse(f'{rootpth}/checked_phage_contigs.fa', "fasta")):
        name = f"PhaGCN_{str(contig_id)}"
        list_out.write(record.id + "," + name + "\n")
        contig2name[record.id] = name
        record.id = name
        _ = SeqIO.write(record, f"{single_pth}/{name}.fa", "fasta")

rename_rec = []
for record in SeqIO.parse(f'{rootpth}/checked_phage_protein.fa',"fasta"):
    old_name = record.id
    idx = old_name.rsplit('_', 1)[1]
    record.id = contig2name[old_name.rsplit('_', 1)[0]] +"_"+ idx
    rename_rec.append(record)

SeqIO.write(rename_rec, f'{rootpth}/{midfolder}/phagcn_renamed_protein.fa', 'fasta')



# sequence encoding using CNN
seq_dict = {}
for file in os.listdir(single_pth):
    rec = create_fragments(single_pth, file)
    seq_dict[file.split('.fa')[0]] = rec

int_to_vocab, vocab_to_int = return_kmer_vocab()
for seq in seq_dict:
    int_feature = encode(seq_dict[seq], vocab_to_int)
    inputs_feat = create_cnndataset(int_feature)
    np.savetxt(f"{cnninput_pth}/{seq}.csv", inputs_feat, delimiter=",", fmt='%d') 


cnn, embed = load_cnnmodel(parampth)
compress_feature = []
file_list = os.listdir(cnninput_pth)
file_list = sorted(file_list)

for name in file_list:
    val = np.genfromtxt(f'{cnninput_pth}/{name}', delimiter=',')
    val_label = val[:, -1]
    val_feature = val[:, :-1]
    # comvert format
    val_feature = torch.from_numpy(val_feature).long()
    val_feature = embed(val_feature)
    val_feature = val_feature.reshape(len(val_feature), 1, 1998, 100)
    # prediction
    out = cnn(val_feature)
    out = out.detach().numpy()  
    out = np.sum(out, axis=0)
    compress_feature.append(out)

compress_feature = np.array(compress_feature)
pkl.dump(compress_feature, open(f"{phagcninput_pth}/phagcn_contig.F", 'wb'))

# Generate knowledge graph
# add convertxml (Nov. 8th)
run_diamond(f'{db_dir}/phagcn_database.dmnd', os.path.join(rootpth, midfolder), 'phagcn_renamed_protein.fa', 'phagcn', threads)
convert_xml(os.path.join(rootpth, midfolder), 'phagcn', scriptpth)
if os.path.getsize(f'{rootpth}/{midfolder}/phagcn_results.abc') == 0:
    phagcn_exception_no_visual(rootpth, midfolder, out_dir, ID2length, inputs)
    exit()

abc_fp = f"{rootpth}/{midfolder}/merged.abc"
_ = subprocess.check_call(f"cat {db_dir}/phagcn_database.self-diamond.tab.abc {rootpth}/{midfolder}/phagcn_results.abc > {abc_fp}", shell=True)

# generate gene2genome
generate_gene2genome(os.path.join(rootpth, midfolder), os.path.join(rootpth, midfolder), 'phagcn', rootpth)

# Combining the gene-to-genomes files
_ = subprocess.check_call(f"cat {db_dir}/Caudovirales_gene_to_genomes.csv {rootpth}/{midfolder}/phagcn_contig_gene_to_genome.csv > {rootpth}/{midfolder}/phagcn_gene_to_genome.csv", shell=True)


# Running MCL
print("\n\n" + "{:-^80}".format("Protein clustering"))
print("Loading proteins...")
gene2genome_fp = f"{rootpth}/{midfolder}/phagcn_gene_to_genome.csv"
gene2genome_df = pd.read_csv(gene2genome_fp, sep=',', header=0)
pc_overlap, pc_penalty, pc_haircut, pc_inflation = 0.8, 2.0, 0.1, 2.0
pcs_fp = make_protein_clusters_mcl(abc_fp, os.path.join(rootpth, midfolder), pc_inflation)


print("Building the cluster and profiles (this may take some time...)")
protein_df, clusters_df, profiles_df, contigs_df = build_clusters(pcs_fp, gene2genome_df)

print("Saving files")
dfs = [gene2genome_df, contigs_df, clusters_df]
names = ['proteins', 'contigs', 'pcs']

for name, df in zip(names, dfs):
    fn = "Cyber_phagcn_{}.csv".format(name)
    fp = os.path.join(f'{rootpth}/{midfolder}', fn)
    index_id = name.strip('s') + '_id'
    df.set_index(index_id).to_csv(fp)



contigs_csv_df = contigs_df.copy()
contigs_csv_df['contig_id'] = contigs_csv_df['contig_id'].str.replace(' ', '~')
contigs_csv_df.index.name = "pos"
contigs_csv_df.reset_index(inplace=True)

pcs_csv_df = clusters_df.copy()
profiles = profiles_df.copy()
profiles['contig_id'] = profiles['contig_id'].str.replace(' ', '~')  # ClusterONE can't handle spaces

# Filtering the PC profiles that appears only once
before_filter = len(profiles)
cont_by_pc = profiles.groupby("pc_id").count().contig_id.reset_index()

# get the number of contigs for each pcs and add it to the dataframe
cont_by_pc.columns = ["pc_id", "nb_proteins"]
pcs_csv_df = pd.merge(pcs_csv_df, cont_by_pc, left_on="pc_id", right_on="pc_id", how="left")
pcs_csv_df.fillna({"nb_proteins": 0}, inplace=True)

# Drop the pcs that <= 1 contig from the profiles.
pcs_csv_df = pcs_csv_df[pcs_csv_df['nb_proteins'] > 1]  # .query("nb_contigs>1")
at_least_a_cont = cont_by_pc[cont_by_pc['nb_proteins'] > 1]  # cont_by_pc.query("nb_contigs>1")
profiles = profiles[profiles['pc_id'].isin(at_least_a_cont.pc_id)]

pcs_csv_df = pcs_csv_df.reset_index(drop=True)
pcs_csv_df.index.name = "pos"
pcs_csv_df = pcs_csv_df.reset_index()

matrix, singletons = build_pc_matrices(profiles, contigs_csv_df, pcs_csv_df)
profiles_csv = {"matrix": matrix, "singletons": singletons}
merged_df = contigs_csv_df


ntw = create_network(matrix, singletons, thres=1, max_sig=300)
fi = to_clusterer(ntw, f"{rootpth}/{midfolder}/phagcn_network.ntw", merged_df.copy())


print("\n\n" + "{:-^80}".format("Calculating E-edges"))

# loading database
gene2genome = pd.read_csv(f'{db_dir}/Caudovirales_gene_to_genomes.csv')
contig_id = gene2genome["contig_id"].values
contig_id = [item.replace(" ", "~") for item in contig_id]
gene2genome["contig_id"] = contig_id

protein_to_ref = {protein:ref for protein, ref in zip(gene2genome["protein_id"].values, gene2genome["contig_id"].values)}

contig_set = list(set(gene2genome["contig_id"].values))
ID_to_ref = {i:ref for i, ref in enumerate(contig_set)}
ref_to_ID = {ref:i for i, ref in enumerate(contig_set)}


contig_to_id = {}
file_list = os.listdir(single_pth)
file_list = sorted(file_list)
for file_n in file_list:
    name = file_n.split(".")[0]
    contig_to_id[name] = file_list.index(file_n)


# record the row id for each contigs
id_to_contig = {value: key for key, value in contig_to_id.items()}

blastp = pd.read_csv(f'{rootpth}/{midfolder}/phagcn_results.abc', sep=" ", names = ["contigs", "ref", "e-value"])
gene_to_genome = pd.read_csv(f"{rootpth}/{midfolder}/phagcn_contig_gene_to_genome.csv", sep=",")

e_matrix = np.ones((len(contig_to_id), len(ref_to_ID.keys())))
blast_contigs = blastp["contigs"].values
blast_ref = blastp["ref"].values
blast_value = blastp["e-value"].values
for i in range(len(blast_contigs)):
    contig_name = gene_to_genome[gene_to_genome["protein_id"] == blast_contigs[i]]["contig_id"].values
    contig_name = contig_name[0]
    row_id = contig_to_id[contig_name]
    reference = protein_to_ref[blast_ref[i]]
    col_id = ref_to_ID[reference]
    e_value = float(blast_value[i])
    if e_value == 0:
        e_value = 1e-250
    if e_matrix[row_id][col_id] == 1:
        e_matrix[row_id][col_id] = e_value
    else:
        e_matrix[row_id][col_id] += e_value

e_weight = -np.log10(e_matrix)-50
e_weight[e_weight < 1] = 0

print("\n\n" + "{:-^80}".format("Calculating P-edges"))

name_to_id = {}
reference_df = pd.read_csv(f"{db_dir}/phagcn_reference_name_id.csv")
tmp_ref = reference_df["name"].values
tmp_id  = reference_df["idx"].values
for ref, idx in zip(tmp_ref,tmp_id):
    name_to_id[ref.replace(" ", "~")] = idx



edges = pd.read_csv(f"{rootpth}/{midfolder}/phagcn_network.ntw", sep=' ', names=["node1", "node2", "weight"])
merged_df = pd.read_csv(f"{db_dir}/Caudovirales_genome_profile.csv", header=0, index_col=0)
Taxonomic_df = pd.read_csv(f"{db_dir}/phagcn_taxonomic_label.csv")
merged_df = pd.merge(merged_df, Taxonomic_df, left_on="contig_id", right_on="contig_id", how="inner")
contig_id = merged_df["contig_id"].values
family = merged_df["class"].values
contig_to_family = {name: family for name, family in zip(contig_id, family) if type(family) != type(np.nan) }

G = nx.Graph()
# Add p-edges to the graph
with open(f"{rootpth}/{midfolder}/phagcn_network.ntw") as file_in:
    for line in file_in.readlines():
        tmp = line[:-1].split(" ")
        node1 = tmp[0]
        node2 = tmp[1]
        weight = float(tmp[2])
        
        if "~" in node1 and node1 not in name_to_id.keys():
            print(node1)
            print("ERROR")
            exit(1)
        if "~" in node2 and node2 not in name_to_id.keys():
            print(node2)
            print("ERROR")
            exit(1)

        G.add_edge(node1, node2, weight = 1)

cnt = 0
for i in range(e_weight.shape[0]):
    contig_name = id_to_contig[i]
    if contig_name not in G.nodes():
        sorted_idx = np.argsort(e_weight[i])
        for j in range(5):
            idx = sorted_idx[-j]
            if e_weight[i][idx] != 0:
                ref_name = ID_to_ref[idx]
                if ref_name in G.nodes():
                    G.add_edge(contig_name, ref_name, weight = 1)
                    cnt += 1

node_list = list(G.nodes())
for node in node_list:
    if "~" in node and node not in contig_to_family.keys():
        G.remove_node(node)

test_to_id = {}
# Nov. 16th
#class_to_label = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 5, 9: 5, 10: 5, 11: 5, 12: 5, 
#13: 5, 14: 6, 15: 6, 16: 6, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7, 25: 7, 26: 7}

with open(f'{rootpth}/{midfolder}/phagcn_graph.csv', 'w') as file:
    file.write('Source,Target\n')
    for node in G.nodes():
        for _, neighbor in G.edges(node):
            file.write(f'{node},{neighbor}\n')


# Generating the Knowledge Graph
print("\n\n" + "{:-^80}".format("Generating Knowledge graph"))
mode = "testing"
if mode == "testing":
    test_mask = []
    label = []
    cnt = 0
    for node in G.nodes():
        try:
            label.append(contig_to_family[node]) # Nov. 17th
            cnt+=1
        except:
            if "PhaGCN_" in node:
                try:
                    label.append(-1)
                    test_mask.append(cnt)
                    test_to_id[node] = cnt
                    cnt+=1
                except:
                    print(node)
            else:
                print(node)
    pkl.dump(test_mask, open(f"{phagcninput_pth}/contig.mask", "wb" ) )
    adj = nx.adjacency_matrix(G)
    pkl.dump(adj, open(f"{phagcninput_pth}/contig.graph", "wb" ) )
    pkl.dump(test_to_id, open(f"{phagcninput_pth}/contig.dict", "wb" ) )


# contructing feature map
fn = "database"
contig_feature = pkl.load(open(f"{phagcninput_pth}/phagcn_contig.F",'rb'))
database_feature = pkl.load(open(f"{db_dir}/phagcn_dataset_compressF",'rb'))

feature = []
for node in G.nodes():
    if "~" not in node:
        idx = contig_to_id[node]
        feature.append(contig_feature[idx])
    else:
        try:
            idx = int(name_to_id[node])
            feature.append(database_feature[idx])
        except:
            print(node)

feature = np.array(feature)
pkl.dump(feature, open(f"{phagcninput_pth}/contig.feature", "wb" ) )



# Graph check for each testing samples
cnt = 0
for node in G.nodes:
    if "~" not in node:
        neighbor_label = []
        for edge in G.edges(node):
            neighbor = edge[1]
            if "~" in neighbor:
                neighbor_label.append(contig_to_family[neighbor]) # Nov. 16th
            else:
                continue
        if len(set(neighbor_label)) == 1:
            label[test_to_id[node]] = neighbor_label[0]
            cnt += 1

pkl.dump(label, open(f"{phagcninput_pth}/contig.label", "wb" ) )


phagcninput_pth = os.path.join(rootpth, midfolder, "phgcn/")

seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)



adj        = pkl.load(open(f"{phagcninput_pth}/contig.graph",'rb'))
labels     = pkl.load(open(f"{phagcninput_pth}/contig.label",'rb'))
features   = pkl.load(open(f"{phagcninput_pth}/contig.feature",'rb'))
test_to_id = pkl.load(open(f"{phagcninput_pth}/contig.dict",'rb'))
idx_test   = pkl.load(open(f"{phagcninput_pth}/contig.mask",'rb'))


if not idx_test:
    phagcn_exception_no_visual(rootpth, midfolder, out_dir, ID2length, inputs)
    exit()

idx_test = np.array(idx_test)
labels = np.array(labels)


y_train = np.zeros(labels.shape)
y_test = np.zeros(labels.shape)



idx_train = np.array([i for i in range(len(labels)) if i not in idx_test])


train_mask = sample_mask(idx_train, labels.shape[0])
test_mask = sample_mask(idx_test, labels.shape[0])

y_train[train_mask] = labels[train_mask]
y_test[test_mask] = labels[test_mask]


features = sp.sparse.csc_matrix(features)

print('adj:', adj.shape)
print('features:', features.shape)
print('y:', y_train.shape, y_test.shape) # y_val.shape, 
print('mask:', train_mask.shape, test_mask.shape) # val_mask.shape

features = preprocess_features(features) # [49216, 2], [49216], [2708, 1433]
supports = preprocess_adj(adj)


train_label = torch.from_numpy(y_train).long().to(device)
num_classes = max(labels)+1
train_mask = torch.from_numpy(train_mask.astype(bool)).to(device)
test_label = torch.from_numpy(y_test).long().to(device)
test_mask = torch.from_numpy(test_mask.astype(bool)).to(device)

i = torch.from_numpy(features[0]).long().to(device)
v = torch.from_numpy(features[1]).to(device)
feature = torch.sparse.FloatTensor(i.t(), v, features[2]).float().to(device)

i = torch.from_numpy(supports[0]).long().to(device)
v = torch.from_numpy(supports[1]).to(device)
support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)

print('x :', feature)
print('sp:', support)
num_features_nonzero = feature._nnz()
feat_dim = feature.shape[1]




net = GCN(feat_dim, num_classes, num_features_nonzero)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.01)#args.learning_rate


_ = net.train()
for epoch in range(400):
    # forward pass
    out = net((feature, support))
    loss = masked_loss(out, train_label, train_mask, device)
    loss += 5e-4 * net.l2_loss()
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # output
    if epoch % 10 == 0:
        # calculating the acc
        _ = net.eval()
        out = net((feature, support))
        acc_train = phagcn_accuracy(out.detach().cpu().numpy(), train_mask.detach().cpu().numpy(), labels)
        print(epoch, loss.item(), acc_train)
        if acc_train > 0.98:
            break
    _ = net.train()


net.eval()
out = net((feature, support))
out = F.softmax(out,dim =1)
out = out.cpu().detach().numpy()


pred = np.argmax(out, axis = 1)
score = np.max(out, axis = 1)


pred_to_label = {0: 'Autographiviridae', 1: 'Straboviridae', 2: 'Herelleviridae', 3: 'Drexlerviridae', 4: 'Demerecviridae', 5: 'Peduoviridae', 6: 'Casjensviridae', 7: 'Schitoviridae', 8: 'Kyanoviridae', 9: 'Ackermannviridae', 10: 'Rountreeviridae', 11: 'Salasmaviridae', 12: 'Vilmaviridae', 13: 'Zierdtviridae', 14: 'Mesyanzhinovviridae', 15: 'Chaseviridae', 16: 'Zobellviridae', 17: 'Orlajensenviridae', 18: 'Guelinviridae', 19: 'Steigviridae', 20: 'Duneviridae', 21: 'Pachyviridae', 22: 'Winoviridae', 23: 'Assiduviridae', 24: 'Suoliviridae', 25: 'Naomviridae', 26: 'Intestiviridae', 27: 'Crevaviridae', 28: 'Pervagoviridae'}


with open(f'{rootpth}/{midfolder}/phagcn_mid_prediction.csv', 'w') as f_out:
    _ = f_out.write("Contig,Pred,Score\n")
    for key in test_to_id.keys():
        if labels[test_to_id[key]] == -1:
            _ = f_out.write(str(key) + "," + str(pred_to_label[pred[test_to_id[key]]]) + "," + str(score[test_to_id[key]]) + "\n")
        else:
            _ = f_out.write(str(key) + "," + str(pred_to_label[labels[test_to_id[key]]]) + "," + str(1) + "\n")


name_list = pd.read_csv(f"{rootpth}/{midfolder}/phagcn_name_list.csv")
prediction = pd.read_csv(f'{rootpth}/{midfolder}/phagcn_mid_prediction.csv')
prediction = prediction.rename(columns={'Contig':'idx'})
contig_to_pred = pd.merge(name_list, prediction, on='idx')
contig_to_pred = contig_to_pred.rename(columns={'Contig':'Accession'})
#contig_to_pred = contig_to_pred.drop(columns=['idx'])
contig_to_pred.to_csv(f"{rootpth}/{midfolder}/phagcn_prediction.csv", index = None)

# add no prediction (Nov. 13th)

all_Contigs = contig_to_pred['Accession'].values 
all_Pred = contig_to_pred['Pred'].values
all_Score = contig_to_pred['Score'].values

phage_contig = []
filtered_contig = []
length_dict = {}
seq_dict = {}
for record in SeqIO.parse(f'{contigs}', 'fasta'):
    length_dict[record.id] = len(record.seq)
    seq_dict[record.id] = str(record.seq)
    if len(record.seq) < inputs.len:
        filtered_contig.append(record.id)
    else:
        phage_contig.append(record.id)

# for unpredicted contigs
unpredict_contig = []
unnamed_family = []
for contig in phage_contig:
    if contig not in all_Contigs:
        if contig in check_unknown_all:
            unnamed_family.append(contig)
        else:
            unpredict_contig.append(contig)


unnamed_pred = np.array([check_unknown_all[item] for item in unnamed_family])
unnamed_pred = np.array([f'no_family_avaliable({item})' for item in unnamed_pred])
unnamed_score = np.array([check_unknown_all_score[item] for item in unnamed_family])


all_Contigs = np.concatenate((all_Contigs, np.array(unnamed_family), np.array(filtered_contig), np.array(unpredict_contig)))
all_Pred = np.concatenate((all_Pred, unnamed_pred, np.array(['filtered']*len(filtered_contig)), np.array(['unknown']*len(unpredict_contig))))
all_Score = np.concatenate((all_Score, unnamed_score, np.array([0]*len(filtered_contig)), np.array([0]*len(unpredict_contig))))
all_Length = [length_dict[item] for item in all_Contigs]
all_Pie = np.concatenate((contig_to_pred['Pred'].values, np.array(['unnamed_family']*len(unnamed_family)), np.array(['filtered']*len(filtered_contig)), np.array(['unknown']*len(unpredict_contig))))


contig_to_pred = pd.DataFrame({'Accession': all_Contigs, 'Length': all_Length, 'Pred': all_Pred, 'Score': all_Score})
contig_to_pred.to_csv(f"{rootpth}/{out_dir}/phagcn_prediction.csv", index = None)



#### draw network
if os.path.isfile(os.path.join(rootpth, midfolder, 'phagcn_graph.csv')):
    drop_network('phagcn', rootpth, midfolder, db_dir, out_dir)


name2idx = {contig:idx for contig, idx in zip(name_list['Contig'].values, name_list['idx'].values)}
blast_df = pd.read_csv(f"{rootpth}/{midfolder}/phagcn_results.abc", sep=' ', names=['query', 'ref', 'evalue'])
protein2evalue = parse_evalue(blast_df, f'{rootpth}/{midfolder}', 'phagcn')
rec = []
for record in SeqIO.parse(f'{rootpth}/{midfolder}/test_protein.fa', 'fasta'):
    try:
        name = record.id
        protein2evalue[name2idx[name.rsplit('_',1)[0]]+'_'+name.rsplit('_',1)[1]]
        rec.append(record)
    except:
        pass 
SeqIO.write(rec, f'{rootpth}/{out_dir}/significant_proteins.fa', 'fasta')
os.system(f"cp {rootpth}/{midfolder}/phagcn_results.tab {rootpth}/{out_dir}/blast_results.tab")
os.system(f"sed -i '1i\qseqid\tsseqid\tpident\tlength\tmismatch\tgapopen\tqstart\tqend\tsstart\tsend\tevalue' {rootpth}/{out_dir}/blast_results.tab")


