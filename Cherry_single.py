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
from draw import draw_network, drop_network
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
inputs = parser.parse_args()


contigs   = inputs.contigs
midfolder = inputs.midfolder
rootpth   = inputs.rootpth
db_dir    = inputs.dbdir
out_dir   = inputs.out
parampth  = inputs.parampth
threads   = inputs.threads
length    = inputs.len

if not os.path.isfile(contigs):
    print('cannot find the file')
    exit(1)

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
for record in SeqIO.parse(contigs, 'fasta'):
    if len(record.seq) > inputs.len:
        rec.append(record)
if not rec:
    print('All contigs are filtered!')
    exit()

SeqIO.write(rec, f'{rootpth}/filtered_contigs.fa', 'fasta')

###############################################################
########################## Cherry  ############################
###############################################################
translation(rootpth, os.path.join(rootpth, midfolder), 'filtered_contigs.fa', 'test_protein.fa', threads, inputs.proteins)
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




# generate 4mer feature
cherrypth = f'{rootpth}/{midfolder}/cherry/'
check_path(cherrypth)

test_virus, test_virus2id = return_4mer(f'{rootpth}/CNN_temp/single/')
pkl.dump(test_virus2id, open(f'{cherrypth}/test_virus.dict', 'wb'))
pkl.dump(test_virus, open(f'{cherrypth}/test_virus.F', 'wb'))


try:
    make_diamond_cmd = f'diamond makedb --threads {threads} --in {rootpth}/{midfolder}/phagcn_renamed_protein.fa -d {cherrypth}/test_database.dmnd'
    print("Creating Diamond database...")
    _ = subprocess.check_call(make_diamond_cmd, shell=True)
except:
    print("create database failed")
    exit(1)

run_diamond(f'{db_dir}/cherry_database.dmnd', os.path.join(rootpth, midfolder),  f'phagcn_renamed_protein.fa', 'cherry', threads)
convert_xml(os.path.join(rootpth, midfolder), 'cherry')
if os.path.getsize(f'{rootpth}/{midfolder}/cherry_results.abc') == 0:
    Accession = []
    Length_list = []
    Pred_tmp = []
    for record in SeqIO.parse(f'{contigs}', 'fasta'):
        Accession.append(record.id)
        Length_list.append(len(record.seq))
        Pred_tmp.append('unknown')

    df = pd.DataFrame({"Accession": Accession, "Pred":['unknown']*len(Accession), "Score":[0]*len(Accession)})
    df.to_csv(f"{rootpth}/{out_dir}/cherry_prediction.csv", index = None)
    exit()

run_diamond(f'{cherrypth}/test_database.dmnd', os.path.join(rootpth, midfolder),  f'phagcn_renamed_protein.fa', 'cherry_test', threads)
convert_xml(os.path.join(rootpth, midfolder), 'cherry_test')

database_abc_fp = f"{rootpth}/{midfolder}/cherry_merged.abc"
_ = subprocess.check_call(f"cat {db_dir}/cherry_database.self-diamond.tab.abc {rootpth}/{midfolder}/cherry_results.abc {rootpth}/{midfolder}/cherry_test_results.abc > {database_abc_fp}", shell=True)

blastp = pd.read_csv(database_abc_fp, sep=' ', names=["contig", "ref", "e-value"])
protein_id = sorted(list(set(blastp["contig"].values)|set(blastp["ref"].values)))
contig_protein = [item for item in protein_id if "PhaGCN" == item.split("_")[0]]
contig_id = [item.rsplit("_", 1)[0] for item in contig_protein]
description = ["hypothetical protein" for item in contig_protein]
gene2genome = pd.DataFrame({"protein_id": contig_protein, "contig_id": contig_id ,"keywords": description})
gene2genome.to_csv(f"{rootpth}/{midfolder}/cherry_contig_gene_to_genome.csv", index=None)


_ = subprocess.check_call(f"cat {db_dir}/cherry/database_gene_to_genome.csv {rootpth}/{midfolder}/cherry_contig_gene_to_genome.csv > {rootpth}/{midfolder}/cherry_gene_to_genome.csv", shell=True)

gene2genome_fp = f"{rootpth}/{midfolder}/cherry_gene_to_genome.csv"
gene2genome_df = pd.read_csv(gene2genome_fp, sep=',', header=0)


# Parameters for MCL
pc_overlap, pc_penalty, pc_haircut, pc_inflation = 0.8, 2.0, 0.1, 2.0
pcs_fp = make_protein_clusters_mcl(database_abc_fp, os.path.join(rootpth, midfolder), pc_inflation)
print("Building the cluster and profiles (this may take some time...)")


# Dump MCL results
protein_df, clusters_df, profiles_df, contigs_df = build_clusters(pcs_fp, gene2genome_df)
print("Saving files")
dfs = [gene2genome_df, contigs_df, clusters_df]
names = ['proteins', 'contigs', 'pcs']

for name, df in zip(names, dfs):
    fn = "Cyber_cherry_{}.csv".format(name)
    fp = os.path.join(f'{rootpth}/{midfolder}', fn)
    index_id = name.strip('s') + '_id'
    df.set_index(index_id).to_csv(fp)


# Replace names
contigs_csv_df = contigs_df.copy()
contigs_csv_df.index.name = "pos"
contigs_csv_df.reset_index(inplace=True)

pcs_csv_df = clusters_df.copy()
profiles = profiles_df.copy()

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
merged_fp = os.path.join(cherrypth, 'merged_df.csv')
merged_df.to_csv(merged_fp)

ntw = create_network(matrix, singletons, thres=1, max_sig=300)
fi = to_clusterer(ntw, f"{cherrypth}/intermediate.ntw", merged_df.copy())

# BLASTN
_ = subprocess.check_call(f"cat {rootpth}/CNN_temp/single/* > {cherrypth}/test.fa", shell=True)
query_file = f"{cherrypth}/test.fa"
db_virus_prefix = f"{db_dir}/virus_db/allVIRUS"
output_file = f"{cherrypth}/virus_out.tab"
virus_call = NcbiblastnCommandline(query=query_file,db=db_virus_prefix,out=output_file,outfmt="6 qseqid sseqid evalue pident length qlen", evalue=1e-10,
                                 task='megablast', max_target_seqs=1, perc_identity=90,num_threads=threads)
virus_call()


virus_pred = {}
with open(output_file) as file_out:
    for line in file_out.readlines():
        parse = line.replace("\n", "").split("\t")
        virus = parse[0]
        ref_virus = parse[1].split('|')[1]
        ref_virus = ref_virus.split('.')[0]
        ident = float(parse[-3])
        length = float(parse[-2])
        qlen = float(parse[-1])
        if virus not in virus_pred and length/qlen > 0.95 and ident > 0.95:
            virus_pred[virus] = ref_virus

pkl.dump(virus_pred, open(f'{cherrypth}/virus_pred.dict', 'wb'))


# Dump graph
G = nx.Graph()
# Create graph
with open(f"{cherrypth}/intermediate.ntw") as file_in:
    for line in file_in.readlines():
        tmp = line[:-1].split(" ")
        node1 = tmp[0]
        node2 = tmp[1]
        G.add_edge(node1, node2, weight = 1)


graph = f"{cherrypth}/phage_phage.ntw"
with open(graph, 'w') as file_out:
    for node1 in G.nodes():
        for _,node2 in G.edges(node1):
            _ = file_out.write(node1+","+node2+"\n")


query_file = f"{cherrypth}/test.fa"
db_host_crispr_prefix = f"{db_dir}/crispr_db/allCRISPRs"
output_file = f"{cherrypth}/crispr_out.tab"
crispr_call = NcbiblastnCommandline(query=query_file,db=db_host_crispr_prefix,out=output_file,outfmt="6 qseqid sseqid evalue pident length slen", evalue=1,gapopen=10,penalty=-1,
                                  gapextend=2,word_size=7,dust='no',
                                 task='blastn-short',perc_identity=90,num_threads=threads)
crispr_call()


crispr_pred = {}
with open(output_file) as file_out:
    for line in file_out.readlines():
        parse = line.replace("\n", "").split("\t")
        virus = parse[0]
        prokaryote = parse[1].split('|')[1]
        prokaryote = prokaryote.split('.')[0]
        ident = float(parse[-3])
        length = float(parse[-2])
        slen = float(parse[-1])
        if virus not in crispr_pred and length/slen > 0.95 and ident > 0.95:
            crispr_pred[virus] = prokaryote

pkl.dump(crispr_pred, open(f'{cherrypth}/crispr_pred.dict', 'wb'))

blast_database_out = f'{db_dir}/blast_db/'
blast_tab_out = f'{cherrypth}/blast_tab'
all_blast_tab = f'{cherrypth}/all_blast_tab'
check_path(blast_database_out)
check_path(blast_tab_out)
check_path(all_blast_tab)

# database only 
genome_list = os.listdir(f'{db_dir}/prokaryote')
for genome in genome_list:
    accession = genome.split(".")[0]
    blast_cmd = f'blastn -query {cherrypth}/test.fa -db {blast_database_out}/{accession} -outfmt 6 -out {blast_tab_out}/{accession}.tab -num_threads {threads}'
    print("Running blastn...")
    _ = subprocess.check_call(blast_cmd, shell=True)

for file in os.listdir(blast_tab_out):
    os.system(f"cat {blast_tab_out}/{file} {db_dir}/blast_tab/{file} > {all_blast_tab}/{file}")


# add connections between prokaryotes and viruses
tab_file_list = os.listdir(all_blast_tab)
prokaryote2virus = {}
for file in tab_file_list:
    prokaryote_id = file.split('.')[0]
    virus_id_list = []
    with open(f'{all_blast_tab}/{file}') as file_in:
        for line in file_in.readlines():
            tmp = line.split('\t')
            virus_id = tmp[0]
            try:
                prokaryote2virus[prokaryote_id].append(virus_id)
            except:
                prokaryote2virus[prokaryote_id] = [virus_id]



# De-duplication
for key in prokaryote2virus:
    prokaryote2virus[key] = list(set(prokaryote2virus[key]))


# Save the virus-host graph
with open(f"{cherrypth}/phage_host.ntw", 'w') as file_out:
    for prokaryote in prokaryote2virus:
        for virus in prokaryote2virus[prokaryote]:
            _ = file_out.write(prokaryote + "," + virus + "\n")


phage_phage_ntw = f"{cherrypth}/phage_phage.ntw"
phage_host_ntw = f"{cherrypth}/phage_host.ntw"




# Add virus-virus edges
G = nx.Graph()
with open(phage_phage_ntw) as file_in:
    for line in file_in.readlines():
        tmp = line[:-1].split(",")
        node1 = tmp[0].split('.')[0]
        node2 = tmp[1].split('.')[0]
        G.add_edge(node1, node2, weight = 1)


# Add blastn edges
with open(phage_host_ntw) as file_in:
    for line in file_in.readlines():
        tmp = line[:-1].split(",")
        node1 = tmp[0].split('.')[0]
        node2 = tmp[1].split('.')[0]
        G.add_edge(node1, node2, weight = 1)



bacteria_df = pd.read_csv(f'{db_dir}/cherry/prokaryote.csv')
virus_df = pd.read_csv(f'{db_dir}/cherry/virus.csv')

bacteria_list = os.listdir(f'{db_dir}/prokaryote/')
bacteria_list = [name.split('.')[0] for name in bacteria_list]

# add crispr edges
species2bacteria = {bacteria_df[bacteria_df['Accession'] == item]['Species'].values[0]: item for item in bacteria_list}
crispr_pred = pkl.load(open(f'{cherrypth}/crispr_pred.dict', 'rb'))
for virus, host in crispr_pred.items():
    if host in species2bacteria:
        G.add_edge(virus, species2bacteria[host])

# add dataset edges
for bacteria in bacteria_list:
    species = bacteria_df[bacteria_df['Accession'] == bacteria]['Species'].values[0]
    phage_list = virus_df[virus_df['Species'] == species]['Accession'].values
    for phage in phage_list:
        if phage in G.nodes():
            G.add_edge(bacteria, phage, weight = 1)


# dump the graph G
with open(f'{rootpth}/{midfolder}/cherry_graph.csv', 'w') as file:
    file.write('Source,Target\n')
    for node in G.nodes():
        for _, neighbor in G.edges(node):
            file.write(f'{node},{neighbor}\n')


virus2id      = pkl.load(open(f"{db_dir}/cherry/virus.dict",'rb'))
virusF        = pkl.load(open(f"{db_dir}/cherry/virus.F",'rb'))
prokaryote2id = pkl.load(open(f"{db_dir}/cherry/prokaryote.dict",'rb'))
prokaryoteF   = pkl.load(open(f"{db_dir}/cherry/prokaryote.F",'rb'))
 

test_virus2id      = pkl.load(open(f"{cherrypth}/test_virus.dict",'rb'))
test_virusF        = pkl.load(open(f"{cherrypth}/test_virus.F",'rb'))
test_prokaryote2id = {}


node_feature = []
for node in G.nodes():
    # if prokaryote node
    if node in prokaryote2id.keys():
        node_feature.append(prokaryoteF[prokaryote2id[node]])
    # if virus node
    elif node in virus2id.keys():
        node_feature.append(virusF[virus2id[node]])
    # if test virus node
    elif node in test_virus2id.keys():
        node_feature.append(test_virusF[test_virus2id[node]])
    # if test prokaryote node
    elif node in test_prokaryote2id.keys():
        node_feature.append(test_prokaryoteF[test_prokaryote2id[node]])
    else:
        print(f"node error {node}")
        exit()

node_feature = np.array(node_feature)



crispr_pred = pkl.load(open(f'{cherrypth}/crispr_pred.dict', 'rb'))
virus_pred = pkl.load(open(f'{cherrypth}/virus_pred.dict', 'rb'))
virus_df = virus_df
prokaryote_df = bacteria_df


idx = 0
test_id = {}
node2label = {}
cnt = 0
for node in G.nodes():
    # if test virus node
    if "PhaGCN" in node:
        neighbor_label = []
        for _, neighbor in G.edges(node):
            if neighbor in virus2id.keys():
                virus_label = virus_df[virus_df['Accession'] == neighbor]['Species'].values[0]
                neighbor_label.append(virus_label)
            elif neighbor in prokaryote2id.keys():
                prokaryote_label = prokaryote_df[prokaryote_df['Accession'] == neighbor]['Species'].values[0]
                neighbor_label.append(prokaryote_label)
        # subgraph
        if len(set(neighbor_label)) == 1:
            node2label[node] = neighbor_label[0]
            test_id[node] = 1
        # CRISPR
        elif node in crispr_pred:
            node2label[node] = prokaryote_df[prokaryote_df['Accession'] == crispr_pred[node]]['Species'].values[0]
            test_id[node] = 1
        elif node in virus_pred:
            node2label[node] = virus_df[virus_df['Accession'] == virus_pred[node]]['Species'].values[0]
            test_id[node] = 1
        # unlabelled
        else:
            node2label[node] = 'unknown'
            test_id[node] = 2
    # if phage or host node
    elif node in prokaryote2id.keys():
        prokaryote_label = prokaryote_df[prokaryote_df['Accession'] == node]['Species'].values[0]
        node2label[node] = prokaryote_label
        test_id[node] = 0
    elif node in test_prokaryote2id.keys():
        prokaryote_label = prokaryote_df[prokaryote_df['Accession'] == node]['Species'].values[0]
        node2label[node] = prokaryote_label
        test_id[node] = 0
    elif node in virus2id.keys():
        virus_label = virus_df[virus_df['Accession'] == node]['Species'].values[0]
        node2label[node] = virus_label
        test_id[node] = 0
    else: 
        print("Error: " + node)
    idx += 1


# check subgraph situation 1
for sub in nx.connected_components(G):
    flag = 0
    for node in sub:
        if "PhaGCN" not in node:
            flag = 1
    # use CRISPR
    if not flag:
        CRISPR_label = ""
        CRISPR_cnt = 0
        for node in sub:
            if node in crispr_pred:
                CRISPR_cnt+=1
                CRISPR_label = crispr_pred[node]
        if CRISPR_cnt == 1:
            for node in sub:
                node2label[node] = CRISPR_label

# check subgraph situation 2
for sub in nx.connected_components(G):
    sub_label = []
    for node in sub:
        if node in virus2id.keys():
            virus_label = virus_df[virus_df['Accession'] == node]['Species'].values[0]
            sub_label.append(virus_label)
        elif node in prokaryote2id.keys():
            prokaryote_label = prokaryote_df[prokaryote_df['Accession'] == node]['Species'].values[0]
            sub_label.append(prokaryote_label)
    if len(set(sub_label)) == 1:
        for node in sub:
            node2label[node] = sub_label[0]
            test_id[node] = 1
    elif len(set(sub_label)) == 0:
        for node in sub:
            node2label[node] = 'unknown'
            test_id[node] = 3

# check graph situation 3
for node in G.nodes():
    # if test virus node
    if "PhaGCN" in node:
        neighbor_label = []
        for _, neighbor in G.edges(node):
            if neighbor in virus2id.keys():
                virus_label = virus_df[virus_df['Accession'] == neighbor]['Species'].values[0]
                neighbor_label.append(virus_label)
            elif neighbor in prokaryote2id.keys():
                prokaryote_label = prokaryote_df[prokaryote_df['Accession'] == neighbor]['Species'].values[0]
                neighbor_label.append(prokaryote_label)
        cnt = Counter(neighbor_label)
        most_cnt = cnt.most_common()[0]
        if most_cnt[1]- 1/len(set(sub_label)) > 0.3:
            node2label[node] = most_cnt [0]
            test_id[node] = 1

id2node = {idx: node for idx, node in enumerate(G.nodes())}
node2id = {node: idx for idx, node in enumerate(G.nodes())}

adj = nx.adjacency_matrix(G)
pkl.dump(adj,          open(f"{cherrypth}/graph.list", "wb" ))
pkl.dump(node_feature, open(f"{cherrypth}/feature.list", "wb" ))
pkl.dump(node2label,   open(f"{cherrypth}/node2label.dict", "wb" ))
pkl.dump(id2node,      open(f"{cherrypth}/id2node.dict", "wb" ))
pkl.dump(node2id,      open(f"{cherrypth}/node2id.dict", "wb" ))
pkl.dump(test_id,      open(f"{cherrypth}/test_id.dict", "wb" ))







# model
trainable_host = []
for file in os.listdir(f'{db_dir}/prokaryote/'):
    trainable_host.append(file.rsplit('.', 1)[0])


idx_test= test_id
host2id = {}
label2hostid =  {}
trainable_host_idx = []
trainable_label = []
for idx, node in id2node.items():
    # if prokaryote
    if node in trainable_host:
        host2id[node] = idx
        trainable_host_idx.append(idx)
        trainable_label.append(node2label[node])
        label2hostid[node2label[node]] = idx




# pre-processing
features = sp.sparse.csc_matrix(node_feature)
print('adj:', adj.shape)
print('features:', features.shape)


# convert to torch tensor
features = preprocess_features(features)
supports = preprocess_adj(adj)
num_classes = len(set(list(node2label.values())))+1
# graph
i = torch.from_numpy(features[0]).long().to(device)
v = torch.from_numpy(features[1]).to(device)
feature = torch.sparse.FloatTensor(i.t(), v, features[2]).float().to(device)
feature = feature.to_dense()
i = torch.from_numpy(supports[0]).long().to(device)
v = torch.from_numpy(supports[1]).to(device)
support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)
support = support.to_dense()


print('x :', feature)
print('sp:', support)
feat_dim = adj.shape[0]
node_dim = feature.shape[1]


# Definition of the model
net = Cherry.encoder(feat_dim, node_dim, node_dim, 0)
decoder = Cherry.decoder(node_dim, 128, 32)


# Load pre-trained model
encoder_dict = torch.load(f"{parampth}/cherry/Encoder_Species.pkl", map_location='cpu')
decoder_dict = torch.load(f"{parampth}/cherry/Decoder_Species.pkl", map_location='cpu')
net.load_state_dict(encoder_dict)
decoder.load_state_dict(decoder_dict)

net.to(device)
decoder.to(device)

# end-to-end training
params = list(net.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=0.001)#args.learning_rate
loss_func = nn.BCEWithLogitsLoss()


# predicting host
node2pred = {}
with torch.no_grad():
    encode = net((feature, support))
    for i in range(len(encode)):
        confident_label = 'unknown'
        if idx_test[id2node[i]] == 0 or idx_test[id2node[i]] == 3:
            continue
        if idx_test[id2node[i]] == 1:
            confident_label = node2label[id2node[i]]
        virus_feature = encode[i]
        pred_label_score = []
        for label in set(trainable_label):
            if label == confident_label:
                pred_label_score.append((label, 1))
                continue
            prokaryote_feature = encode[label2hostid[label]]
            pred = decoder(virus_feature - prokaryote_feature)
            pred_label_score.append((label, torch.sigmoid(pred).detach().cpu().numpy()[0]))
        node2pred[id2node[i]] = sorted(pred_label_score, key=lambda tup: tup[1], reverse=True)
    for virus in crispr_pred:
        if virus not in node2pred:
            pred = prokaryote_df[prokaryote_df['Accession'] == crispr_pred[virus]]['Species'].values[0]
            node2pred[virus] = [(pred, 1)]
    # dump the prediction
    with open(f"{rootpth}/{midfolder}/cherry_mid_predict.csv", 'w') as file_out:
        file_out.write('Contig,')
        for i in range(inputs.topk):
            file_out.write(f'Top_{i+1}_label,Score_{i+1},')
        file_out.write('Type\n')
        for contig in node2pred:
            file_out.write(f'{contig},')
            cnt = 1
            for label, score in node2pred[contig]:
                if cnt > inputs.topk:
                    break
                cnt+=1
                file_out.write(f'{label},{score:.2f},')
            if contig in crispr_pred:
                file_out.write(f'CRISPR')
            else:
                file_out.write(f'Predict')
            file_out.write('\n')

tmp_pred = pd.read_csv(f"{rootpth}/{midfolder}/cherry_mid_predict.csv")
name_list = pd.read_csv(f"{rootpth}/{midfolder}/phagcn_name_list.csv")
prediction = tmp_pred.rename(columns={'Contig':'idx'})
contig_to_pred = pd.merge(name_list, prediction, on='idx')
contig_to_pred = contig_to_pred.rename(columns={'Contig':'Accession'})
#contig_to_pred = contig_to_pred.drop(columns=['idx'])
contig_to_pred.to_csv(f"{rootpth}/{midfolder}/cherry_prediction.csv", index = None)


# add no prediction (Nov. 13th)

all_Contigs = contig_to_pred['Accession'].values 
all_Pred = contig_to_pred['Top_1_label'].values
all_Score = contig_to_pred['Score_1'].values
all_Type = contig_to_pred['Type'].values

if len(set(all_Type)) == 1 and all_Type[0] == 'CRISPR':
    pass

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

unpredict_contig = []
for contig in phage_contig:
    if contig not in all_Contigs:
        unpredict_contig.append(contig)



all_Contigs = np.concatenate((all_Contigs, np.array(filtered_contig), np.array(unpredict_contig)))
all_Pred = np.concatenate((all_Pred, np.array(['filtered']*len(filtered_contig)), np.array(['unknown']*len(unpredict_contig))))
all_Length = [length_dict[item] for item in all_Contigs]
all_Score = np.concatenate((all_Score, np.array([0]*len(filtered_contig)), np.array([0]*len(unpredict_contig))))
all_Type = np.concatenate((all_Type, np.array(['-']*len(filtered_contig)), np.array(['-']*len(unpredict_contig))))


contig_to_pred = pd.DataFrame({'Accession': all_Contigs, 'Length': all_Length, 'Pred': all_Pred, 'Score': all_Score, 'Type': all_Type})
contig_to_pred.to_csv(f"{rootpth}/{out_dir}/cherry_prediction.csv", index = None)



### Draw networks
drop_network('cherry', rootpth, midfolder, db_dir, out_dir)


name2idx = {contig:idx for contig, idx in zip(name_list['Contig'].values, name_list['idx'].values)}
blast_df = pd.read_csv(f"{rootpth}/{midfolder}/cherry_results.abc", sep=' ', names=['query', 'ref', 'evalue'])
protein2evalue = parse_evalue(blast_df, f'{rootpth}/{midfolder}', 'cherry')
rec = []
for record in SeqIO.parse(f'{rootpth}/{midfolder}/test_protein.fa', 'fasta'):
    try:
        name = record.id
        protein2evalue[name2idx[name.rsplit('_',1)[0]]+'_'+name.rsplit('_',1)[1]]
        rec.append(record)
    except:
        pass 
SeqIO.write(rec, f'{rootpth}/{out_dir}/significant_proteins.fa', 'fasta')
os.system(f"cp {rootpth}/{midfolder}/cherry_results.tab {rootpth}/{out_dir}/blast_results.tab")
os.system(f"sed -i '1i\qseqid\tsseqid\tpident\tlength\tmismatch\tgapopen\tqstart\tqend\tsstart\tsend\tevalue' {rootpth}/{out_dir}/blast_results.tab")










