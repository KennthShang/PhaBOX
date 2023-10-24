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
from collections import Counter
from scipy.special import softmax
from scripts.data import load_data, preprocess_features, preprocess_adj, sample_mask






parser = argparse.ArgumentParser(description="""Main script of PhaSUIT.""")
parser.add_argument('--contigs', help='FASTA file of contigs',  default = 'inputs.fa')
parser.add_argument('--threads', help='number of threads to use', type=int, default=8)
parser.add_argument('--len', help='minimum length of contigs', type=int, default=3000)
parser.add_argument('--reject', help='threshold to reject prophage',  type=float, default = 0.3)
parser.add_argument('--rootpth', help='rootpth of the user', default='user_0/')
parser.add_argument('--out', help='output path of the user', default='out/')
parser.add_argument('--midfolder', help='mid folder for intermidiate files', default='midfolder/')
parser.add_argument('--dbdir', help='database directory',  default = 'database/')
parser.add_argument('--parampth', help='path of parameters',  default = 'parameters/')
parser.add_argument('--scriptpth', help='path of parameters',  default = 'scripts/')
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
ID2length = {}
for record in SeqIO.parse(contigs, 'fasta'):
    ID2length[record.id] = len(record.seq)
    if len(record.seq) > inputs.len:
        rec.append(record)

if not rec:
    with open(f'{rootpth}/{out_dir}/phamer_prediction.csv', 'w') as file_out:
        file_out.write("Accession,Pred,Score\n")
        for record in SeqIO.parse(contigs, 'fasta'):
            file_out.write(f'{record.id},filtered,0\n')
    exit()

SeqIO.write(rec, f'{rootpth}/filtered_contigs.fa', 'fasta')


###############################################################
##########################  PhaMer ############################
###############################################################


# add convertxml (Nov. 8th)
translation(rootpth, os.path.join(rootpth, midfolder), 'filtered_contigs.fa', 'test_protein.fa', threads, inputs.proteins)
run_diamond(f'{db_dir}/phamer_database.dmnd', os.path.join(rootpth, midfolder), 'test_protein.fa', 'phamer', threads)
convert_xml(os.path.join(rootpth, midfolder), 'phamer', scriptpth)
if os.path.getsize(f'{rootpth}/{midfolder}/phamer_results.abc') == 0:
    with open(f'{rootpth}/{out_dir}/phamer_prediction.csv', 'w') as file_out:
        file_out.write("Accession,Pred,Score\n")
        for record in SeqIO.parse(contigs, 'fasta'):
            file_out.write(f'{record.id},non-phage,0\n')
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
        for record in SeqIO.parse(contigs, 'fasta'):
            file_out.write(f'{record.id},non-phage,0\n')
    exit()   

### Add filtered label (Nov. 8th)
contigs_list = list(id2contig.values())
contigs_add = []
for record in SeqIO.parse(f'{contigs}', 'fasta'):
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




pred_phage_dict = {}
for contig, pred in zip(pred_csv['Accession'].values, pred_csv['Pred'].values):
    if pred == 'phage':
        pred_phage_dict[contig] = contig2id[contig]

# FLAGS
if pred_phage_dict:
    pass
else:
    exit()


###############################################################
##########################  PhaTYP ############################
###############################################################

id2contig = {key: item for key, item in enumerate(pred_phage_dict.keys())}
recruit_sentence = sentence[list(pred_phage_dict.values())]
pkl.dump(recruit_sentence, open(f'{rootpth}/{midfolder}/phatyp_sentence.feat', 'wb'))
pkl.dump(pcs2idx, open(f'{rootpth}/{midfolder}/phatyp_pc2wordsid.dict', 'wb'))

generate_bert_input(db_dir, os.path.join(rootpth, midfolder), os.path.join(rootpth, midfolder), 'phatyp')

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



pred_csv = pd.DataFrame({"Accession":id2contig.values(), "Pred":all_pred, "Score":all_score})
pred_csv.to_csv(f'{rootpth}/{out_dir}/phatyp_prediction.csv', index = False)


###############################################################
########################## PhaGCN  ############################
###############################################################
nucl, protein = recruit_phage_file(rootpth, midfolder, pred_phage_dict)

SeqIO.write(nucl, f'{rootpth}/checked_phage_contigs.fa',"fasta")
SeqIO.write(protein, f'{rootpth}/checked_phage_protein.fa',"fasta")


# Filter unknown family (Nov. 16th)
query_file = f"{rootpth}/checked_phage_contigs.fa"
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
        if length/qlen > 0.95 and ident > 95:
            check_unknown[virus] = target
        if virus not in check_unknown_all:
            ident  = float(parse[-3])/100
            ident  = float(f"{ident:.3f}")
            check_unknown_all[virus] = target
            check_unknown_all_score[virus] = ident

rec = []
for record in SeqIO.parse(f'{rootpth}/checked_phage_contigs.fa', 'fasta'):
    try:
        if check_unknown[record.id]:
            continue
    except:
        rec.append(record)


if not rec:
    phagcn_exception_no_visual(rootpth, midfolder, out_dir, ID2length, inputs, 'checked_phage_contigs.fa')

else:
    SeqIO.write(rec, f'{rootpth}/checked_phage_phagcn_contigs.fa', 'fasta')

    rec = []
    for record in SeqIO.parse(f'{rootpth}/checked_phage_protein.fa', 'fasta'):
        name = record.id
        name = name.rsplit('_', 1)[0]
        try:
            if check_unknown[name]:
                continue
        except:
            rec.append(record)

    SeqIO.write(rec, f'{rootpth}/checked_phage_phagcn_protein.fa', 'fasta')

    single_pth   = os.path.join(rootpth, "CNN_temp/single")
    cnninput_pth = os.path.join(rootpth, "CNN_temp/input")
    phagcninput_pth = os.path.join(rootpth, midfolder, "phgcn/")
    check_path(single_pth)
    check_path(cnninput_pth)
    check_path(phagcninput_pth)

    contig2name = {}
    with open(f"{rootpth}/{midfolder}/phagcn_name_list.csv",'w') as list_out:
        list_out.write("Contig,idx\n")
        for contig_id, record in enumerate(SeqIO.parse(f'{rootpth}/checked_phage_phagcn_contigs.fa', "fasta")):
            name = f"PhaGCN_{str(contig_id)}"
            list_out.write(record.id + "," + name+ "\n")
            contig2name[record.id] = name
            record.id = name
            _ = SeqIO.write(record, f"{single_pth}/{name}.fa", "fasta")

    rename_rec = []
    for record in SeqIO.parse(f'{rootpth}/checked_phage_phagcn_protein.fa',"fasta"):
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
    cnn = cnn.to(device)
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
        val_feature = val_feature.reshape(len(val_feature), 1, 1998, 100).to(device)
        # prediction
        out = cnn(val_feature)
        out = out.cpu().detach().numpy()  
        out = np.sum(out, axis=0)
        compress_feature.append(out)

    compress_feature = np.array(compress_feature)
    pkl.dump(compress_feature, open(f"{phagcninput_pth}/phagcn_contig.F", 'wb'))

    # Generate knowledge graph
    # add convertxml (Nov. 8th)
    run_diamond(f'{db_dir}/phagcn_database.dmnd', os.path.join(rootpth, midfolder), 'phagcn_renamed_protein.fa', 'phagcn', threads)
    convert_xml(os.path.join(rootpth, midfolder), 'phagcn', scriptpth)

    #FLAGS
    if os.path.getsize(f'{rootpth}/{midfolder}/phagcn_results.abc') == 0:
        phagcn_exception_no_visual(rootpth, midfolder, out_dir, ID2length, inputs, 'checked_phage_contigs.fa')
    else:
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
                    label.append(contig_to_family[node])
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
                        neighbor_label.append(contig_to_family[neighbor])
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
        else:
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
            train_mask = torch.from_numpy(train_mask.astype(np.bool)).to(device)
            test_label = torch.from_numpy(y_test).long().to(device)
            test_mask = torch.from_numpy(test_mask.astype(np.bool)).to(device)

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
                loss = masked_loss(out, train_label, train_mask)
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
            all_acc_phagcn = contig_to_pred['Accession'].values
            phamer_df = pd.read_csv(f'{rootpth}/{out_dir}/phamer_prediction.csv')
            phage_contig = phamer_df[phamer_df['Pred']=='phage']['Accession'].values

            unpredict_contig = []
            unnamed_family = []
            for contig in phage_contig:
                if contig not in all_acc_phagcn:
                    if contig in check_unknown_all:
                        unnamed_family.append(contig)
                    else:
                        unpredict_contig.append(contig)

            unnamed_pred = np.array([check_unknown_all[item] for item in unnamed_family])
            unnamed_pred = np.array([f'no_family_avaliable({item})' for item in unnamed_pred])
            unnamed_score = np.array([check_unknown_all_score[item] for item in unnamed_family])

            unpredict_df = pd.DataFrame({'Accession':unpredict_contig, 'Pred': ['unknown']*len(unpredict_contig), 'Score':[0]*len(unpredict_contig)})
            unnamed_df   = pd.DataFrame({'Accession':unnamed_family, 'Pred': unnamed_pred, 'Score':unnamed_score})
            contig_to_pred = pd.concat((contig_to_pred, unpredict_df, unnamed_df))
            contig_to_pred.drop(columns=['idx'])
            contig_to_pred.to_csv(f"{rootpth}/{out_dir}/phagcn_prediction.csv", index = None)


###############################################################
########################## Cherry  ############################
###############################################################

single_pth   = os.path.join(rootpth, "CNN_temp/single")
check_path(single_pth)

contig2name = {}
with open(f"{rootpth}/{midfolder}/cherry_name_list.csv",'w') as list_out:
    list_out.write("Contig,idx\n")
    for contig_id, record in enumerate(SeqIO.parse(f'{rootpth}/checked_phage_contigs.fa', "fasta")):
        name = f"PhaGCN_{str(contig_id)}"
        list_out.write(record.id + "," + name+ "\n")
        contig2name[record.id] = name
        record.id = name
        _ = SeqIO.write(record, f"{single_pth}/{name}.fa", "fasta")

rename_rec = []
for record in SeqIO.parse(f'{rootpth}/checked_phage_protein.fa',"fasta"):
    old_name = record.id
    idx = old_name.rsplit('_', 1)[1]
    record.id = contig2name[old_name.rsplit('_', 1)[0]] +"_"+ idx
    rename_rec.append(record)

SeqIO.write(rename_rec, f'{rootpth}/{midfolder}/cherry_renamed_protein.fa', 'fasta')



# generate 4mer feature
cherrypth = f'{rootpth}/{midfolder}/cherry/'
check_path(cherrypth)

test_virus, test_virus2id = return_4mer(f'{rootpth}/CNN_temp/single/')
pkl.dump(test_virus2id, open(f'{cherrypth}/test_virus.dict', 'wb'))
pkl.dump(test_virus, open(f'{cherrypth}/test_virus.F', 'wb'))


try:
    make_diamond_cmd = f'diamond makedb --threads {threads} --in {rootpth}/{midfolder}/cherry_renamed_protein.fa -d {cherrypth}/test_database.dmnd'
    print("Creating Diamond database...")
    _ = subprocess.check_call(make_diamond_cmd, shell=True)
except:
    print("create database failed")
    exit(1)

# add convertxml (Nov. 8th)
run_diamond(f'{db_dir}/cherry_database.dmnd', os.path.join(rootpth, midfolder), 'cherry_renamed_protein.fa', 'cherry', threads)
convert_xml(os.path.join(rootpth, midfolder), 'cherry', scriptpth)
run_diamond(f'{cherrypth}/test_database.dmnd', os.path.join(rootpth, midfolder),  f'cherry_renamed_protein.fa', 'cherry_test', threads)
convert_xml(os.path.join(rootpth, midfolder), 'cherry_test', scriptpth)

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
try:
    rec = []
    for file in os.listdir(f'{rootpth}/CNN_temp/single/'):
        for record in SeqIO.parse(f'{rootpth}/CNN_temp/single/{file}', 'fasta'):
            rec.append(record)
    SeqIO.write(rec, f"{cherrypth}/test.fa", 'fasta')
except:
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
        if virus not in virus_pred and length/qlen > 0.95 and ident > 90:
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
        if virus not in crispr_pred and length/slen > 0.95 and ident > 90:
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
        try:
            if not neighbor_label:
                continue
            cnt = Counter(neighbor_label)
            most_cnt = cnt.most_common()[0]
            if len(set(sub_label)) == 0:
                continue
            if most_cnt[1]- 1/len(set(sub_label)) > 0.3:
                node2label[node] = most_cnt[0]
                test_id[node] = 1
        except:
            continue

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
                pred_label_score.append((label, 10))
                continue
            prokaryote_feature = encode[label2hostid[label]]
            pred = decoder(virus_feature - prokaryote_feature)
            pred_label_score.append((label, torch.sigmoid(pred).detach().cpu().numpy()[0]))
        node2pred[id2node[i]] = sorted(pred_label_score, key=lambda tup: tup[1], reverse=True)
    for virus in crispr_pred:
        #if virus not in node2pred:
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
                if score > 1:
                    score = 1
                file_out.write(f'{label},{score:.3f},')
            if contig in crispr_pred:
                file_out.write(f'CRISPR')
            else:
                file_out.write(f'Predict')
            file_out.write('\n')

tmp_pred = pd.read_csv(f"{rootpth}/{midfolder}/cherry_mid_predict.csv")
name_list = pd.read_csv(f"{rootpth}/{midfolder}/cherry_name_list.csv")
prediction = tmp_pred.rename(columns={'Contig':'idx'})
contig_to_pred = pd.merge(name_list, prediction, on='idx')
contig_to_pred = contig_to_pred.rename(columns={'Contig': 'Accession'})
#contig_to_pred = contig_to_pred.drop(columns=['idx'])
contig_to_pred.to_csv(f"{rootpth}/{midfolder}/cherry_prediction.csv", index = None)

all_Contigs = contig_to_pred['Accession'].values 
all_Pred = contig_to_pred['Top_1_label'].values
all_Score = contig_to_pred['Score_1'].values
all_Type = contig_to_pred['Type'].values

# add no prediction (Nov. 13th)
all_acc_cherry = contig_to_pred['Accession'].values
phamer_df = pd.read_csv(f'{rootpth}/{out_dir}/phamer_prediction.csv')
phage_contig = phamer_df[phamer_df['Pred']=='phage']['Accession'].values

unpredict_contig = []
for contig in phage_contig:
    if contig not in all_acc_cherry:
        unpredict_contig.append(contig)


all_Contigs = np.concatenate((all_Contigs, np.array(unpredict_contig)))
all_Pred = np.concatenate((all_Pred, np.array(['unknown']*len(unpredict_contig))))
all_Score = np.concatenate((all_Score, np.array([0]*len(unpredict_contig))))
all_Type = np.concatenate((all_Type, np.array(['-']*len(unpredict_contig))))


contig_to_pred = pd.DataFrame({'Accession': all_Contigs, 'Pred': all_Pred, 'Score': all_Score, 'Type': all_Type})
contig_to_pred.to_csv(f"{rootpth}/{out_dir}/cherry_prediction.csv", index = None)




#### draw network
if os.path.isfile(os.path.join(rootpth, midfolder, 'phagcn_graph.csv')):
    drop_network('phagcn', rootpth, midfolder, db_dir, out_dir)
if os.path.isfile(os.path.join(rootpth, midfolder, 'cherry_graph.csv')):
    drop_network('cherry', rootpth, midfolder, db_dir, out_dir)

#### download files
# protein files 
blast_df = pd.read_csv(f"{rootpth}/{midfolder}/phamer_results.abc", sep=' ', names=['query', 'ref', 'evalue'])
protein2evalue = parse_evalue(blast_df, f'{rootpth}/{midfolder}', 'phamer')
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
