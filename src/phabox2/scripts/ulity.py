import  os
import  re
import shutil
import kcounter
import datasets
import  numpy as np
import  pandas as pd
from Bio import SeqIO
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
import pyarrow as pa
import  torch
from    torch import nn
from    torch import optim

from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer
from transformers import DataCollatorWithPadding
from transformers import logging as transformers_logging

import subprocess
# Set logging level to ERROR
transformers_logging.set_verbosity_error()
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.trainer").setLevel(logging.ERROR)


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import math




def run_command(command):
    try:
        # Using subprocess.run to execute the command
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        #print(result.stdout)  # Print the standard output from the command
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with exit code {e.returncode}")
        print(e.stderr)  # Print the standard error from the command
        print("please run the command manually to check the error")
        exit(e.returncode)  # Exit with the non-zero exit code

# draw network for phagcn and cherry
def draw_network(in_fn = './', out_fn='./', task='phagcn'):
    if task == 'phagcn':
        label = 'Genus'
    elif task == 'cherry':
        label = 'Host'
    else:
        print(f'Wrong task name: {task}, ONLY phagcn and cherry is available')
        exit(1)

    # Load data
    edges_df = pd.read_csv(f'{in_fn}/{task}_network_edges.tsv', sep='\t')
    nodes_df = pd.read_csv(f'{in_fn}/{task}_network_nodes.tsv', sep='\t')
    nodes_df = nodes_df.fillna('-')

    # Get unique category and assign colors
    category = nodes_df[label].fillna('-').unique()
    palette = sns.color_palette('hsv', len(category))
    color_map = {item: palette[i] for i, item in enumerate(category)}
    color_map['-'] = 'grey'

    acc2label = {acc: label for acc, label in zip(nodes_df['Accession'], nodes_df[label])}
    acc2type = {acc: node_type for acc, node_type in zip(nodes_df['Accession'], nodes_df['TYPE'])}

    # Create a main graph
    G_main = nx.Graph()

    # Add edges to the graph
    for _, row in edges_df.iterrows():
        if row['Source'] != row['Target']:
            G_main.add_edge(row['Source'], row['Target'], weight=2)

    # Find connected components and sort by size
    connected_components = sorted(nx.connected_components(G_main), key=len, reverse=True)

    # Filter components to ensure they contain 'Query' nodes
    largest_subgraphs = []
    for component in connected_components:
        subgraph = G_main.subgraph(component).copy()
        #largest_subgraphs.append(subgraph)
        if any(acc2type[node] == 'Query' for node in subgraph.nodes()):
            largest_subgraphs.append(subgraph)
        if len(largest_subgraphs) == 10:
            break

    # Determine subplot grid size
    num_subgraphs = len(largest_subgraphs)
    cols = 5
    rows = math.ceil(num_subgraphs / cols)

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten()

    # Collect category used in the plot
    used_category = set()

    for i, ax in enumerate(axes):
        if i < num_subgraphs:
            subgraph = largest_subgraphs[i]
            positions = nx.spring_layout(subgraph, seed=0, k=0.5)

            # Draw edges
            nx.draw_networkx_edges(subgraph, positions, alpha=0.08, width=0.8, ax=ax)

            # Draw nodes with different shapes and sizes
            ref_nodes = [node for node in subgraph.nodes() if acc2type[node] == 'Ref']
            query_nodes = [node for node in subgraph.nodes() if acc2type[node] == 'Query']

            ref_colors = [color_map[acc2label[node]] for node in ref_nodes]
            query_colors = [color_map[acc2label[node]] for node in query_nodes]

            nx.draw_networkx_nodes(subgraph, positions, nodelist=ref_nodes, node_size=50, node_color=ref_colors, ax=ax, node_shape='o')
            nx.draw_networkx_nodes(subgraph, positions, nodelist=query_nodes, node_size=100, node_color=query_colors, ax=ax, node_shape='^')

            ax.set_title(f"Subgraph {i + 1}", fontsize=12)
            ax.axis('off')

            # Collect used category
            used_category.update(acc2label[node] for node in subgraph.nodes())
        else:
            ax.axis('off')

    # Create a single legend with used colors
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label], markersize=10) for label in used_category]
    labels = [label if label != '-' else 'Unclassified' for label in used_category]

    # Add the legend below the subplots
    fig.legend(handles, labels, loc='lower center', title=label, fontsize='large', ncol=5, title_fontsize='large')

    # Adjust layout to make space for the legend
    plt.subplots_adjust(bottom=0.2)

    # Show the plot
    #plt.show()
    plt.savefig(f'{out_fn}/{task}.png',dpi=300)
    plt.close() 

# logger for user feedback
def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(stream_handler)
    return logger


class Genome:
    def __init__(self):
        pass


class Gene:
    def __init__(self):
        pass




def check_path(pth):
    if not os.path.isdir(pth):
        os.makedirs(pth)

def reset_model(Transformer, src_vocab_size, device):
    model = Transformer(
                src_vocab_size=src_vocab_size, 
                src_pad_idx=0, 
                device=device, 
                max_length=300, 
                dropout=0.1
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.BCEWithLogitsLoss()
    return model, optimizer, loss_func


def return_tensor(var, device):
    return torch.from_numpy(var).to(device)


def reject_prophage(all_pred, weight, reject):
    all_pred = np.array(all_pred.detach().cpu())
    all_pred[weight < reject] = 0
    return all_pred


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def init_bert(rootpth, midfolder, parampth):

    bert_feat = pd.read_csv(f'{rootpth}/{midfolder}/bert_feat.csv')
    test  = pa.Table.from_pandas(bert_feat)
    test  = datasets.Dataset(test)
    data = datasets.DatasetDict({"test": test})


    tokenizer = BertTokenizer.from_pretrained(f'{parampth}/bert_config', do_basic_tokenize=False)
    tokenized_data = data.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, desc="")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(f'{parampth}/bert', num_labels=2)


    training_args = TrainingArguments(
        output_dir=f'{rootpth}/{midfolder}/bert_model_out',
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        learning_rate=2e-5,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["test"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return trainer, tokenized_data


def special_match(strg, search=re.compile(r'[^ACGT]').search):
    return not bool(search(strg))

def recruit_phage_file(rootpth, midfolder, pred_phage_dict=None, filein=None):
    if pred_phage_dict is not None:
        phage_rec = []
        for record in SeqIO.parse(f'{rootpth}/filtered_contigs.fa', 'fasta'):
            try:
                pred_phage_dict[record.id]
                phage_rec.append(record)
            except:
                continue

        SeqIO.write(phage_rec, f"{rootpth}/phage_contigs.fa", "fasta")
    else:
        shutil.copyfile(f"{rootpth}/{filein}", f"{rootpth}/phage_contigs.fa")


    records = []
    check_phage = {}
    for record in SeqIO.parse(f'{rootpth}/phage_contigs.fa', 'fasta'):
        seq = str(record.seq)
        seq = seq.upper()
        #if special_match(seq) and len(seq) > 2000: # Nov 17th
        if len(seq) > 2000:
            check_phage[record.id] = 1
            records.append(record)



    phage_protein_rec = []
    for record in SeqIO.parse(f'{rootpth}/{midfolder}/test_protein.fa', 'fasta'):
        try:
            name = record.id
            name = name.rsplit('_', 1)[0]
            check_phage[name]
            phage_protein_rec.append(record)
        except:
            continue

    return records, phage_protein_rec


def generate_gene2genome(inpth, outpth, tool, rootpth):
    blastp = pd.read_csv(f'{inpth}/{tool}_results.abc', sep=' ', names=["contig", "ref", "e-value"])
    protein_id = sorted(list(set(blastp["contig"].values)))
    contig_id = [item.rsplit("_", 1)[0] for item in protein_id]
    description = ["hypothetical protein" for item in protein_id]
    gene2genome = pd.DataFrame({"protein_id": protein_id, "contig_id": contig_id ,"keywords": description})
    gene2genome.to_csv(f"{outpth}/{tool}_contig_gene_to_genome.csv", index=None)


"""
def make_protein_clusters_mcl(abc_fp, out_p, inflation=2):
    print("Running MCL...")
    abc_fn = "merged"
    mci_fn = '{}.mci'.format(abc_fn)
    mci_fp = os.path.join(out_p, mci_fn)
    mcxload_fn = '{}_mcxload.tab'.format(abc_fn)
    mcxload_fp = os.path.join(out_p, mcxload_fn)
    subprocess.check_call("mcxload -abc {0} --stream-mirror --stream-neg-log10 -stream-tf 'ceil(200)' -o {1}"
                          " -write-tab {2}".format(abc_fp, mci_fp, mcxload_fp), shell=True)
    mcl_clstr_fn = "{0}_mcl{1}.clusters".format(abc_fn, int(inflation*10))
    mcl_clstr_fp = os.path.join(out_p, mcl_clstr_fn)
    subprocess.check_call("mcl {0} -I {1} -use-tab {2} -o {3}".format(
        mci_fp, inflation, mcxload_fp, mcl_clstr_fp), shell=True)
    return mcl_clstr_fp



def load_mcl_clusters(fi):
    # Read MCL
    with open(fi) as f:
        c = [line.rstrip("\n").split("\t") for line in f]
    c = [x for x in c if len(c) > 1]
    nb_clusters = len(c)
    formatter = "PC_{{:>0{}}}".format(int(round(np.log10(nb_clusters))+1))
    name = [formatter.format(str(i)) for i in range(nb_clusters)]
    size = [len(i) for i in c]
    clusters_df = pd.DataFrame({"size": size, "pc_id": name}).set_index("pc_id")
    return clusters_df, name, c


def build_clusters(fp, gene2genome):
    # Read MCL
    clusters_df, name, c = load_mcl_clusters(fp)
    print("Using MCL to generate PCs.")
    # Assign each prot to its cluster
    gene2genome.set_index("protein_id", inplace=True)  # id, contig, keywords, cluster
    for prots, clust in zip(c, name):
        try:
            gene2genome.loc[prots, "cluster"] = clust
        except KeyError:
            prots_in = [p for p in prots if p in gene2genome.index]
            not_in = frozenset(prots) - frozenset(prots_in)
            print("{} protein(s) without contig: {}".format(len(not_in), not_in))
            gene2genome.loc[prots_in, "cluster"] = clust
    # Keys
    for clust, prots in gene2genome.groupby("cluster"):
        clusters_df.loc[clust, "annotated"] = prots.keywords.count()
        if prots.keywords.count():
            keys = ";".join(prots.keywords.dropna().values).split(";")
            key_count = {}
            for k in keys:
                k = k.strip()
                try:
                    key_count[k] += 1
                except KeyError:
                    key_count[k] = 1
            clusters_df.loc[clust, "keys"] = "; ".join(["{} ({})".format(x, y) for x, y in key_count.items()])
    gene2genome.reset_index(inplace=True)
    clusters_df.reset_index(inplace=True)
    profiles_df = gene2genome.loc[:, ["contig_id", "cluster"]].drop_duplicates()
    profiles_df.columns = ["contig_id", "pc_id"]
    contigs_df = pd.DataFrame(gene2genome.fillna(0).groupby("contig_id").count().protein_id)
    contigs_df.index.name = "contig_id"
    contigs_df.columns = ["proteins"]
    contigs_df.reset_index(inplace=True)
    return gene2genome, clusters_df, profiles_df, contigs_df


def build_pc_matrices(profiles, contigs, pcs):
    pc_by_cont = profiles.groupby("contig_id").count().pc_id
    pc_by_cont = pd.merge(contigs.sort_values("pos").loc[:, ["pos", "contig_id", "proteins"]], pc_by_cont.to_frame(), how="left",
                          left_on="contig_id", right_on="contig_id").fillna(0)
    singletons = (pc_by_cont.proteins - pc_by_cont.pc_id).values
    singletons = sparse.lil_matrix(singletons).transpose()
    # Matrix
    profiles.index.name = "pos"
    profiles.reset_index(inplace=True)
    # pc_id or contig?
    profiles = pd.merge(profiles, pcs.loc[:, ["pc_id", "pos"]], left_on="pc_id", right_on="pc_id", how="inner",
                            suffixes=["", "_pc"])  # pos, contig_id, pc_id, id (pc), pos_pc
    profiles = pd.merge(profiles, contigs.loc[:, ["contig_id", "pos"]], left_on="contig_id", right_on="contig_id", how="inner",
                            suffixes=["", "_contig"])
    profiles = profiles.loc[:, ["pos_contig", "pos_pc"]]
    matrix = sparse.coo_matrix(([1]*len(profiles), (zip(*profiles.values))), shape=(len(contigs), len(pcs)),
                               dtype="bool")
    return matrix.tocsr(), singletons.tocsr()

def create_network(matrix, singletons, thres=1, max_sig=1000):
    contigs, pcs = matrix.shape
    pcs += singletons.sum()
    # Number of comparisons
    T = 0.5 * contigs * (contigs - 1)
    logT = np.log10(T)
    # Number of protein clusters in each contig
    # = # shared pcs + #singletons
    number_of_pc = matrix.sum(1) + singletons
    number_of_pc = number_of_pc.A1  # Transform into a flat array
    # Number of common protein clusters between two contigs, tuple + commons
    commons_pc = matrix.dot(sparse.csr_matrix(matrix.transpose(), dtype=int))
    S = sparse.lil_matrix((contigs, contigs))
    total_c = float(commons_pc.getnnz())
    i = 0  # Display
    for A, B in zip(*commons_pc.nonzero()):  # For A & B sharing contigs
        if A != B:
            # choose(a, k) * choose(C - a, b - k) / choose(C, b)
            # sf(k) = survival function = 1 -cdf(k) = 1 - P(x<k) = P(x>k)
            # sf(k-1)= P(x>k-1) = P(x>=k)
            # It is symmetric but I put the smallest before to avoid numerical bias.
            a, b = sorted([number_of_pc[A], number_of_pc[B]])
            pval = stats.hypergeom.sf(commons_pc[A, B] - 1, pcs, a, b)
            sig = min(max_sig, np.nan_to_num(-np.log10(pval) - logT))
            if sig > thres:
                S[min(A, B), max(A, B)] = sig
            # Display
            i += 1
            if i % 1000 == 0:
                sys.stdout.write(".")
            if i % 10000 == 0:
                sys.stdout.write("{:6.2%} {}/{}\n".format(i / total_c, i, total_c))
    S += S.T  # Symmetry
    S = S.tocsr()
    if len(S.data) != 0:
        print("Hypergeometric contig-similarity network:\n {0:10} contigs,\n {1:10} edges (min:{2:.2}"
                    "max: {3:.2}, threshold was {4})".format(contigs, S.getnnz(), S.data.min(), S.data.max(), thres))
    else:
        raise ValueError("No edge in the similarity network !") 
    return S


def to_clusterer(matrix, fi, contigs=None,names=None):
    names = contigs if names is None else names
    names = names.set_index("pos").contig_id
    with open(fi, "wt") as f:
        matrix = sparse.dok_matrix(matrix)
        for r, c in zip(*matrix.nonzero()):
            f.write(" ".join([str(x) for x in (names[r], names[c], matrix[r, c])]))
            f.write("\n")
    print("Saving network in file {0} ({1} lines).".format(fi, matrix.getnnz()))
    return fi


def return_4mer(file_in_fn):
    # alphbet
    k_list = ["A", "C", "G", "T"]
    nucl_list = ["A", "C", "G", "T"]
    for i in range(3):
        tmp = []
        for item in nucl_list:
            for nucl in k_list:
                tmp.append(nucl+item)
        k_list = tmp
    # dictionary
    mer2dict = {mer: idx for idx, mer in enumerate(k_list)}
    # search files
    file_list = os.listdir(file_in_fn)
    num_file = len(file_list)
    file2idx = {}
    # convert to words
    feature = np.zeros((num_file, 256))
    for idx, file in enumerate(file_list):
        file2idx[file.rsplit('.', 1)[0]] = idx
        for record in SeqIO.parse(file_in_fn + file, 'fasta'):
            seq = str(record.seq)
            for pos in range(len(seq)-3):
                try:
                    feature[idx][mer2dict[seq[pos:pos+4]]] += 1
                except:
                    #print(seq[pos:pos+4])
                    pass
    # nomarlization
    norm_feature = np.zeros((num_file, 256))
    for i in range(len(feature)):
        norm_feature[i] = (feature[i] - np.min(feature[i]))/(np.max(feature[i]) - np.min(feature[i]))
    return norm_feature, file2idx
"""

#############################################################
########################  output  ###########################
#############################################################

def convert_output_evalue(num):
    num = float(num)
    if num == 0:
        return '0'
    str_num = f"{num:.1e}"
    str_num = str_num.split('e-')
    str_num = ''.join([str_num[0][:3], 'xxx-', str_num[1]])
    return str_num


##############################################################
#####################  PhaGCN exception ######################
##############################################################
"""
def phagcn_exception(rootpth, midfolder, visual, out_dir, ID2length, inputs, fasta='filtered_contigs.fa'):
    if os.path.getsize(f'{rootpth}/{midfolder}/unknown_out.tab') != 0:
        with open(f'{rootpth}/{midfolder}/unknown_out.tab') as file_out:
            check_name_single = {}
            Accession = []
            Pred = []
            Pred_tmp = []
            Score = []
            Length_list = []
            for line in file_out.readlines():
                parse = line.replace("\n", "").split("\t")
                virus = parse[0]
                target = parse[1]
                target = target.split('|')[1]
                ident  = float(parse[-3])/100
                length = float(parse[-2])
                qlen   = float(parse[-1])
                tmp_score = (length/qlen)*ident
                if tmp_score < 0.2:
                    continue
                tmp_score  = f"{tmp_score:.3f}"
                if virus in check_name_single:
                    continue
                check_name_single[virus] = 1
                Accession.append(virus)
                Pred.append(f'no_family_avaliable({target})')
                Score.append(tmp_score)
                Length_list.append(ID2length[virus])
                Pred_tmp.append(f'no_family_avaliable')

            if Accession:
                with open(f'{rootpth}/{visual}/phage_flag.txt', 'w') as file_out:
                    file_out.write('phage_flag\n')
                # add unknown label
                unknown_acc = []
                unknown_length_list = []
                for record in SeqIO.parse(f'{rootpth}/{fasta}', 'fasta'):
                    if len(record.seq) < inputs.len:
                        continue
                    if record.id not in Accession:
                        unknown_acc.append(record.id)
                        unknown_length_list.append(len(record.seq))

                Accession   = Accession+unknown_acc
                Length_list = Length_list+unknown_length_list
                Pred        = Pred + ['unknown']*len(unknown_acc)
                Score       = Score + ['0']*len(unknown_acc)
                Pred_tmp    = Pred_tmp + ['unknown']*len(unknown_acc)
                
                df = pd.DataFrame({"ID": [item+1 for item in range(len(Accession))], "Accession": Accession, "Length": Length_list, "PhaGCN":Pred, "PhaGCN_score":Score, "Pielist": Pred_tmp})
                df.to_csv(f'{rootpth}/{visual}/contigtable.csv', index=False)
                cnt = Counter(df['Pielist'].values)
                pred_dict = {}
                for key, value in zip(cnt.keys(), cnt.values()):
                    pred_dict[key] = value
                pkl.dump(pred_dict, open(f"{rootpth}/visual/phagcn_pred.dict", 'wb'))
                df = pd.DataFrame({"Accession": Accession, "Pred":Pred, "Score":Score})
                df.to_csv(f"{rootpth}/{out_dir}/phagcn_prediction.csv", index = None)
                with open(f'{rootpth}/{visual}/no_family_flag.txt', 'w') as file_out:
                    file_out.write('no_family_flag\n')
            else:
                # unknow label
                Accession = []
                Length_list = []
                for record in SeqIO.parse(f'{rootpth}/{fasta}', 'fasta'):
                    if len(record.seq) < inputs.len:
                        continue
                    Accession.append(record.id)
                    Length_list.append(len(record.seq))
                    Pred_tmp.append('unknown')
                df = pd.DataFrame({"ID": [item+1 for item in range(len(Accession))], "Accession": Accession, "Length": Length_list, "PhaGCN":['unknown']*len(Accession), "PhaGCN_score":[0]*len(Accession), "Pielist": Pred_tmp})
                df.to_csv(f'{rootpth}/{visual}/contigtable.csv', index=False)
                cnt = Counter(df['Pielist'].values)
                pred_dict = {}
                for key, value in zip(cnt.keys(), cnt.values()):
                    pred_dict[key] = value
                pkl.dump(pred_dict, open(f"{rootpth}/visual/phagcn_pred.dict", 'wb'))
                df = pd.DataFrame({"Accession": Accession, "Pred":['unknown']*len(Accession), "Score":[0]*len(Accession)})
                df.to_csv(f"{rootpth}/{out_dir}/phagcn_prediction.csv", index = None)
                with open(f'{rootpth}/{visual}/no_family_flag.txt', 'w') as file_out:
                    file_out.write('no_family_flag\n')
                with open(f'{rootpth}/{visual}/phage_flag.txt', 'w') as file_out:
                    file_out.write('phage_flag\n')
    else:
        # unknow label
        Accession = []
        Length_list = []
        Pred_tmp = []
        for record in SeqIO.parse(f'{rootpth}/{fasta}', 'fasta'):
            Accession.append(record.id)
            Length_list.append(len(record.seq))
            Pred_tmp.append('unknown')
        df = pd.DataFrame({"ID": [item+1 for item in range(len(Accession))], "Accession": Accession, "Length": Length_list, "PhaGCN":['unknown']*len(Accession), "PhaGCN_score":[0]*len(Accession), "Pielist": Pred_tmp})
        df.to_csv(f'{rootpth}/{visual}/contigtable.csv', index=False)
        cnt = Counter(df['Pielist'].values)
        pred_dict = {}
        for key, value in zip(cnt.keys(), cnt.values()):
            pred_dict[key] = value
        pkl.dump(pred_dict, open(f"{rootpth}/visual/phagcn_pred.dict", 'wb'))
        df = pd.DataFrame({"Accession": Accession, "Pred":['unknown']*len(Accession), "Score":[0]*len(Accession)})
        df.to_csv(f"{rootpth}/{out_dir}/phagcn_prediction.csv", index = None)
        with open(f'{rootpth}/{visual}/no_family_flag.txt', 'w') as file_out:
            file_out.write('no_family_flag\n')
        with open(f'{rootpth}/{visual}/phage_flag.txt', 'w') as file_out:
            file_out.write('phage_flag\n')
"""

"""
def phagcn_exception_no_visual(rootpth, midfolder, out_dir, ID2length, inputs, fasta='filtered_contigs.fa'):
    if os.path.getsize(f'{rootpth}/{midfolder}/unknown_out.tab') != 0:
        with open(f'{rootpth}/{midfolder}/unknown_out.tab') as file_out:
            check_name_single = {}
            Accession = []
            Pred = []
            Pred_tmp = []
            Score = []
            Length_list = []
            for line in file_out.readlines():
                parse = line.replace("\n", "").split("\t")
                virus = parse[0]
                target = parse[1]
                target = target.split('|')[1]
                ident  = float(parse[-3])/100
                length = float(parse[-2])
                qlen   = float(parse[-1])
                tmp_score = (qlen/length)*ident
                if tmp_score < 0.2:
                    continue
                tmp_score  = f"{tmp_score:.3f}"
                if virus in check_name_single:
                    continue
                check_name_single[virus] = 1
                Accession.append(virus)
                Pred.append(f'no_family_avaliable({target})')
                Score.append(tmp_score)
                Length_list.append(ID2length[virus])
                Pred_tmp.append(f'no_family_avaliable')

            if Accession:
                # add unknown label
                unknown_acc = []
                unknown_length_list = []
                for record in SeqIO.parse(f'{rootpth}/{fasta}', 'fasta'):
                    if len(record.seq) < inputs.len:
                        continue
                    if record.id not in Accession:
                        unknown_acc.append(record.id)
                        unknown_length_list.append(len(record.seq))

                Accession   = Accession+unknown_acc
                Length_list = Length_list+unknown_length_list
                Pred        = Pred + ['unknown']*len(unknown_acc)
                Score       = Score + ['0']*len(unknown_acc)
                Pred_tmp    = Pred_tmp + ['unknown']*len(unknown_acc)
                
                df = pd.DataFrame({"Accession": Accession, "Pred":Pred, "Score":Score})
                df.to_csv(f"{rootpth}/{out_dir}/phagcn_prediction.csv", index = None)

            else:
                # unknow label
                Accession = []
                Length_list = []
                for record in SeqIO.parse(f'{rootpth}/{fasta}', 'fasta'):
                    if len(record.seq) < inputs.len:
                        continue
                    Accession.append(record.id)
                    Length_list.append(len(record.seq))
                    Pred_tmp.append('unknown')
                
                df = pd.DataFrame({"Accession": Accession, "Pred":['unknown']*len(Accession), "Score":[0]*len(Accession)})
                df.to_csv(f"{rootpth}/{out_dir}/phagcn_prediction.csv", index = None)

    else:
        # unknow label
        Accession = []
        Length_list = []
        Pred_tmp = []
        for record in SeqIO.parse(f'{rootpth}/{fasta}', 'fasta'):
            Accession.append(record.id)
            Length_list.append(len(record.seq))
            Pred_tmp.append('unknown')
        
        df = pd.DataFrame({"Accession": Accession, "Pred":['unknown']*len(Accession), "Score":[0]*len(Accession)})
        df.to_csv(f"{rootpth}/{out_dir}/phagcn_prediction.csv", index = None)
"""



def compute_aai(pth, file_name, genome_size):
    # load the diamond output
    df = pd.read_csv(f'{pth}/{file_name}.abc', sep=' ', names=['qseqid', 'sseqid', 'pident', 'bitscore'])
    if file_name == 'db_results':
        swapped_df = df.copy()
        swapped_df['qseqid'], swapped_df['sseqid'] = df['sseqid'], df['qseqid']
        # Append the new entries to the original DataFrame
        df = pd.concat([df, swapped_df], ignore_index=True)
    # add the genome information for both query and target
    df['query'] = df['qseqid'].apply(lambda x: x.rsplit('_', 1)[0])
    df['target'] = df['sseqid'].apply(lambda x: x.rsplit('_', 1)[0])
    # maintain the best hit
    df = df.drop_duplicates(['query', 'target', 'qseqid'], keep='first')
    # group by query and target and compute AAI and shared genes
    tmp_df = df.groupby(['query', 'target']).agg({'pident': 'mean', 'qseqid': 'count'}).reset_index()
    # replace name pident and qseqid into aai and sgenes
    tmp_df = tmp_df.rename(columns={'pident': 'aai', 'qseqid': 'sgenes'})
    # compute coverage
    # Calculate qcov and tcov vectorized
    tmp_df['qcov'] = 100.0 * tmp_df['sgenes'] / tmp_df['query'].map(genome_size)
    tmp_df['tcov'] = 100.0 * tmp_df['sgenes'] / tmp_df['target'].map(genome_size)
    # Add query_genes and target_genes vectorized
    tmp_df['qgenes'] = tmp_df['query'].map(genome_size)
    tmp_df['tgenes'] = tmp_df['target'].map(genome_size)
    tmp_df.to_csv(f'{pth}/{file_name}_aai.tsv', sep='\t', index=False)



def parse_alignment(alignment_file):
    # Read the TSV file with column names
    df = pd.read_csv(alignment_file, sep=" ", names=['qseqid', 'sseqid', 'pident', 'bitscore'])
    df['bitscore'] = df['bitscore'].astype(float)
    ORF2hits = {}
    all_hits = set()
    # Group by the 'qseqid' column
    grouped = df.groupby('qseqid')
    for ORF, group in grouped:
        top_bitscore = group['bitscore'].max()
        hits = group[group['bitscore'] >= 0.9 * top_bitscore]
        ORF2hits[ORF] = list(zip(hits['sseqid'], hits['bitscore']))
        all_hits.update(hits['sseqid'])
    return ORF2hits, all_hits
    





def import_nodes(nodes_dmp):
    # Read the file into a DataFrame
    # df = pd.read_csv(nodes_dmp, sep="\t|\t", engine='python', header=None, usecols=[0, 2, 4])
    # Assign column names
    # df.columns = ['taxid', 'parent', 'rank']
    # Create dictionaries
    df = pd.read_csv(nodes_dmp)
    taxid2parent = df.set_index('taxid')['parent'].to_dict()
    taxid2rank = df.set_index('taxid')['rank'].to_dict()
    return taxid2parent, taxid2rank



def import_names(names_dmp):
    # Read the file into a DataFrame
    # df = pd.read_csv(names_dmp, sep="\t|\t", engine='python', header=None, usecols=[0, 2, 4])
    # Assign column names
    # df.columns = ['taxid', 'name', 'rank']
    # Create dictionaries
    df = pd.read_csv(names_dmp)
    taxid2name = df.set_index('taxid')['name'].to_dict()
    return taxid2name


def find_lineage(taxid, taxid2parent):
    lineage = []
    while True:
        lineage.append(taxid)
        parent = taxid2parent[taxid]
        if parent == taxid:
            break
        taxid = parent
    return lineage




def find_LCA_for_ORF(hits, fastaid2LCAtaxid, taxid2parent):
    list_of_lineages = []
    top_bitscore = 0
    for hit, bitscore in hits:
        if bitscore > top_bitscore:
            top_bitscore = bitscore
        taxid = fastaid2LCAtaxid.get(hit)
        if taxid:
            lineage = find_lineage(taxid, taxid2parent)
            list_of_lineages.append(lineage)

    if not list_of_lineages:
        return ("no taxid found ({0})".format(";".join([i[0] for i in hits])), top_bitscore)
    # Find the intersection of all lineages
    overlap = set.intersection(*map(set, list_of_lineages))
    # Return the first common taxid
    for taxid in list_of_lineages[0]:
        if taxid in overlap:
            return (taxid, top_bitscore)

    return ("no common ancestor found", top_bitscore)



def find_weighted_LCA(orf_hits, taxid2parent, threshold):
    # Store lineages and corresponding bit scores
    lineages = []
    bit_scores = []
    total_bit_score = 0
    # Process each ORF hit
    for taxid, bit_score in orf_hits:
        if type(taxid) == int:
            lineage = find_lineage(taxid, taxid2parent)
            lineages.append(lineage)
            bit_scores.append(bit_score)
            total_bit_score += bit_score
    # Check if any lineages were found
    if not lineages:
        return "no ORFs with taxids found.", None
    # Calculate cumulative bit scores for each taxid
    taxid_to_cumulative_score = {}
    for lineage, bit_score in zip(lineages, bit_scores):
        for taxid in lineage:
            taxid_to_cumulative_score[taxid] = taxid_to_cumulative_score.get(taxid, 0) + bit_score
    # Filter lineages based on threshold
    valid_lineages = []
    for taxid in taxid_to_cumulative_score:
        cumulative_score = taxid_to_cumulative_score[taxid] / total_bit_score
        if cumulative_score > threshold:
            lineage = find_lineage(taxid, taxid2parent)
            scores = [taxid_to_cumulative_score[t] / total_bit_score for t in lineage]
            valid_lineages.append((lineage, scores))
    # Check if any valid lineages were found
    if not valid_lineages:
        return "no lineage larger than threshold.", None
    # Sort by length and score, then select the best lineage
    valid_lineages.sort(key=lambda x: (len(x[0]), sum(x[1])), reverse=True)
    best_lineage, best_score = valid_lineages[0]
    return best_lineage, best_score[::-1]




def convert_lineage_to_names(lineage, taxid2name, taxid2rank):
    named_lineage = []
    for taxid in lineage:
        if taxid == 1:
            continue
        name = taxid2name.get(taxid, 'unknown')
        if name == 'Tailed phages':
            name = 'Caudoviricetes'
        rank = taxid2rank.get(taxid, 'unknown')
        if rank == 'species':
            continue
        if rank == 'no rank':
            continue
        named_lineage.append(f"{rank}:{name}")
    return ";".join(named_lineage[::-1])



def annotate_genes(hmm_info_df, genomes, genes, align_df):
    # Define the label for each category
    label = {"viral": 1, "microbial": -1}
    for query, target, evalue in zip(align_df["query"], align_df["target"], align_df["evalue"]):
        try:
            gene = genes[query]
            gene.label = label[hmm_info_df[target]['category']]
            gene.marker_hit = {'target': target, 'evalue': evalue, 'function': hmm_info_df[target]['function']}
        except KeyError:
            pass
        
    # Count the number of viral and host genes in each genome
    for genome in genomes.values():
        genome.count_viral = sum(genes[_].label == 1 for _ in genome.genes)
        genome.count_host = sum(genes[_].label == -1 for _ in genome.genes)


def compute_delta(my_genes, s1, e1, s2, e2, gc_weight):
    # Extend windows to ensure at least 1 annotated gene
    if all(g.label == 0 for g in my_genes[s1:e1]):
        s1 = next((j for j in range(s1 - 1, -1, -1) if my_genes[j].label != 0), s1)
    if all(g.label == 0 for g in my_genes[s2:e2]):
        e2 = next((j for j in range(e2 + 1, len(my_genes)) if my_genes[j].label != 0), e2)

    # Get gene values for 2 windows
    win1 = my_genes[s1:e1]
    win2 = my_genes[s2:e2]
    v1 = [g.label for g in win1 if g.label != 0]
    v2 = [g.label for g in win2 if g.label != 0]
    g1 = [g.gc for g in win1]
    g2 = [g.gc for g in win2]

    # Compute delta between windows
    if v1 and v2:
        delta_v = np.mean(v1) - np.mean(v2)
        delta_g = abs(np.mean(g1) - np.mean(g2))
        delta = (abs(delta_v) + delta_g * gc_weight) * np.sign(delta_v)
    else:
        delta = 0

    # Store results
    d = {
        "delta": delta,
        "coords": [s1, e1, s2, e2],
        "v1": v1,
        "v2": v2,
        "v1_len": len(v1),
        "v2_len": len(v2),
        "win1_len": len(win1),
        "win2_len": len(win2),
        "win1_fract_host": len([_ for _ in v1 if _ == -1]) / len(win1) if win1 else 0,
        "win2_fract_host": len([_ for _ in v2 if _ == -1]) / len(win2) if win2 else 0,
    }

    return d


def define_regions(genome, genes, gc_weight, delta_cutoff, min_host_fract, min_host_genes, min_viral_genes):
    # Fetch genes
    my_genes = [genes[_] for _ in genome.genes]

    # Determine window size
    win_size = min(max(15, round(0.30 * len(my_genes))), 50)

    # Identify breakpoints
    breaks = []
    while True:
        s1 = 0 if not breaks else breaks[-1]["coords"][-2]

        # Determine window coordinates (gene indexes)
        coords = [
            [s1, i, i, min(i + win_size, len(my_genes))]
            for i in range(1, len(my_genes))
            if (i - s1 >= win_size or s1 == 0) and (min(i + win_size, len(my_genes)) - i >= win_size or min(i + win_size, len(my_genes)) == len(my_genes))
        ]

        # Score each possible breakpoint
        deltas = [compute_delta(my_genes, s1, e1, s2, e2, gc_weight) for s1, e1, s2, e2 in coords]

        # Filter each possible breakpoint
        filtered = [
            d for d in deltas
            if (
                abs(d["delta"]) >= delta_cutoff and
                ((d["delta"] < 0 and d["v1"].count(-1) > 0 and d["v2"].count(1) > 0) or
                 (d["delta"] > 0 and d["v1"].count(1) > 0 and d["v2"].count(-1) > 0)) and
                (d["delta"] < 0 and d["v1"].count(-1) >= min_host_genes or
                 d["delta"] > 0 and d["v2"].count(-1) >= min_host_genes or
                 len(my_genes) <= 10) and
                (d["delta"] > 0 and d["v1"].count(1) >= min_viral_genes or
                 d["delta"] < 0 and d["v2"].count(1) >= min_viral_genes or
                 len(my_genes) <= 10) and
                (d["win1_fract_host"] >= min_host_fract or d["win2_fract_host"] >= min_host_fract)
            )
        ]

        # Select breakpoint
        selected = None
        for d in filtered:
            if selected is None or (np.sign(d["delta"]) == np.sign(selected["delta"]) and abs(d["delta"]) > abs(selected["delta"])):
                selected = d
            else:
                break

        if selected is None:
            break
        else:
            breaks.append(selected)

    # Update last break so end coord is contig end
    if breaks:
        breaks[-1]["coords"][-1] = len(my_genes)

    # Define regions based on breakpoints
    regions = []
    for b in breaks:
        s1, e1, s2, e2 = b["coords"]
        d = compute_delta(my_genes, s1, e1, s2, e2, gc_weight)
        region = {
            "delta": d["delta"],
            "type": "host" if d["delta"] < 0 else "viral",
            "start_gene": s1,
            "end_gene": e1,
            "start_pos": regions[-1]["end_pos"] + 1 if regions else 1,
            "end_pos": my_genes[e1 - 1].end,
            "size": e1 - s1,
            "length": my_genes[e1 - 1].end - (regions[-1]["end_pos"] + 1 if regions else 1) + 1,
            "host_genes": d["v1"].count(-1),
            "viral_genes": d["v1"].count(1),
        }
        regions.append(region)

    # Handle last region
    if regions:
        s1, e1 = regions[-1]["start_gene"], regions[-1]["end_gene"]
        s2, e2 = e1, len(genome.genes)
        d = compute_delta(my_genes, s1, e1, s2, e2, gc_weight)
        region = {
            "type": "viral" if regions[-1]["type"] == "host" else "host",
            "start_gene": s2,
            "end_gene": e2,
            "start_pos": regions[-1]["end_pos"] + 1,
            "end_pos": genome.length,
            "size": e2 - s2,
            "length": genome.length - (regions[-1]["end_pos"] + 1) + 1,
            "host_genes": d["v2"].count(-1),
            "viral_genes": d["v2"].count(1),
        }
        regions.append(region)

    return regions



def get_average_kmer_freq(genome):
    counts = list(kcounter.count_kmers(genome.seq, 21, canonical_kmers=True).values())
    return round(np.mean(counts), 2)


def run_crt(db_dir, inpth, outpth, bfile):
    cmd = ["java", "-cp", f"{db_dir}/CRT1.2-CLI.jar", "crt", inpth, outpth]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return f"{bfile}"


def is_valid_dna_sequence(sequence):
    valid_bases = {'A', 'T', 'C', 'G', 'N'}
    return all(base in valid_bases for base in sequence.upper())


# ANI
def parse_blast(input_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            r = line.split()
            if r[0] != r[1]:  # Skip self hits
                data.append({
                    'qname': r[0],
                    'tname': r[1],
                    'pid': float(r[2]),
                    'len': float(r[3]),
                    'qcoords': sorted([int(r[6]), int(r[7])]),
                    'tcoords': sorted([int(r[8]), int(r[9])]),
                    'qlen': float(r[-2]),
                    'tlen': float(r[-1]),
                    'evalue': float(r[-4])
                })
    return pd.DataFrame(data)

def compute_ani(alns):
    return round((alns['len'] * alns['pid']).sum() / alns['len'].sum(), 2)

def compute_cov(alns, coord_col, len_col):
    coords = sorted(alns[coord_col].to_list())
    nr_coords = [coords[0]]
    for start, stop in coords[1:]:
        if start <= (nr_coords[-1][1] + 1):
            nr_coords[-1][1] = max(nr_coords[-1][1], stop)
        else:
            nr_coords.append([start, stop])
    alen = sum([stop - start + 1 for start, stop in nr_coords])
    return round(100.0 * alen / alns[len_col].iloc[0], 2)



def phavip_dump_result(genomes, rootpth, out_dir, logger, supplementary):
    try:
        # summarize the annotation rate for each genome (phavip_prediction.tsv)
        df = pd.read_csv(f'{rootpth}/{out_dir}/{supplementary}/gene_annotation.tsv', sep='\t')
        anno_df = df[df['Annotation'] != 'hypothetical protein (no hit)']
        genome = df['Genome'].unique()
        length = [genomes[item].length for item in genome]
        protein_num = [len(genomes[item].genes) for item in genome]
        anno_num = [len(anno_df[anno_df['Genome'] == item]) for item in genome]
        anno_rate = [item1 / item2 for item1, item2 in zip(anno_num, protein_num)]
        anno_rate = [f'{item:.2f}' for item in anno_rate]

        with open(f'{rootpth}/{out_dir}/phavip_prediction.tsv', 'w') as f:
            f.write('Accession\tLength\tProtein_num\tAnnotated_num\tAnnotation_rate\n')
            for item in zip(genome, length, protein_num, anno_num, anno_rate):
                f.write(f'{item[0]}\t{item[1]}\t{item[2]}\t{item[3]}\t{item[4]}\n')
    except:
        logger.info("phavip failed. Please check the gene annotation file.")

def load_gene_info(file, genomes):
    genes = {}
    for record in SeqIO.parse(file, 'fasta'):
        gene = Gene()
        rec_info = record.description.split()
        gene.id = rec_info[0]
        gene.start = int(rec_info[2])
        gene.end = int(rec_info[4])
        gene.strand = int(rec_info[6])
        gene.genome_id = gene.id.rsplit("_", 1)[0]
        gene.gc = float(rec_info[-1].split('gc_cont=')[-1])
        gene.anno = 'hypothetical protein (no hit)'
        gene.pident = 0
        gene.coverage = 0
        genes[gene.id] = gene
        try:
            genomes[gene.genome_id].genes.append(gene.id)
        except:
            pass
    return genes
