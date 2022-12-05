import  os
import  re
import  sys
import shutil
import  numpy as np
import  pandas as pd
import  pickle as pkl
import  argparse
import  datasets
import subprocess
import  pyarrow as pa
from Bio import SeqIO
import scipy.stats as stats
import scipy.sparse as sparse
import scipy as sp
import networkx as nx
import  torch
from    torch import nn
from    torch.nn import functional as F
from    torch import optim
import  torch.utils.data as Data
from collections import Counter
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer, LineByLineTextDataset
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer




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


def return_batch(train_sentence, label, flag):
    X_train = torch.from_numpy(train_sentence).to(device)
    y_train = torch.from_numpy(label).float().to(device)
    train_dataset = Data.TensorDataset(X_train, y_train)
    training_loader = Data.DataLoader(
        dataset=train_dataset,    
        batch_size=200,
        shuffle=flag,               
        num_workers=0,              
    )
    return training_loader

def return_tensor(var, device):
    return torch.from_numpy(var).to(device)


def reject_prophage(all_pred, weight, reject):
    all_pred = np.array(all_pred.detach().cpu())
    all_pred[weight < reject] = 0
    return all_pred


def masked_loss(out, label, mask):
    if torch.cuda.is_available():
        w = torch.Tensor([1,1,2,3,3,3,3,3,3,3,10,10,10,10,10,10,10,10,10]).cuda()
    else:
        w = torch.Tensor([1,1,2,3,3,3,3,3,3,3,10,10,10,10,10,10,10,10,10])
    loss = F.cross_entropy(out, label, w, reduction='none')
    
    #all phage
    #w = torch.Tensor([3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 3.0, 2.0, 3.0]).cuda()
    #loss = F.cross_entropy(out, label, w, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss


def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc

def phagcn_accuracy(out, mask, labels):
    pred = np.argmax(out, axis = 1)
    mask_pred = np.array([pred[i] for i in range(len(labels)) if mask[i] == True])
    mask_label = np.array([labels[i] for i in range(len(labels)) if mask[i] == True])
    return np.sum(mask_label == mask_pred)/len(mask_pred)



def init_bert(midpth, bert_feat, model_pth, tokenizer, tokenized_data, data_collator):
    
    model = AutoModelForSequenceClassification.from_pretrained(model_pth, num_labels=2)


    training_args = TrainingArguments(
        output_dir=f'{midpth}/ber_model_out',
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        learning_rate=2e-5,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["test"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return trainer


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



def make_protein_clusters_mcl(abc_fp, out_p, inflation=2):
    """
    Args: 
        blast_fp (str): Path to blast results file
        inflation (float): MCL inflation value
        out_p (str): Output directory path
    Returns:
        str: fp for MCL clustering file
    """
    #logger.debug("Generating abc file...")
    #blast_fn = os.path.basename(blast_fp)
    #abc_fn = '{}.abc'.format(blast_fn)
    #abc_fp = os.path.join(out_p, abc_fn)
    #subprocess.check_call("awk '$1!=$2 {{print $1,$2,$11}}' {0} > {1}".format(blast_fp, abc_fp), shell=True)
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
    """
    Load given clusters file
    
    Args:
        fi (str): path to clusters file
        proteins_df (dataframe): A dataframe giving the protein and its contig.
    Returns: 
        tuple: dataframe proteins and dataframe clusters
    """
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
    """
        Build clusters given clusters file

        Args:
            fp (str): filepath of clusters file
            gene2genome (dataframe): A dataframe giving the protein and its genome.
            mode (str): clustering method
        Returns:
            tuple: dataframe of proteins, clusters, profiles and contigs
        """
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
    """
    Build the pc profiles matrices (shared & singletons) from dataframes.

    Args:
        profiles (dataframe): required fields are contig_id and pc_id. # pos, contig_id, pc_id
        contigs (dataframe): contigs info, required field are proteins, pos and id. # pos, contig_id, proteins
        pcs (dataframe): pcs info, required field are pos and id.  # pos, id, size, annotated

    Returns:
        (tuple of sparse matrix): Shared PCs and singletons matrix.
    """
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
    """
    Compute the hypergeometric-similarity contig network.

    Args:
        matrix (scipy.sparse)x: contigs x protein clusters :
            M(c,p) == True <-> PC p is in Contig c.
        thres (float): Minimal significativity to store an edge value.
        max_sig (int): Maximum significance score
        
    Return
        scipy.sparse: S symmetric lil matrix, contigs x contigs.
        S(c,c) = sig(link)
    """
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
    """Save a network in a file ready for MCL and/or ClusterONE

    Args:
        matrix (scipy.sparse_matrix): network.
        fi (str): filename .
        names (pandas.dataframe): with the columns
            "pos":  (int) is the position in the matrix.
            "id": (str) column contain the id of the node.
            If None, self.contigs is used.

    Returns:
        str: filename
    """
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



