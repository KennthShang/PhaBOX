import os
import pandas as pd
import numpy as np
import pickle as pkl
import subprocess
import argparse
import shutil
from shutil import which
from collections import Counter
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


#############################################################
######################  translateion   ######################
#############################################################


def translation(inpth, outpth, infile, outfile, threads, proteins):
    if proteins is None:
        prodigal = "prodigal"
        # check if pprodigal is available
        if which("pprodigal") is not None:
            print("Using parallelized prodigal...")
            prodigal = f'pprodigal -T {threads}'

        prodigal_cmd = f'{prodigal} -i {inpth}/{infile} -a {outpth}/{outfile} -f gff -p meta'
        print("Running prodigal...")
        _ = subprocess.check_call(prodigal_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        shutil.copyfile(proteins, f'{out_fn}/{outfile}')


#############################################################
####################  DIAMOND BLASTP  #######################
#############################################################

def run_diamond(diamond_db, outpth, infile, tool, threads):
    try:
        # running alignment
        diamond_cmd = f'diamond blastp --threads {threads} --sensitive -d {diamond_db} -q {outpth}/{infile} -o {outpth}/{tool}_results.tab -k 1'
        print("Running Diamond...")
        _ = subprocess.check_call(diamond_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        diamond_out_fp = f"{outpth}/{tool}_results.tab"
        database_abc_fp = f"{outpth}/{tool}_results.abc"
        _ = subprocess.check_call("awk '$1!=$2 {{print $1,$2,$11}}' {0} > {1}".format(diamond_out_fp, database_abc_fp), shell=True)
    except:
        print(diamond_cmd)
        print("diamond blastp failed")
        exit(1)




#############################################################
####################  Contig2Sentence  ######################
#############################################################

def contig2sentence(db_dir, outpth, infile, tool):
    # Load dictonary and BLAST results
    proteins_df = pd.read_csv(f'{db_dir}/proteins.csv')
    proteins_df.dropna(axis=0, how='any', inplace=True)
    pc2wordsid = {pc: idx for idx, pc in enumerate(sorted(set(proteins_df['cluster'].values)))}
    protein2pc = {protein: pc for protein, pc in zip(proteins_df['protein_id'].values, proteins_df['cluster'].values)}
    blast_df = pd.read_csv(f"{outpth}/{tool}_results.abc", sep=' ', names=['query', 'ref', 'evalue'])

    # Parse the DIAMOND results
    contig2pcs = {}
    for query, ref, evalue in zip(blast_df['query'].values, blast_df['ref'].values, blast_df['evalue'].values):
        conitg = query.rsplit('_', 1)[0]
        idx    = query.rsplit('_', 1)[1]
        pc     = pc2wordsid[protein2pc[ref]]
        try:
            contig2pcs[conitg].append((idx, pc, evalue))
        except:
            contig2pcs[conitg] = [(idx, pc, evalue)]

    # Sorted by position
    for contig in contig2pcs:
        contig2pcs[contig] = sorted(contig2pcs[contig], key=lambda tup: tup[0])



    # Contigs2sentence
    contig2id = {contig:idx for idx, contig in enumerate(contig2pcs.keys())}
    id2contig = {idx:contig for idx, contig in enumerate(contig2pcs.keys())}
    sentence = np.zeros((len(contig2id.keys()), 300))
    sentence_weight = np.ones((len(contig2id.keys()), 300))
    for row in range(sentence.shape[0]):
        contig = id2contig[row]
        pcs = contig2pcs[contig]
        for col in range(len(pcs)):
            try:
                _, sentence[row][col], sentence_weight[row][col] = pcs[col]
                sentence[row][col] += 1
            except:
                break


    # propotion
    rec = []
    for key in blast_df['query'].values:
        name = key.rsplit('_', 1)[0]
        rec.append(name)
    counter = Counter(rec)
    mapped_num = np.array([counter[item] for item in id2contig.values()])

    rec = []
    for record in SeqIO.parse(f'{outpth}/{infile}', 'fasta'):
        name = record.id
        name = name.rsplit('_', 1)[0]
        rec.append(name)
    counter = Counter(rec)
    total_num = np.array([counter[item] for item in id2contig.values()])
    proportion = mapped_num/total_num


    # Store the parameters
    pkl.dump(sentence,        open(f'{outpth}/{tool}_sentence.feat', 'wb'))
    pkl.dump(id2contig,       open(f'{outpth}/{tool}_sentence_id2contig.dict', 'wb'))
    pkl.dump(proportion,      open(f'{outpth}/{tool}_sentence_proportion.feat', 'wb'))
    pkl.dump(pc2wordsid,      open(f'{outpth}/{tool}_pc2wordsid.dict', 'wb'))




#############################################################
#################  Convert2BERT input  ######################
#############################################################

def generate_bert_input(db_dir, inpth, outpth, tool):
    feat = pkl.load(open(f'{inpth}/{tool}_sentence.feat', 'rb'))
    pcs = pkl.load(open(f'{inpth}/{tool}_pc2wordsid.dict', 'rb'))
    id2pcs = {item: key for key, item in pcs.items()}
    text = []
    label = []
    for line in feat:
        sentence = ""
        flag = 0
        for i in range(len(line)-2):
            if line[i]-1 == -1:
                flag = 1
                sentence = sentence[:-1]
                break
            sentence = sentence + id2pcs[line[i]-1] + ' '
        if flag == 0:
            sentence = sentence[:-1]
        text.append(sentence)
        label.append(1)

    feat_df = pd.DataFrame({'label':label, 'text':text})
    feat_df.to_csv(f'{outpth}/{tool}_bert_feat.csv', index=None)


