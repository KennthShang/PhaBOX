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
from xml.etree import ElementTree as ElementTree


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
        diamond_cmd = f'diamond blastp --outfmt 5 --threads {threads} --sensitive -d {diamond_db} -q {outpth}/{infile} -o {outpth}/{tool}_results.xml -k 5'
        print("Running Diamond...")
        _ = subprocess.check_call(diamond_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        content = open(f'{outpth}/{tool}_results.xml', 'r').read()
        content = content.replace('&', '')
        with open(f'{outpth}/{tool}_results.xml', 'w') as file:
            file.write(content)
    except:
        print(diamond_cmd)
        print("diamond blastp failed")
        exit(1)

def convert_xml(outpth, tool, scripts='scripts/'):
    try:
        # running alignment
        try:
            diamond_cmd = f'{scripts}/blastxml_to_tabular.py -o {outpth}/{tool}_results.tab -c qseqid,sseqid,pident,length,mismatch,gapopen,qstart,qend,sstart,send,evalue {outpth}/{tool}_results.xml'
        except:
            diamond_cmd = f'blastxml_to_tabular.py -o {outpth}/{tool}_results.tab -c qseqid,sseqid,pident,length,mismatch,gapopen,qstart,qend,sstart,send,evalue {outpth}/{tool}_results.xml'
        _ = subprocess.check_call(diamond_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        diamond_out_fp = f"{outpth}/{tool}_results.tab"
        database_abc_fp = f"{outpth}/{tool}_results.abc"
        _ = subprocess.check_call("awk '$1!=$2 {{print $1,$2,$11}}' {0} > {1}".format(diamond_out_fp, database_abc_fp), shell=True)
    except:
        print(diamond_cmd)
        print("convert xml failed")
        exit(1)


def parse_evalue(blast_df, outpth, tool):
    protein2evalue = {}
    for protein, evalue in zip(blast_df['query'].values, blast_df['evalue'].values):
        try:
            protein2evalue[protein]
        except:
            protein2evalue[protein] = evalue        

    return protein2evalue

def parse_coverage(blast_df):
    with open(f'{blast_df}') as file_out:
        check_name_single = {}
        for line in file_out.readlines():
            parse = line.replace("\n", "").split("\t")
            virus = parse[0]
            qstart = float(parse[-5])
            tqend = float(parse[-4])
            sstart   = float(parse[-3]) 
            ssend = float(parse[-2])     
            tmp_score = np.abs((qstart-tqend)/(sstart-ssend))
            if tmp_score < 0.7:
                continue
            if virus in check_name_single:
                continue
            check_name_single[virus] = tmp_score
            
    return check_name_single

def parse_xml(protein2id, outpth, tool):
    xml_files = ['']*len(protein2id)
    flag = 0 # 0 for common, 1 for specific item
    with open(f'{outpth}/{tool}_results.xml', 'r') as file:
        content = file.readlines()
        for i in range(len(content)):
            line = content[i]
            if '<Iteration>\n' == line:
                item = content[i+3]
                qseqid = item.split('>', 1)[1]
                qseqid = qseqid.split(' ', 1)[0]
                try:
                    idx = protein2id[qseqid]
                    xml_files[idx]+=line
                    flag = 1
                except:
                    flag = 2
            elif line == '</Iteration>\n':
                if flag == 1:
                    xml_files[idx]+=line
                flag = 0
            elif flag == 0:
                for j in range(len(xml_files)):
                    xml_files[j]+=line
            elif flag == 1:
                xml_files[idx]+=line
            elif flag == 2:
                continue
    return xml_files


def parse_xml2(protein2id, outpth, tool):
    #protein2id = {'YP_009984512.1': 0, 'YP_009984889.1':1}
    xml_files = ['<?xml version="1.0"?>\n<!DOCTYPE BlastOutput PUBLIC "">\n']*len(protein2id)
    flag = 0 # 0 for common, 1 for specific item
    start = 0 # start position to writein
    context = ElementTree.iterparse(f'{outpth}/{tool}_results.xml', events=("start", "end"))
    for event, elem in context:
        if elem.tag == 'Iteration' and event == 'start':
            try:
                qseqid = elem.findtext("Iteration_query-def").split(" ", 1)[0]
                idx = protein2id[qseqid]
                xml_files[idx]+=f'<{elem.tag}>{elem.text}'
                flag = 1
            except:
                flag = 2
        elif elem.tag == 'Iteration' and event == 'end':
            if flag == 1:
                xml_files[idx]+=f'</{elem.tag}>\n'
            elif flag == 2:
                pass
            flag = 0
        elif flag == 0 and event =='start':
            for i in range(len(xml_files)):
                xml_files[i]+=f'<{elem.tag}>{elem.text}'
        elif flag == 0 and event =='end':
            for i in range(len(xml_files)):
                xml_files[i]+=f'</{elem.tag}>\n'
        elif flag == 1 and event =='start':
            xml_files[idx]+=f'<{elem.tag}>{elem.text}'
        elif flag == 1 and event =='end':
            xml_files[idx]+=f'</{elem.tag}>\n'
        elif flag == 2:
            continue
    return xml_files

def parse_position(outpth):
    protein2start = {}
    protein2end   = {}
    for record in SeqIO.parse(f'{outpth}/test_protein.fa', 'fasta'):
        description = str(record.description)
        description = description.split(' # ')
        start = description[1]
        end   = description[2]
        protein2start[record.id] = f'{start}'
        protein2end[record.id] =f'{end}'
    return protein2start, protein2end




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
    old_query = ''
    for query, ref, evalue in zip(blast_df['query'].values, blast_df['ref'].values, blast_df['evalue'].values):
        if old_query == query:
            continue
        old_query = query
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
    for key in set(blast_df['query'].values):
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


