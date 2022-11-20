import numpy as np
import pandas as pd
import os
import Bio
from Bio import SeqIO
import pandas as pd
import subprocess
import argparse
import re


parser = argparse.ArgumentParser(description="""Initial script to generate database for PhaSUIT.""")
parser.add_argument('--threads', help='number of threads to use', type=int, default=8)
parser.add_argument('--dbdir', help='initial directory',  default = 'initial_files/')
parser.add_argument('--out', help='database directory',  default = 'database/')
inputs = parser.parse_args()


threads = inputs.threads
out_fn = inputs.out
db_dir = inputs.dbdir

if not os.path.isdir(out_fn):
    os.makedirs(out_fn)

if not os.path.isdir(db_dir):
    os.makedirs(db_dir)







###############################################################
#####################  Initial database #######################
###############################################################

# PhaMer & PhaTYP
diamond_db = f'{db_dir}/phamer_database.dmnd'
try:
    if os.path.exists(diamond_db):
        print(f'Using preformatted DIAMOND database ({diamond_db}) ...')
    else:
        # create database
        make_diamond_cmd = f'diamond makedb --threads {threads} --in {db_dir}/phamer_database.fa -d {out_fn}/phamer_database.dmnd'
        print("Creating Diamond database...")
        _ = subprocess.check_call(make_diamond_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        diamond_db = f'{out_fn}/database.dmnd'
except:
    print("PhaMer makedb failed")
    exit(1)


# PhaGCN
diamond_db = f'{out_fn}/phagcn_database.self-diamond.tab.abc'
try:
    if os.path.exists(diamond_db):
        print(f'Using preformatted PhaGCN database ({diamond_db}) ...')
    else:
        make_diamond_cmd = f'diamond makedb --threads {threads}  --in {db_dir}/Caudovirales_protein.fasta -d {out_fn}/phagcn_database.dmnd'
        print("Creating Diamond database...")
        _ = subprocess.check_call(make_diamond_cmd, shell=True)
        
        diamond_cmd = f'diamond blastp --threads {threads}  --sensitive -d {out_fn}/phagcn_database.dmnd -q {db_dir}/Caudovirales_protein.fasta -o {out_fn}/phagcn_database.self-diamond.tab'
        print("Running Diamond...")
        _ = subprocess.check_call(diamond_cmd, shell=True)
        diamond_out_fp = f"{out_fn}/phagcn_database.self-diamond.tab"
        database_abc_fp = f"{out_fn}/phagcn_database.self-diamond.tab.abc"
        _ = subprocess.check_call("awk '$1!=$2 {{print $1,$2,$11}}' {0} > {1}".format(diamond_out_fp, database_abc_fp), shell=True)
except:
    print("PhaGCN database failed")
    exit(1)


# Cherry
diamond_db = f'{out_fn}/cherry_database.self-diamond.tab.abc'
try:
    if os.path.exists(diamond_db):
        print(f'Using preformatted Cherry database ({diamond_db}) ...')
    else:
        make_diamond_cmd = f'diamond makedb --threads {threads} --in {db_dir}/cherry_protein.fasta -d {out_fn}/cherry_database.dmnd'
        print("Creating Diamond database...")
        _ = subprocess.check_call(make_diamond_cmd, shell=True)
        diamond_cmd = f'diamond blastp --threads {threads}  --sensitive -d {out_fn}/cherry_database.dmnd -q {db_dir}/cherry_protein.fasta -o {out_fn}/cherry_database.self-diamond.tab'
        print("Running Diamond...")
        _ = subprocess.check_call(diamond_cmd, shell=True)
        diamond_out_fp = f"{out_fn}/cherry_database.self-diamond.tab"
        database_abc_fp = f"{out_fn}/cherry_database.self-diamond.tab.abc"
        _ = subprocess.check_call("awk '$1!=$2 {{print $1,$2,$11}}' {0} > {1}".format(diamond_out_fp, database_abc_fp), shell=True)
except:
    print("create database failed")
    exit(1)



# database only 
genome_list = os.listdir(f'{out_fn}/prokaryote')
for genome in genome_list:
    accession = genome.split(".")[0]
    make_blast_cmd = f'makeblastdb -in {out_fn}/prokaryote/{genome} -dbtype nucl -parse_seqids -out {out_fn}/blast_db/{accession}'
    print("Creating blast database...")
    _ = subprocess.check_call(make_blast_cmd, shell=True)
    blast_cmd = f'blastn -query {out_fn}/cherry/nucl.fasta -db {out_fn}/blast_db/{accession} -outfmt 6 -out {out_fn}/blast_tab/{accession}.tab -num_threads {treads}'
    print("Running blastn...")
    _ = subprocess.check_call(blast_cmd, shell=True)



