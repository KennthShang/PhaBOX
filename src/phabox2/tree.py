#!/usr/bin/env python
import os
import time
import pandas as pd

from .scripts.ulity import *
from .scripts.preprocessing import *
from .scripts.parallel_prodigal_gv import main as parallel_prodigal_gv
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord





def run(inputs):

    logger = get_logger()
    logger.info("Running program: marker gene searching")
    program_start = time.time()

    contigs   = inputs.contigs
    proteins  = inputs.proteins
    midfolder = inputs.midfolder
    rootpth   = inputs.outpth
    db_dir    = inputs.dbdir
    out_dir   = 'final_prediction/'
    threads   = inputs.threads
    cov       = inputs.mcov
    pident    = inputs.mpident

    marker    = inputs.marker
    msa       = inputs.msa
    tree      = inputs.tree
    candidate = ['terl', 'portal', 'head', 'endolysin', 'holin']
    for item in marker:
        if item not in candidate:
            logger.error(f"Invalid marker: {item}")
            logger.error(f"Please select from {candidate}")
            exit(1)

    check_path(os.path.join(rootpth, out_dir))
    check_path(os.path.join(rootpth, midfolder))
    check_path(os.path.join(rootpth, midfolder, 'marker'))
    check_path(os.path.join(rootpth, midfolder, 'msa'))
    check_path(os.path.join(rootpth, out_dir, 'tree_supplementary'))

    ###############################################################
    #######################  Filter length ########################
    ###############################################################
    logger.info("[1/5]filtering the length of contigs...")
    genomes = {}
    rec = []
    for record in SeqIO.parse(contigs, 'fasta'):
        if len(record.seq) >= inputs.len:
            rec.append(record)
            genome = Genome()
            genome.seq = record.seq
            genome.id = record.id
            genome.length = len(record.seq)
            genome.genes = []
            genome.viral_hits = {}
            genome.regions = None
            genome.proportion = 0
            genomes[genome.id] = genome

    if not rec:
        logger.info(f"No contigs longer than {inputs.len} bp!!!! Please check the input file")
        exit()

    _ = SeqIO.write(rec, f'{rootpth}/filtered_contigs.fa', 'fasta')

    ###############################################################
    #####################  Tree main program ######################
    ###############################################################

    if os.path.exists(f'{inputs.proteins}'):
        logger.info("[2/5] using provided protein file...")
        rec = []
        for record in SeqIO.parse(inputs.proteins, 'fasta'):
            genome_id = record.id.rsplit('_', 1)[0]
            try:
                _ = genomes[genome_id]
                rec.append(record)
            except:
                pass
        if rec:
            SeqIO.write(rec, f'{rootpth}/{midfolder}/query_protein.fa', 'fasta')
        else:
            logger.info("WARNING: no proteins found in the provided file.\nPlease check whether the genes is called by the prodigal.")
            logger.info("[2/5] calling genes with prodigal...")
            parallel_prodigal_gv(f'{rootpth}/filtered_contigs.fa', f'{rootpth}/{midfolder}/query_protein.fa', threads)
    elif os.path.exists(f'{rootpth}/{midfolder}/query_protein.fa'):
        logger.info("[2/5] reusing existing protein file...")
    else:
        logger.info("[2/5] calling genes with prodigal...")
        parallel_prodigal_gv(f'{rootpth}/filtered_contigs.fa', f'{rootpth}/{midfolder}/query_protein.fa', threads)


    # run marker alignment
    logger.info(f"[3/5] running marker:")
    for item in marker:
        logger.info(f"Current marker: {item}...")
        _ = os.system(f'diamond makedb --in {db_dir}/marker/{item}.fa -d {rootpth}/{midfolder}/marker/{item}.dmnd --threads {threads} --quiet')
        _ = os.system(f'diamond blastp --threads {threads} --query {rootpth}/{midfolder}/query_protein.fa --db {rootpth}/{midfolder}/marker/{item}.dmnd --out {rootpth}/{midfolder}/{item}.blast -k 1 --outfmt 6 qseqid sseqid qlen slen length pident evalue --quiet')
        df = pd.read_csv(f'{rootpth}/{midfolder}/{item}.blast', sep='\t', names=['qseqid', 'sseqid', 'qlen', 'slen', 'length', 'pident', 'evalue'])
        df['query'] = df['qseqid'].apply(lambda x: x.rsplit('_', 1)[0])
        df['qcovhsp'] = df['length'] / df['qlen'] * 100
        df['scovhsp'] = df['length'] / df['slen'] * 100
        df['cov'] = df[['qcovhsp', 'scovhsp']].max(axis=1)
        df = df.sort_values(['query', 'cov', 'pident'], ascending=False)
        df = df.drop_duplicates(['query'], keep='first')
        df = df[df['cov'] > cov]
        df = df[df['pident'] > pident]
        df.to_csv(f'{rootpth}/{midfolder}/marker/{item}_filtered.csv', index=False)
        rec = [SeqRecord(genomes[genome].seq, id=genomes[genome].id, description=f'{item}') for genome in df['query']]
        if not rec:
            logger.info(f"No sequences passed the filter for marker {item}. Please check the file in {rootpth}/{midfolder}/marker/{item}_filtered.csv")
            logger.info(f"You probably shouldn't choose the marker {item} or use a more flexible --mcov and --mpident.")
            exit(1)
        _ = SeqIO.write(rec, f'{rootpth}/{midfolder}/marker/{item}.fa', 'fasta')
        _ = os.system(f'cat {rootpth}/{midfolder}/marker/{item}.fa {db_dir}/marker/{item}.fa > {rootpth}/{midfolder}/marker/{item}_conbined_db.fa')
        _ = os.system(f'cp {rootpth}/{midfolder}/marker/{item}_conbined_db.fa {rootpth}/{out_dir}/tree_supplementary/finded_marker_{item}_conbined_db.fa')

    logger.info("[4/5] Running the msa...")
    if msa == 'Y':
        for item in marker:
            logger.info(f"Running msa: {item}...This may take a long time...")
            _ = os.system(f'mafft --auto --quiet --thread {threads} {rootpth}/{midfolder}/marker/{item}_conbined_db.fa > {rootpth}/{midfolder}/msa/{item}.aln')
        msa = {}
        for item in marker:
            msa[item] = {}
            for record in SeqIO.parse(f'{rootpth}/{midfolder}/msa/{item}.aln', 'fasta'):
                msa[item][record.id] = record.seq
    
        # get the common ids
        for item in marker:
            common = set(msa[item].keys())
        common = set.intersection(*common)

        if all('phabox' in item for item in common):
            logger.info(f"No common sequences found for joint msa. Please check the files in the {rootpth}/{midfolder}/msa/")
            logger.info(f"You may need to adjust the number of selected marker genes.")
            exit(1)

        # combine the sequences in common ids
        record = []
        des = ''.join(marker)
        for key in common:
            seq = ''
            for item in marker:
                seq += msa[item][key]
            record.append(SeqRecord(seq, id=key, description=des))
        
        SeqIO.write(record, f'{rootpth}/{out_dir}/combined_marker.msa', 'fasta')

        if tree == 'Y':
            logger.info("[5/5] building tree...")
            logger.info("This may take a long time...")
            _ = os.system(f'export OMP_NUM_THREADS={threads};fasttree -quiet {rootpth}/{out_dir}/combined_marker.msa > {rootpth}/{out_dir}/combined.tree')
        else:
            logger.info("Skip the tree building step...")
    else:
        logger.info("Skip the msa and tree building step...")