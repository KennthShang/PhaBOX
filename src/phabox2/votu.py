#!/usr/bin/env python
import os
import time
import pandas as pd
import numpy as np
import pickle as pkl

from .scripts.ulity import *
from .scripts.preprocessing import *
from .scripts.parallel_prodigal_gv import main as parallel_prodigal_gv
from Bio import SeqIO
from collections import defaultdict




def run(inputs):
    
    logger = get_logger()
    logger.info("Running program: vOTU groupping")
    program_start = time.time()

    contigs   = inputs.contigs
    proteins  = inputs.proteins
    midfolder = inputs.midfolder
    rootpth   = inputs.outpth
    db_dir    = inputs.dbdir
    out_dir   = 'final_prediction/'
    threads   = inputs.threads

    aai       = inputs.aai
    pcov      = inputs.pcov
    share     = inputs.share
    mode      = inputs.mode

    

    check_path(os.path.join(rootpth, out_dir))
    check_path(os.path.join(rootpth, midfolder))
    check_path(os.path.join(rootpth, out_dir, 'votu_supplementary'))

    ###############################################################
    #######################  Filter length ########################
    ###############################################################
    logger.info("filtering the length of contigs...")
    genomes = {}
    rec = []
    for record in SeqIO.parse(contigs, 'fasta'):
        if len(record.seq) > inputs.len:
            rec.append(record)
            genome = Genome()
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
    #####################  vOTU main program ######################
    ###############################################################

    if mode.upper() == 'ANI':
        logger.info("[1/5] calling genes with prodigal...")
        _ = os.system(f"makeblastdb -in {rootpth}/filtered_contigs.fa -dbtype nucl -parse_seqids -out {rootpth}/{midfolder}/selfdb > /dev/null 2>&1")
        _ = os.system(f"blastn -query {rootpth}/filtered_contigs.fa -db {rootpth}/{midfolder}/selfdb -out {rootpth}/{midfolder}/ani_blast.tsv -outfmt '6 std qlen slen' -max_target_seqs 25000 -perc_identity 90  -num_threads 40 -evalue 1e-3")
        df = parse_blast(f'{rootpth}/{midfolder}/ani_blast.tsv')
        if df.empty:
            data = []
            for genome in genomes:
                data.append((genomes[genome].id, 'singleton', genomes[genome].id, genomes[genome].length))
            df = pd.DataFrame(data, columns=['Sequence', 'vOTU', 'Representative', 'Length'])
            df.to_csv(f'{rootpth}/{out_dir}/ANI_based_vOTU.tsv', index=False, sep='\t')
            return

        with open(f'{rootpth}/{midfolder}/vOTU_ani.tsv', 'w') as out:
            out.write('qname\ttname\tnum_alns\tpid\tqcov\ttcov\n')
            grouped = df.groupby(['qname', 'tname'])
            for (qname, tname), alns in grouped:
                ani = compute_ani(alns)
                qcov = compute_cov(alns, 'qcoords', 'qlen')
                tcov = compute_cov(alns, 'tcoords', 'tlen')
                row = [qname, tname, len(alns), ani, qcov, tcov]
                out.write('\t'.join(map(str, row)) + '\n')

        # Read sequences
        logger.info("[2/5] reading sequences...")
        seqs = {genomes[genome].id: genomes[genome].length for genome in genomes}
        seqs = [id for id, _ in sorted(seqs.items(), key=lambda x: x[1], reverse=True)]


        # Store edges
        logger.info("[3/5] storing edges...")
        edges = {x: [] for x in seqs}
        with open(f'{rootpth}/{midfolder}/vOTU_ani.tsv') as handle:
            for line in handle:
                qname, tname, _, ani, _, tcov = line.split()
                if qname == tname or qname not in edges or tname not in edges:
                    continue
                if float(tcov) < inputs.tcov or float(ani) < inputs.ani:
                    continue
                edges[qname].append(tname)

        # Clustering
        logger.info("[4/5] clustering...")
        clust_to_seqs = {}
        seq_to_clust = {}
        for seq_id in seqs:
            if seq_id in seq_to_clust:
                continue
            clust_to_seqs[seq_id] = [seq_id]
            seq_to_clust[seq_id] = seq_id
            for mem_id in edges[seq_id]:
                if mem_id not in seq_to_clust:
                    clust_to_seqs[seq_id].append(mem_id)
                    seq_to_clust[mem_id] = seq_id


        # Write output
        logger.info("[5/5] writing clusters...")
        data = []
        for idx, (rep_seq, members) in enumerate(clust_to_seqs.items()):
            if len(members) > 1:
                cluster_id = f'group_{idx}'
            else:
                cluster_id = 'singleton'
            
            for member in members:
                data.append((member, cluster_id, rep_seq, genomes[member].length))

        df = pd.DataFrame(data, columns=['Sequence', 'vOTU', 'Representative', 'Length'])
        df.to_csv(f'{rootpth}/{out_dir}/ANI_based_vOTU.tsv', index=False, sep='\t')
        return
                
    elif mode.upper() == 'AAI':
        if os.path.exists(f'{inputs.proteins}'):
            logger.info("[2/8] using provided protein file...")
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
                logger.info("[2/8] calling genes with prodigal...")
                parallel_prodigal_gv(f'{rootpth}/filtered_contigs.fa', f'{rootpth}/{midfolder}/query_protein.fa', threads)
        elif os.path.exists(f'{rootpth}/{midfolder}/query_protein.fa'):
            logger.info("[2/8] reusing existing protein file...")
        else:
            logger.info("[2/8] calling genes with prodigal...")
            parallel_prodigal_gv(f'{rootpth}/filtered_contigs.fa', f'{rootpth}/{midfolder}/query_protein.fa', threads)

        logger.info("[2/5] running diamond blastp...")
        _ = os.system(f"diamond makedb --in {rootpth}/{midfolder}/query_protein.fa -d {rootpth}/{midfolder}/query_protein.dmnd --quiet")
        _ = os.system(f"diamond blastp --db {rootpth}/{midfolder}/query_protein.dmnd --query {rootpth}/{midfolder}/query_protein.fa --out {rootpth}/{midfolder}/self_results.tab --outfmt 6 --threads {threads} --evalue 1e-5 --max-target-seqs 10000 --query-cover 50 --subject-cover 50 --quiet")
        _ = os.system(f"awk '{{print $1,$2,$3,$12}}' {rootpth}/{midfolder}/self_results.tab > {rootpth}/{midfolder}/self_results.abc")

        genome_size = defaultdict(int)
        for index, r in enumerate(SeqIO.parse(f'{rootpth}/{midfolder}/query_protein.fa', 'fasta')):
            genome_id = r.id.rsplit('_', 1)[0]
            genome_size[genome_id] += 1
        
        logger.info("[3/5] computing AAI...")
        compute_aai(f'{rootpth}/{midfolder}', 'self_results', genome_size)
        df = pd.read_csv(f'{rootpth}/{midfolder}/self_results_aai.tsv', sep='\t')
        sub_df = df[((df['aai']>=aai)&((df['qcov']>=pcov)|(df['tcov']>=pcov)|(df['sgenes']>=share)))].copy()
        sub_df['score'] = sub_df['aai']/100.0 * sub_df[['qcov', 'tcov']].max(axis=1)/100.0

        sub_df.drop(['qcov', 'tcov', 'qgenes', 'tgenes', 'sgenes', 'aai'], axis=1, inplace=True)
        sub_df.to_csv(f'{rootpth}/{midfolder}/phagcn_network.tsv', sep='\t', index=False, header=False)
        #### drop network
        sub_df.rename(columns={'query':'Source', 'target':'Target', 'score':'Weight'}, inplace=True)
        _ = os.system(f'mcl {rootpth}/{midfolder}/phagcn_network.tsv -te {threads} -I 2.0 --abc -o {rootpth}/{midfolder}/phagcn_genus_clusters.txt > /dev/null 2>&1')


        logger.info("[4/5] generating vOTU...")
        seq2cluster = {}
        cluster2seq = {}
        for index, line in enumerate(open(f'{rootpth}/{midfolder}/phagcn_genus_clusters.txt')):
            aln = line.split()
            for seqs in aln:
                seq2cluster[seqs] = index
                try:
                    cluster2seq[index].append(seqs)
                except:
                    cluster2seq[index] = [seqs]

        logger.info("[5/5] writing vOTU prediction...")
        cluster_df = pd.DataFrame({"Accession": [genomes[item].id for item in genomes], "Length": [genomes[item].length for item in genomes]})
        cluster_df['cluster'] = cluster_df['Accession'].apply(lambda x: seq2cluster[x] if x in seq2cluster else -1)
        cluster_df = cluster_df.sort_values('cluster')
        # assign cluster id to the unassigned entries (continuous number with the existing cluster id)
        cluster_df.loc[cluster_df['cluster'] == -1, 'cluster'] = range(cluster_df['cluster'].max() + 1, cluster_df['cluster'].max() + 1 + len(cluster_df[cluster_df['cluster'] == -1]))
        cluster_df.reset_index(drop=True, inplace=True)

        groups = cluster_df.groupby('cluster')
        cluster_idx = 0
        for cluster, group in groups:
            if group.shape[0] == 1:
                cluster_df.loc[group.index, 'cluster'] = 'singleton'
                cluster_df.loc[group.index, 'Representative'] = group['Accession'].values[0]
            else:
                cluster_df.loc[group.index, 'cluster'] = f'group_{cluster_idx}'
                # assign the longest sequence as the representative
                rep_seq = group.loc[group['Length'].idxmax(), 'Accession']
                cluster_df.loc[group.index, 'Representative'] = rep_seq
                cluster_idx += 1

        cluster_df.rename(columns={'cluster':'vOTU'}, inplace=True)
        cluster_df.to_csv(f'{rootpth}/{out_dir}/AAI_based_vOTU.tsv', index=False, sep='\t')

    else:
        logger.error("Please specify the mode for clustering: ANI or AAI")
        exit()

    logger.info("Run time: %s seconds" % round(time.time() - program_start, 2))