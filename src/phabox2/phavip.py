#!/usr/bin/env python
import os
import pandas as pd
from .scripts.ulity import *
from .scripts.preprocessing import *
from .scripts.parallel_prodigal_gv import main as parallel_prodigal_gv
from Bio import SeqIO
from collections import defaultdict
import time
from tqdm import tqdm
from collections import Counter



def run(inputs):
    logger = get_logger()
    logger.info("Running program: PhaVIP (protein annotation)")
    program_start = time.time()


    contigs   = inputs.contigs
    midfolder = inputs.midfolder
    out_dir   = 'final_prediction/'
    rootpth   = inputs.outpth
    db_dir    = inputs.dbdir
    threads   = inputs.threads


    if not os.path.isfile(contigs):
        exit()


    if not os.path.exists(db_dir):
        print(f'Database directory {db_dir} missing or unreadable')
        exit(1)

    supplementary = 'phavip_supplementary'
    check_path(os.path.join(rootpth, out_dir))
    check_path(os.path.join(rootpth, out_dir, supplementary))
    check_path(os.path.join(rootpth, midfolder))


    ###############################################################
    #######################  Filter length ########################
    ###############################################################
    genomes = {}
    if os.path.exists(f'{rootpth}/filtered_contigs.fa'):
        logger.info("[1/8] reusing existing filtered contigs...")
        for record in SeqIO.parse(f'{rootpth}/filtered_contigs.fa', 'fasta'):
            genome = Genome()
            genome.id = record.id
            genome.length = len(record.seq)
            genome.genes = []
            genome.viral_hits = {}
            genome.regions = None
            genome.proportion = 0
            genomes[genome.id] = genome
    else:
        logger.info("[1/8] filtering the length of contigs...")
        rec = []
        for record in SeqIO.parse(contigs, 'fasta'):
            if len(record.seq) >= inputs.len:
                rec.append(record)
                genome = Genome()
                genome.id = record.id
                genome.length = len(record.seq)
                genome.genes = []
                genome.viral_hits = {}
                genome.regions = None
                genomes[genome.id] = genome
        # FLAGS: no contigs passed the length filter
        if not rec:
            logger.info(f"No contigs passed the length filter (length > {inputs.len})")
            logger.info(f"Please check the contigs file.")
            exit()


        _ = SeqIO.write(rec, f'{rootpth}/filtered_contigs.fa', 'fasta')

    ###############################################################
    ##################### PhaVIP (annotation)  ####################
    ###############################################################

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

    
    logger.info("[3/8] running all-against-all alignment...")
    run_command(f"diamond blastp --db {db_dir}/RefVirus.dmnd --query {rootpth}/{midfolder}/query_protein.fa --out {rootpth}/{midfolder}/db_results.tab --outfmt 6 --threads {threads} --evalue 1e-5 --max-target-seqs 10000 --query-cover 50 --subject-cover 50 --quiet")
    run_command(f"awk '{{print $1,$2,$3,$12}}' {rootpth}/{midfolder}/db_results.tab > {rootpth}/{midfolder}/db_results.abc")



    run_command(f"cp {rootpth}/filtered_contigs.fa {rootpth}/{out_dir}/{supplementary}/all_predicted_contigs.fa")
    run_command(f"cp {rootpth}/{midfolder}/query_protein.fa {rootpth}/{out_dir}/{supplementary}/all_predicted_protein.fa")
    run_command(f"cp {rootpth}/{midfolder}/db_results.tab {rootpth}/{out_dir}/{supplementary}/alignment_results.tab")
    run_command(f'sed -i "1i\qseqid\tsseqid\tpident\tlength\tmismatch\tgapopen\tqstart\tqend\tsstart\tsend\tevalue\tbitscore" {rootpth}/{out_dir}/{supplementary}/alignment_results.tab')

    genes = load_gene_info(f'{rootpth}/{midfolder}/query_protein.fa', genomes)

    anno_df = pkl.load(open(f'{db_dir}/RefVirus_anno.pkl', 'rb'))
    try:
        df = pd.read_csv(f'{db_dir}/db2ref_protein_map.csv', names=['dbid', 'refid'])
        db2ref = {key: value for key, value in zip(df['dbid'], df['refid'])}
    except:
        logger.info("WARNING: db2ref_protein_map.csv not found in the database directory. Please upgrade your database to version 2.1.")
        db2ref = {key: value for key, value in zip(anno_df.keys(), anno_df.values())}
    # protein annotation
    df = pd.read_csv(f'{rootpth}/{midfolder}/db_results.tab', sep='\t', names=['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'])
    df = df.drop_duplicates('qseqid', keep='first').copy()
    df['coverage'] = df['length'] / df['qend']
    df.loc[df['coverage'] > 1, 'coverage'] = 1
    for idx, row in df.iterrows():
        gene = genes[row['qseqid']]
        try:
            gene.anno = anno_df[row['sseqid']]
        except:
            gene.anno = 'hypothetical protein'
        gene.pident = row['pident']
        gene.coverage = row['coverage']
        gene.inference = db2ref.get(row['sseqid'], 'unknown')
        

    # write the gene annotation by genomes
    with open(f'{rootpth}/{out_dir}/{supplementary}/gene_annotation.tsv', 'w') as f:
        f.write('Genome\tORF\tStart\tEnd\tStrand\tGC\tAnnotation\tpident\tcoverage\tcloest_gene\n')
        for genome in genomes:
            for gene in genomes[genome].genes:
                f.write(f'{genome}\t{gene}\t{genes[gene].start}\t{genes[gene].end}\t{genes[gene].strand}\t{genes[gene].gc}\t{genes[gene].anno}\t{genes[gene].pident:.2f}\t{genes[gene].coverage:.2f}\t{genes[gene].inference}\n')

    phavip_dump_result(genomes, rootpth, out_dir, logger, supplementary = 'phavip_supplementary')


    logger.info("Run time: %s seconds\n" % round(time.time() - program_start, 2))