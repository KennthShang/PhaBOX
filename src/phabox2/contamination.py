#!/usr/bin/env python
import os
from .scripts.ulity import *
from .scripts.preprocessing import *
from .scripts.parallel_prodigal_gv import main as parallel_prodigal_gv
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import multiprocessing as mp
import time


def run(inputs):
    
    logger = get_logger()
    logger.info("Running program: Contamination Search")
    program_start = time.time()

    contigs   = inputs.contigs
    proteins  = inputs.proteins
    midfolder = inputs.midfolder
    rootpth   = inputs.outpth
    db_dir    = inputs.dbdir
    out_dir   = 'final_prediction/'
    threads   = inputs.threads
    sensitive   = inputs.sensitive


    check_path(os.path.join(rootpth, out_dir))
    check_path(os.path.join(rootpth, midfolder))
    check_path(os.path.join(rootpth, out_dir, 'contamination_supplementary'))



    # Reading database info...
    db_info_df = pd.read_csv(f'{db_dir}/contamination_marker.tsv')
    db_info_df = {domain: {'category': category, 'function': desc} for domain, category, desc in zip(db_info_df['domain'], db_info_df['category'], db_info_df['desc'])}

    # Reading genome info
    genomes = {}
    for record in SeqIO.parse(f'{contigs}', 'fasta'):
        genome = Genome()
        genome.id = record.id
        genome.length = len(record.seq)
        genome.genes = []
        genome.seq = str(record.seq)
        genome.viral_hits = {}
        genome.regions = None
        genome.proportion = 0
        genomes[genome.id] = genome

    logger.info("[1/5] calling genes with prodigal...")
    if not os.path.exists(f'{proteins}'):
        run_command(f'cp {contigs} {rootpth}/check_contig.fa')
        parallel_prodigal_gv(f'{rootpth}/check_contig.fa', f'{rootpth}/{midfolder}/check_protein.fa', threads)
        proteins = f'{rootpth}/{midfolder}/check_protein.fa'


    # Reading gene info
    genes = {}
    for record in SeqIO.parse(f'{proteins}', 'fasta'):
        gene = Gene()
        rec_info = record.description.split()
        gene.id = rec_info[0]
        gene.start = int(rec_info[2])
        gene.end = int(rec_info[4])
        gene.strand = int(rec_info[6])
        gene.genome_id = gene.id.rsplit("_", 1)[0]
        gene.label = 0
        gene.gc = float(rec_info[-1].split('gc_cont=')[-1])
        gene.marker_hit = None
        genes[gene.id] = gene
        genomes[gene.genome_id].genes.append(gene.id)




    logger.info("[2/5] Annotating genes...")
    contamination_db = f'{db_dir}/contamination.dmnd'
    if sensitive == 'N':
        run_command(f'diamond blastp -d {contamination_db} -q {proteins} -o {rootpth}/{midfolder}/contamination.tsv --sensitive -k 1 --quiet -f 6 qseqid sseqid evalue')
    if sensitive == 'Y':
        run_command(f'diamond blastp -d {contamination_db} -q {proteins} -o {rootpth}/{midfolder}/contamination.tsv --very-sensitive -k 1 --quiet -f 6 qseqid sseqid evalue')
    align_df = pd.read_csv(f'{rootpth}/{midfolder}/contamination.tsv', sep='\t', names=['query', 'target', 'evalue'])
    align_df['target'] = align_df['target'].apply(lambda x: x.split('-consensus')[0])
    annotate_genes(db_info_df, genomes, genes, align_df)


    
    logger.info("[3/5] Identifying host regions...")
    for genome in genomes.values():
        genome.regions = define_regions(genome, genes, min_host_genes=2, min_viral_genes=2, min_host_fract=0.30, gc_weight=0.02, delta_cutoff=1.2)



    logger.info("[4/5] Determining genome copy number...")
    with mp.Pool(threads) as pool:
        kmer_freq_list = pool.map(get_average_kmer_freq, genomes.values())

    low_rec = []
    medium_rec = []
    high_rec = []
    for kmer_freq, genome in zip(kmer_freq_list, genomes.values()):
        genome.kmer_freq = kmer_freq
        if genome.regions:
            genome.provirus = 'Yes'
            genome.host_length =  sum(r["length"] for r in genome.regions if r["type"] == "host")
            genome.viral_length = sum(r["length"] for r in genome.regions if r["type"] == "viral")
            genome.contamination = round(100.0 * int(genome.host_length) / genome.length, 2)
            genome.confident = 'Low quality'
        else:
            genome.provirus = 'No'
            genome.contamination = 0
            if genome.count_viral == 0 and genome.count_host > 0:
                genome.confident = 'Low quality;no viral marker found'
                seq = genome.seq
                record = SeqRecord(Seq(seq), id=genome.id, description="")
                low_rec.append(record)
            elif genome.count_viral > 0 and genome.count_host > genome.count_viral:
                genome.confident = 'Medium quality'
                seq = genome.seq
                record = SeqRecord(Seq(seq), id=genome.id, description="")
                medium_rec.append(record)
            elif genome.kmer_freq < 1.25:
                genome.confident = 'High quality'
                seq = genome.seq
                record = SeqRecord(Seq(seq), id=genome.id, description="")
                high_rec.append(record)
            else:
                genome.confident = 'Medium quality'
                seq = genome.seq
                record = SeqRecord(Seq(seq), id=genome.id, description="")
                medium_rec.append(record)

    SeqIO.write(low_rec, f"{rootpth}/{out_dir}/contamination_supplementary/low_quality_virus.fa", "fasta")
    SeqIO.write(medium_rec, f"{rootpth}/{out_dir}/contamination_supplementary/medium_quality_virus.fa", "fasta")
    SeqIO.write(high_rec, f"{rootpth}/{out_dir}/contamination_supplementary/high_quality_virus.fa", "fasta")




    # Prepare file headers
    contamination_header = "Accession\tLength\tTotal_genes\tViral_genes\tProkaryotic_genes\tKmer_freq\tContamination\tProvirus\tPure_viral\n"
    candidate_prophage_header = "Accession\tLength\tTotal_genes\tViral_genes\tProkaryotic_genes\tProvirus\tContamination\tProviral_length\tProkaryotic_length\tRegion_types\tRegion_lengths\tRegion_coords_bp\tRegion_coords_genes\tRegion_viral_genes\tRegion_prokaryotic_genes\n"
    gene_annotation_header = "Accession\tGene_num\tStart\tEnd\tStrand\tGC\tmarker_label\tmarker_name\tEvalue\tScore\n"

    records = []

    # Open all files
    logger.info("[5/5] writing the results...")
    with open(f'{rootpth}/{out_dir}/contamination_prediction.tsv', 'w') as f1, \
        open(f'{rootpth}/{out_dir}/contamination_supplementary/candidate_provirus.tsv', 'w') as f2, \
        open(f'{rootpth}/{out_dir}/contamination_supplementary/marker_gene_from_contamination_search.tsv', 'w') as f3:

        # Write headers
        f1.write(contamination_header)
        f2.write(candidate_prophage_header)
        f3.write(gene_annotation_header)
        
        # Iterate through genomes
        for genome in genomes.values():
            # Write to contamination.csv
            f1.write(
                f"{genome.id}\t{genome.length}\t{len(genome.genes)}\t"
                f"{genome.count_viral}\t{genome.count_host}\t{genome.kmer_freq}\t"
                f"{genome.contamination}\t{genome.provirus}\t{genome.confident}\n"
            )

            # Write to candidate_prophage.csv if Provirus is 'Yes'
            if genome.provirus == 'Yes':
                region_types = ";".join(region["type"] for region in genome.regions)
                region_lengths = ";".join(str(region["length"]) for region in genome.regions)
                region_coords_bp = ";".join(f"{region['start_pos']}-{region['end_pos']}" for region in genome.regions)
                region_coords_genes = ";".join(f"{region['start_gene'] + 1}-{region['end_gene']}" for region in genome.regions)
                region_viral_genes = ";".join(str(region["viral_genes"]) for region in genome.regions)
                region_host_genes = ";".join(str(region["host_genes"]) for region in genome.regions)

                f2.write(
                    f"{genome.id}\t{genome.length}\t{len(genome.genes)}\t"
                    f"{genome.count_viral}\t{genome.count_host}\tYes\t"
                    f"{genome.contamination}\t{genome.viral_length}\t{genome.host_length}\t"
                    f"{region_types}\t{region_lengths}\t{region_coords_bp}\t{region_coords_genes}\t"
                    f"{region_viral_genes}\t{region_host_genes}\n"
                )

            # Write to marker_gene_from_contamination_search.csv
            for gene_id in genome.genes:
                gene = genes[gene_id]
                marker_hit = gene.marker_hit if gene.marker_hit else {"target": "NA", "evalue": "NA", "function": "NA"}
                f3.write(
                    f"{genome.id}\t{gene_id.split('_')[-1]}\t{gene.start},{gene.end}\t"
                    f"{gene.strand}\t{round(gene.gc, 1)}\t{gene.label}\t"
                    f"{marker_hit['target']}\t{marker_hit['evalue']}\t{marker_hit['function']}\n"
                )

            # Prepare records for FASTA file
            viral_regions = [region for region in genome.regions if region["type"] == "viral"] or genome.regions
            for i, region in enumerate(viral_regions):
                header = f"{genome.id}_{i+1} {region['start_pos']}-{region['end_pos']}/{genome.length}"
                seq = genome.seq[region["start_pos"] - 1: region["end_pos"]]
                record = SeqRecord(Seq(seq), id=header, description="")
                records.append(record)

    # Write FASTA file
    SeqIO.write(records, f"{rootpth}/{out_dir}/contamination_supplementary/proviruses.fa", "fasta")
    logger.info("Run time: %s seconds" % round(time.time() - program_start, 2))