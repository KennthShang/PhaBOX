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
    logger.info("Running program: PhaGCN (taxonomy classification)")
    program_start = time.time()


    contigs   = inputs.contigs
    midfolder = inputs.midfolder
    out_dir   = 'final_prediction/'
    rootpth   = inputs.outpth
    db_dir    = inputs.dbdir
    aai       = inputs.aai
    pcov      = inputs.pcov
    share     = inputs.share
    threads   = inputs.threads


    if not os.path.isfile(contigs):
        exit()


    if not os.path.exists(db_dir):
        print(f'Database directory {db_dir} missing or unreadable')
        exit(1)

    supplementary = 'phagcn_supplementary'
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
        if not genomes:
            with open(f'{rootpth}/{out_dir}/phagcn_prediction.tsv', 'w') as file_out:
                file_out.write("Accession\tLength\tLineage\tPhaGCNScore\tGenus\tGenusCluster\n")
                for record in SeqIO.parse(contigs, 'fasta'):
                    file_out.write(f'{record.id}\t{len({record.seq})}\tfiltered\t0\t-\t-\n')
            logger.info(f"PhaGCN finished! please check the results in {os.path.join(rootpth,out_dir, 'phagcn_prediction.tsv')}")
            exit()
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
            with open(f'{rootpth}/{out_dir}/phagcn_prediction.tsv', 'w') as file_out:
                file_out.write("Accession\tLength\tLineage\tPhaGCNScore\tGenus\tGenusCluster\n")
                for record in SeqIO.parse(contigs, 'fasta'):
                    file_out.write(f'{record.id}\t{len({record.seq})}\tfiltered\t0\t-\t-\n')
            logger.info(f"PhaGCN finished! please check the results in {os.path.join(rootpth,out_dir, 'phagcn_prediction.tsv')}")
            exit()


        _ = SeqIO.write(rec, f'{rootpth}/filtered_contigs.fa', 'fasta')

    ###############################################################
    ##################### PhaGCN (clustering)  ####################
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
    
    # combine the database with the predicted proteins
    run_command(f"cat {db_dir}/RefVirus.faa {rootpth}/{midfolder}/query_protein.fa > {rootpth}/{midfolder}/ALLprotein.fa")
    # generate the diamond database
    run_command(f'diamond makedb --in {rootpth}/{midfolder}/query_protein.fa -d {rootpth}/{midfolder}/query_protein.dmnd --threads {threads} --quiet')
    # run diamond
    # align to the database
    if os.path.exists(f'{rootpth}/{midfolder}/db_results.tab'):
        logger.info("[3/8] using existing all-against-all alignment results...")
    else:  
        logger.info("[3/8] running all-against-all alignment...")
        run_command(f"diamond blastp --db {db_dir}/RefVirus.dmnd --query {rootpth}/{midfolder}/query_protein.fa --out {rootpth}/{midfolder}/db_results.tab --outfmt 6 --threads {threads} --evalue 1e-5 --max-target-seqs 10000 --query-cover 50 --subject-cover 50 --quiet")
        run_command(f"awk '{{print $1,$2,$3,$12}}' {rootpth}/{midfolder}/db_results.tab > {rootpth}/{midfolder}/db_results.abc")
    # align to itself
    run_command(f"diamond blastp --db {rootpth}/{midfolder}/query_protein.dmnd --query {rootpth}/{midfolder}/query_protein.fa --out {rootpth}/{midfolder}/self_results.tab --outfmt 6 --threads {threads} --evalue 1e-5 --max-target-seqs 10000 --query-cover 50 --subject-cover 50 --quiet")
    run_command(f"awk '{{print $1,$2,$3,$12}}' {rootpth}/{midfolder}/self_results.tab > {rootpth}/{midfolder}/self_results.abc")

    logger.info("[4/8] generating phagcn networks...")
    # FLAGS: no proteins aligned to the database
    if os.path.getsize(f'{rootpth}/{midfolder}/db_results.abc') == 0:
        Accession = []
        Length_list = []
        for record in SeqIO.parse(f'{contigs}', 'fasta'):
            Accession.append(record.id)
            Length_list.append(len(record.seq))
        df = pd.DataFrame({"Accession": Accession, "Length": Length_list,  "Lineage":['unknown']*len(Accession), "PhaGCNScore":[0]*len(Accession), "Genus": ['-']*len(Accession), "GenusCluster": ['-']*len(Accession)})
        df.to_csv(f"{rootpth}/{out_dir}/phagcn_prediction.tsv", index = None, sep='\t')
        exit()

    # add the genome size
    genome_size = defaultdict(int)
    for index, r in enumerate(SeqIO.parse(f'{rootpth}/{midfolder}/ALLprotein.fa', 'fasta')):
        genome_id = r.id.rsplit('_', 1)[0]
        genome_size[genome_id] += 1



    compute_aai(f'{rootpth}/{midfolder}', 'db_results', genome_size)
    compute_aai(f'{rootpth}/{midfolder}', 'self_results', genome_size)


    # filter the network
    df1 = pd.read_csv(f'{rootpth}/{midfolder}/db_results_aai.tsv', sep='\t')
    df2 = pd.read_csv(f'{rootpth}/{midfolder}/self_results_aai.tsv', sep='\t')
    df3 = pd.read_csv(f'{db_dir}/database_aai.tsv', sep='\t')
    df = pd.concat([df1, df2, df3])
    sub_df = df[((df['aai']>=aai)&((df['qcov']>=pcov)|(df['tcov']>=pcov)|(df['sgenes']>=share)))].copy()
    sub_df['score'] = sub_df['aai']/100.0 * sub_df[['qcov', 'tcov']].max(axis=1)/100.0

    # write the network
    sub_df.drop(['qcov', 'tcov', 'qgenes', 'tgenes', 'sgenes', 'aai'], axis=1, inplace=True)
    sub_df.to_csv(f'{rootpth}/{midfolder}/phagcn_network.tsv', sep='\t', index=False, header=False)
    #### drop network
    sub_df.rename(columns={'query':'Source', 'target':'Target', 'score':'Weight'}, inplace=True)
    sub_df.to_csv(f"{rootpth}/{out_dir}/{supplementary}/phagcn_network_edges.tsv", index=False, sep='\t')
    run_command(f"mcl {rootpth}/{midfolder}/phagcn_network.tsv -te {threads} -I 2.0 --abc -o {rootpth}/{midfolder}/phagcn_genus_clusters.txt > /dev/null 2>&1")
    

    ###############################################################
    ##################### PhaGCN (prediction)  ####################
    ###############################################################
    logger.info("[5/8] predicting the taxonomy...")
    contig2ORFs = {}
    for record in SeqIO.parse(f'{rootpth}/{midfolder}/query_protein.fa', "fasta"):
        contig = record.id.rsplit("_", 1)[0]
        try:
            contig2ORFs[contig].append(record.id)
        except:
            contig2ORFs[contig] = [record.id]

    contig_names = list([record.id for record in SeqIO.parse(f'{rootpth}/filtered_contigs.fa', 'fasta')])
    ORF2hits, all_hits = parse_alignment(f'{rootpth}/{midfolder}/db_results.abc')
    taxid2parent, taxid2rank = import_nodes(f'{db_dir}/nodes.csv')
    taxid2name = import_names(f'{db_dir}/names.csv')
    database_taxa_df = pd.read_csv(f'{db_dir}/taxid.csv', sep=",", header=None)
    database_taxa_df = database_taxa_df[database_taxa_df[0].isin(all_hits)]
    database2taxid = database_taxa_df.set_index(0)[1].to_dict()


    # List to store results for each contig
    results = []
    # Iterate over sorted contig names
    with tqdm(total=len(contig_names)) as pbar:
        for contig in sorted(contig_names):
            _ = pbar.update(1)
            # Check if contig has associated ORFs
            if contig not in contig2ORFs:
                # No ORFs found for this contig
                results.append([contig, "no ORFs found", ""])
                continue
            # Find LCA for each ORF in the contig
            LCAs_ORFs = [
                find_LCA_for_ORF(ORF2hits[ORF], database2taxid, taxid2parent)
                for ORF in contig2ORFs[contig]
                if ORF in ORF2hits
            ]
            # Check if any LCAs were found
            if not LCAs_ORFs:
                results.append([contig,"no hits to database", -1])
                continue
            # Find the weighted LCA for the contig
            lineages, lineages_scores = find_weighted_LCA(LCAs_ORFs, taxid2parent, 0.5)
            # Handle cases with no valid ORFs or lineages
            if lineages == "no ORFs with taxids found.":
                results.append([contig, "hits not found in taxonomy files", -1])
                continue
            if lineages == "no lineage larger than threshold.":
                results.append([contig, "no lineage larger than threshold.", -1])
                continue
            # Prepare lineage and score strings
            lineage_str, scores_str = convert_lineage_to_names(lineages, lineages_scores, taxid2name, taxid2rank)
            results.append([contig, lineage_str, scores_str])
            

    # Convert results to a DataFrame and save as CSV
    df = pd.DataFrame(results, columns=["Accession", "Lineage", "Score"])
    df['Length'] = df['Accession'].apply(lambda x: genomes[x].length if x in genomes else '-')


    Genus = [
        next((item.split(':')[1] for item in line.split(';') if item.startswith('genus:')), '-')
        if line not in {'no hits to database', 'no ORFs found', 'hits not found in taxonomy files', 'no lineage larger than threshold.'}
        else '-'
        for line in df['Lineage']
    ]

    df['Genus'] = Genus

    df.to_csv(f'{rootpth}/{midfolder}/phagcn_classification.tsv', index=False, sep='\t')


    ###############################################################
    #################### summarize results ########################
    ###############################################################

    logger.info("[6/8] summarizing the results...")
    #query_df = pd.read_csv(f'{rootpth}/{midfolder}/phagcn_classification.csv', sep='\t')
    query_df = df.copy()
    ref_df = pd.read_csv(f'{db_dir}/RefVirus.csv')
    refacc2genus = {acc: genus for acc, genus in zip(ref_df['Accession'], ref_df['Genus'])}
    acc2score = {acc: score for acc, score in zip(query_df['Accession'], query_df['Score'])}
    query_df = query_df[['Accession', 'Lineage', 'Length', 'Genus']]
    ref_df = ref_df[['Accession', 'Lineage', 'Length', 'Genus']]
    cluster_df = pd.concat([query_df, ref_df])


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

    cluster_df['cluster'] = cluster_df['Accession'].apply(lambda x: seq2cluster[x] if x in seq2cluster else -1)
    cluster_df = cluster_df.sort_values('cluster')
    # assign cluster id to the unassigned entries (continuous number with the existing cluster id)
    cluster_df.loc[cluster_df['cluster'] == -1, 'cluster'] = range(cluster_df['cluster'].max() + 1, cluster_df['cluster'].max() + 1 + len(cluster_df[cluster_df['cluster'] == -1]))
    cluster_df.reset_index(drop=True, inplace=True)

    logger.info("[7/8] generating genus level clusters...")
    # assign the Lineage according to the entry with longest length
    groups = cluster_df.groupby('cluster')
    for cluster, group in groups:
        acc_list = group['Accession']
        # check whether  NCBI accession (phabox) exist
        if any('phabox' in acc for acc in acc_list):
            continue
        idx = group['Length'].idxmax()
        Lineage = group.loc[idx, 'Lineage']
        cluster_df.loc[group.index, 'Lineage'] = Lineage
        Genus = group.loc[idx, 'Genus']
        cluster_df.loc[group.index, 'Genus'] = Genus

    # assign the Lineage according to the entry in database
    for cluster, group in groups:
        acc_list = group['Accession']
        # check whether  NCBI accession (phabox) exist
        if not any('phabox' in acc for acc in acc_list):
            continue
        ref_group = group[group['Accession'].apply(lambda x: 'phabox' in x)]
        query_group = group[group['Accession'].isin(query_df['Accession'])]
        if len(ref_group[ref_group['Genus'] != '-']['Genus'].unique()) == 1:
            Lineage = ref_group['Lineage'].values[0]
            cluster_df.loc[query_group.index, 'Lineage'] = Lineage
            Genus = ref_group['Genus'].values[0]
            cluster_df.loc[group.index, 'Genus'] = Genus


    df = cluster_df[cluster_df['Accession'].isin(genomes.keys())].copy()
    df = df.reset_index(drop=True)

    # issues that NaN will assign known_genus
    predicted = df[(df['Genus'] != '-') & (df['Genus'].notna())]
    unpredicted = df[(df['Genus'] == '-') | (df['Genus'].isna())]
    df.loc[predicted.index, 'cluster'] = 'known_genus'

    groups = unpredicted.groupby('cluster')
    cluster_idx = 0
    for cluster, group in groups:
        if group.shape[0] == 1:
            df.loc[group.index, 'cluster'] = 'singleton'
        else:
            df.loc[group.index, 'cluster'] = f'group_{cluster_idx}'
            cluster_idx += 1

    df = df.sort_values('Accession')
    df.rename(columns={'cluster':'GenusCluster', 'Lineage': 'Pred'}, inplace=True)
    df['Score'] = [acc2score[acc] for acc in df['Accession']]
    df.to_csv(f'{rootpth}/{midfolder}/phagcn_prediction.tsv', index=False, sep='\t')


    ###############################################################
    ####################### dump results ##########################
    ###############################################################
    logger.info("[8/8] writing the results...")
    df = df.reset_index(drop=True)

    contigs_list = {item:1 for item in list(df['Accession'])}
    filtered_contig = []
    filtered_lenth = []
    unpredicted_contig = []
    unpredicted_length = []
    for record in SeqIO.parse(f'{contigs}', 'fasta'):
        try:
            _ = contigs_list[record.id]
        except:
            if len(record.seq) < inputs.len:
                filtered_contig.append(record.id)
                filtered_lenth.append(len(record.seq))
            else:
                unpredicted_contig.append(record.id)
                unpredicted_length.append(len(record.seq))


    # Create lists by combining existing data with new entries
    all_contigs = df['Accession'].tolist() + filtered_contig + unpredicted_contig
    all_pred = df['Pred'].tolist() + ['filtered'] * len(filtered_contig) + ['-'] * len(unpredicted_contig)
    all_score = df['Score'].tolist() + [0] * len(filtered_contig) + [0] * len(unpredicted_contig)
    all_length = df['Length'].tolist() + filtered_lenth + unpredicted_length
    all_genus = df['Genus'].tolist() + ['-'] * len(filtered_contig) + ['-'] * len(unpredicted_contig)
    all_cluster = df['GenusCluster'].tolist() + ['-'] * len(filtered_contig) + ['-'] * len(unpredicted_contig)

    # Create DataFrame directly from lists
    contig_to_pred = pd.DataFrame({
        'Accession': all_contigs,
        'Length': all_length,
        'Lineage': all_pred,
        'PhaGCNScore': all_score,
        'Genus': all_genus,
        'GenusCluster': all_cluster
    })

    ProkaryoticGroup = pkl.load(open(f'{db_dir}/ProkaryoticGroup.pkl', 'rb'))
    rows_to_update = contig_to_pred['Lineage'].apply(lambda lineage: any(group in lineage for group in ProkaryoticGroup))
    # Update the pred_csv dataframe
    contig_to_pred.loc[rows_to_update, 'Prokaryotic virus (Bacteriophages and Archaeal virus)'] = 'Y'
    contig_to_pred.loc[~rows_to_update, 'Prokaryotic virus (Bacteriophages and Archaeal virus)'] = 'N'



    # Save DataFrame to CSV
    contig_to_pred.to_csv(f"{rootpth}/{out_dir}/phagcn_prediction.tsv", index=False, sep='\t')
    query2genus = {acc: genus for acc, genus in zip(df['Accession'], df['Genus'])}


    phagcn_node = pd.DataFrame({
        'Accession': list(refacc2genus.keys()) + list(query2genus.keys()), 
        'Genus': list(refacc2genus.values()) + list(query2genus.values()), 
        'TYPE': ['Ref']*len(refacc2genus) + ['Query']*len(query2genus)
        })
    
    phagcn_node.to_csv(f"{rootpth}/{out_dir}/{supplementary}/phagcn_network_nodes.tsv", index=False, sep='\t')

    if inputs.draw == 'Y':
        draw_network(f'{rootpth}/{out_dir}/{supplementary}/', f'{rootpth}/{out_dir}/{supplementary}/', 'phagcn')


    if inputs.task != 'end_to_end':
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

        phavip_dump_result(genomes, rootpth, out_dir, logger, supplementary = 'phagcn_supplementary')


    
    logger.info("Run time: %s seconds\n" % round(time.time() - program_start, 2))
