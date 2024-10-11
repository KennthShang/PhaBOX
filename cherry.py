#!/usr/bin/env python
import os
import time
import pandas as pd
import pickle as pkl


from scripts.ulity import *
from scripts.preprocessing import *
from scripts.parallel_prodigal_gv import main as parallel_prodigal_gv
from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastnCommandline
from collections import defaultdict
from tqdm import tqdm

def run(inputs):
    logger = get_logger()
    logger.info("Running program: CHERRY (Host prediction)")
    program_start = time.time()

    contigs   = inputs.contigs
    midfolder = inputs.midfolder
    out_dir   = 'final_prediction/'
    rootpth   = inputs.outpth
    db_dir    = inputs.dbdir
    threads   = inputs.threads
    length    = inputs.len
    pident     = inputs.pident
    cov        = inputs.cov/100
    aai        = inputs.aai
    pcov       = inputs.pcov
    share      = inputs.share
    blast      = inputs.blast



    if not os.path.isfile(contigs):
        print('cannot find the file')
        exit(1)

    if not os.path.exists(db_dir):
        print(f'Database directory {db_dir} missing or unreadable')
        exit(1)

    check_path(os.path.join(rootpth, out_dir))
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
            genomes[genome.id] = genome
    else:
        logger.info("[1/8] filtering the length of contigs...")
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
                genomes[genome.id] = genome
        # FLAGS: no contigs passed the length filter
        if not rec:
            with open(f'{rootpth}/{out_dir}/cherry_prediction.tsv', 'w') as file_out:
                file_out.write("Accession\tLength\tHost\tCHERRYScore\tMethod\n")
                for record in SeqIO.parse(contigs, 'fasta'):
                    file_out.write(f'{record.id},{len({record.seq})},filtered,0,-\n')
            logger.info(f"Cherry finished! please check the results in {os.path.join(rootpth,out_dir, 'cherry_prediction.tsv')}")
            exit()

        SeqIO.write(rec, f'{rootpth}/filtered_contigs.fa', 'fasta')

    ###############################################################
    ########################## CRISPRs  ###########################
    ###############################################################
    logger.info("[2/8] predicting CRISPRs...")
    query_file = f"{rootpth}/filtered_contigs.fa"
    db_host_crispr_prefix = f"{db_dir}/crispr_db/allCRISPRs"
    output_file = f"{rootpth}/{midfolder}/crispr_out.tab"

    if blast == 'blastn-short':
        crispr_call = NcbiblastnCommandline(query=query_file,db=db_host_crispr_prefix,out=output_file,outfmt="6 qseqid sseqid evalue pident length slen", evalue=1,gapopen=10,penalty=-1,
                                    gapextend=2,word_size=7,dust='no', max_target_seqs=5,
                                    task=f'blastn-short',perc_identity=90,num_threads=threads)
    elif blast == 'blastn':
        crispr_call = NcbiblastnCommandline(query=query_file,db=db_host_crispr_prefix,out=output_file,outfmt="6 qseqid sseqid evalue pident length slen", evalue=1,
                                    task='blastn', max_target_seqs=1, perc_identity=90,num_threads=threads)


    crispr_call()



    crispr_pred = {}
    with open(output_file) as file_out:
        for line in file_out.readlines():
            parse = line.replace("\n", "").split("\t")
            virus = parse[0]
            prokaryote = parse[1].split('|')[1]
            prokaryote = prokaryote.split('.')[0]
            ident = float(parse[-3])
            length = float(parse[-2])
            slen = float(parse[-1])
            if virus not in crispr_pred:
                if length/slen > cov or ident > pident:
                    crispr_pred[virus] = {'pred': prokaryote, 'ident': round(ident/100, 2)}

    pkl.dump(crispr_pred, open(f'{rootpth}/{midfolder}/crispr_pred.dict', 'wb'))



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
            logger.info("Calling genes with prodigal...")
            parallel_prodigal_gv(f'{rootpth}/filtered_contigs.fa', f'{rootpth}/{midfolder}/query_protein.fa', threads)
    elif os.path.exists(f'{rootpth}/{midfolder}/query_protein.fa'):
        logger.info("[3/8] reusing existing protein file...")
    else:
        logger.info("[3/8] calling genes with prodigal...")
        parallel_prodigal_gv(f'{rootpth}/filtered_contigs.fa', f'{rootpth}/{midfolder}/query_protein.fa', threads)
    
    if os.path.exists(f'{rootpth}/{midfolder}/self_results.abc') and os.path.exists(f'{rootpth}/{midfolder}/db_results.abc'):
        logger.info("[4/8] reusing all-against-all alignment from PhaGCN...")
    else:
        logger.info("[4/8] running all-against-all alignment...")
        # combine the database with the predicted proteins
        _ = os.system(f"cat {db_dir}/RefVirus.faa {rootpth}/{midfolder}/query_protein.fa > {rootpth}/{midfolder}/ALLprotein.fa")
        # generate the diamond database
        _ = os.system(f"diamond makedb --in {rootpth}/{midfolder}/query_protein.fa -d {rootpth}/{midfolder}/query_protein.dmnd --quiet")
        # align to the database
        _ = os.system(f"diamond blastp --db {db_dir}/RefVirus.dmnd --query {rootpth}/{midfolder}/query_protein.fa --out {rootpth}/{midfolder}/db_results.tab --outfmt 6 --threads {threads} --evalue 1e-5 --max-target-seqs 10000 --query-cover 50 --subject-cover 50 --quiet")
        _ = os.system(f"awk '{{print $1,$2,$3,$12}}' {rootpth}/{midfolder}/db_results.tab > {rootpth}/{midfolder}/db_results.abc")
        # align to itself
        _ = os.system(f"diamond blastp --db {rootpth}/{midfolder}/query_protein.dmnd --query {rootpth}/{midfolder}/query_protein.fa --out {rootpth}/{midfolder}/self_results.tab --outfmt 6 --threads {threads} --evalue 1e-5 --max-target-seqs 10000 --query-cover 50 --subject-cover 50 --quiet")
        _ = os.system(f"awk '{{print $1,$2,$3,$12}}' {rootpth}/{midfolder}/self_results.tab > {rootpth}/{midfolder}/self_results.abc")


    logger.info("[5/8] generating cherry networks...")
    # FLAGS: no proteins aligned to the database
    if os.path.getsize(f'{rootpth}/{midfolder}/db_results.abc') == 0:
        if os.path.getsize(f'{rootpth}/{midfolder}/crispr_pred.dict') == 0:
            Accession = []
            Length_list = []
            for record in SeqIO.parse(f'{contigs}', 'fasta'):
                Accession.append(record.id)
                Length_list.append(len(record.seq))
            df = pd.DataFrame({"Accession": Accession, "Host":['unknown']*len(Accession), "CHERRYScore":[0]*len(Accession), "Method":['-']*len(Accession)})
            df.to_csv(f"{rootpth}/{out_dir}/cherry_prediction.tsv", index = None, sep='\t')
            exit()
        else:
            crispr_pred = pkl.load(open(f'{rootpth}/{midfolder}/crispr_pred.dict', 'rb'))
            CRISPRs_acc2host = pkl.load(open(f'{db_dir}/CRISPRs_acc2host.pkl', 'rb'))
            Accession = []
            Length_list = []
            Pred = []
            Score = []
            Method = []
            for record in SeqIO.parse(f'{contigs}', 'fasta'):
                Accession.append(record.id)
                Length_list.append(len(record.seq))
                try:
                    Pred.append(CRISPRs_acc2host[crispr_pred[record.id]['pred']])
                    Score.append(crispr_pred[record.id]['ident'])
                    Method.append('CRISPR-based')
                except:
                    Pred.append('-')
                    Score.append(0.0)
                    Method.append('-')
            df = pd.DataFrame({"Accession": Accession, "Length": Length_list, "Host":Pred, "CHERRYScore": Score, "Method": Method})
            df.to_csv(f"{rootpth}/{out_dir}/cherry_prediction.tsv", index = None, sep='\t')
            exit()

    # add the genome size
    genome_size = defaultdict(int)
    for index, r in enumerate(SeqIO.parse(f'{rootpth}/{midfolder}/ALLprotein.fa', 'fasta')):
        genome_id = r.id.rsplit('_', 1)[0]
        genome_size[genome_id] += 1


    # filter the network
    if os.path.exists(f'{rootpth}/{midfolder}/phagcn_network.tsv'):
        cherry_network = f'phagcn_network.tsv'
        _ = os.system(f'cp {rootpth}/{out_dir}/phagcn_network_edges.tsv {rootpth}/{out_dir}/cherry_network_edges.tsv')
    else:
        cherry_network = f'cherry_network.tsv'
        compute_aai(f'{rootpth}/{midfolder}', 'db_results', genome_size)
        compute_aai(f'{rootpth}/{midfolder}', 'self_results', genome_size)
        df1 = pd.read_csv(f'{rootpth}/{midfolder}/db_results_aai.tsv', sep='\t')
        df2 = pd.read_csv(f'{rootpth}/{midfolder}/self_results_aai.tsv', sep='\t')
        df3 = pd.read_csv(f'{db_dir}/database_aai.tsv', sep='\t')
        df = pd.concat([df1, df2, df3])
        sub_df = df[((df['aai']>=aai)&((df['qcov']>=pcov)|(df['tcov']>=pcov)|(df['sgenes']>=share)))].copy()
        sub_df['score'] = sub_df['aai']/100.0 * sub_df[['qcov', 'tcov']].max(axis=1)/100.0
        # write the network
        sub_df.drop(['qcov', 'tcov', 'qgenes', 'tgenes', 'sgenes', 'aai'], axis=1, inplace=True)
        sub_df.to_csv(f'{rootpth}/{midfolder}/{cherry_network}', sep='\t', index=False, header=False)
        sub_df.rename(columns={'query':'Source', 'target':'Target', 'score':'Weight'}, inplace=True)
        sub_df.to_csv(f"{rootpth}/{out_dir}/cherry_network_edges.tsv", index=False, sep='\t')
    
    _ = os.system(f'mcl {rootpth}/{midfolder}/{cherry_network} -te {threads} -I 2.0 --abc -o {rootpth}/{midfolder}/cherry_genus_clusters.txt > /dev/null 2>&1')


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
            lineage_str = convert_lineage_to_names(lineages, taxid2name, taxid2rank)
            scores_str = ";".join(f"{score:.2f}" for score in lineages_scores)
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

    df.to_csv(f'{rootpth}/{midfolder}/cherry_clustering.tsv', index=False, sep='\t')

    
    logger.info("[6/8] predicting the host...")
    #query_df = pd.read_csv(f'{rootpth}/{midfolder}/cherry_clustering.tsv')
    query_df = df.copy()
    ref_df = pd.read_csv(f'{db_dir}/RefVirus.csv')
    acc2score = {acc: score for acc, score in zip(query_df['Accession'], query_df['Score'])}
    query_df = query_df[['Accession', 'Lineage', 'Length', 'Genus']]
    ref_df = ref_df[['Accession', 'Lineage', 'Length', 'Genus']]
    cluster_df = pd.concat([query_df, ref_df])
    genus2host = pkl.load(open(f'{db_dir}/genus2host.pkl', 'rb'))

    
    cluster_df['Host'] = cluster_df['Genus'].apply(lambda x: genus2host[x] if x in genus2host else '-')
    ref_df = cluster_df[cluster_df['Accession'].isin(ref_df['Accession'])]
    refacc2host = {acc:host for acc, host in zip(ref_df['Accession'], ref_df['Host'])}

    #prokaryote_df = pd.read_csv(f'{db_dir}/prokaryote.csv')
    CRISPRs_acc2host = pkl.load(open(f'{db_dir}/CRISPRs_acc2host.pkl', 'rb'))
    crispr2host = {}
    for acc in crispr_pred:
        host = CRISPRs_acc2host[crispr_pred[acc]['pred']]
        crispr2host[acc] = host

    # add crispr information to df['Crispr']
    cluster_df['Crispr'] = cluster_df['Accession'].apply(lambda x: crispr2host[x] if x in crispr2host else '-')
    cluster_df['Crispr_score'] = cluster_df['Accession'].apply(lambda x: crispr_pred[x]['ident'] if x in crispr_pred else 0.0)



    seq2cluster = {}
    cluster2seq = {}
    for index, line in enumerate(open(f'{rootpth}/{midfolder}/cherry_genus_clusters.txt')):
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


    # assign the Lineage according to the entry with longest length
    groups = cluster_df.groupby('cluster')
    for cluster, group in groups:
        acc_list = group['Accession']
        # check whether  NCBI accession (NC_) exist
        if any('NC_' in acc for acc in acc_list):
            continue
        idx = group['Length'].idxmax()
        Lineage = group.loc[idx, 'Lineage']
        cluster_df.loc[group.index, 'Lineage'] = Lineage
        Genus = group.loc[idx, 'Genus']
        cluster_df.loc[group.index, 'Genus'] = Genus
        Host = group.loc[idx, 'Host']
        cluster_df.loc[group.index, 'Host'] = Host

    # assign the Lineage according to the entry in database
    for cluster, group in groups:
        acc_list = group['Accession']
        # check whether  NCBI accession (NC_) exist
        if not any('NC_' in acc for acc in acc_list):
            continue
        ref_group = group[group['Accession'].apply(lambda x: 'NC_' in x)]
        query_group = group[group['Accession'].isin(query_df['Accession'])]
        if len(ref_group[ref_group['Genus'] != '-']['Genus'].unique()) == 1:
            Lineage = ref_group['Lineage'].values[0]
            cluster_df.loc[query_group.index, 'Lineage'] = Lineage
            Genus = ref_group['Genus'].values[0]
            cluster_df.loc[group.index, 'Genus'] = Genus
            Host = ref_group['Host'].values[0]
            cluster_df.loc[group.index, 'Host'] = Host
        else:
            # assign the host for the query group accoding to the genus
            for idx, row in query_group[query_group['Genus'] != '-'].iterrows():
                try:
                    cluster_df.loc[idx, 'Host'] = genus2host[row['Genus']]
                except:
                    pass


    df = cluster_df[cluster_df['Accession'].isin(genomes.keys())].copy()
    df = df.reset_index(drop=True)

    logger.info("[7/8] summarizing the results...")
    predicted = df[df['Host'] != '-']
    unpredicted = df[df['Host'] == '-']

    df.loc[predicted.index, 'Score'] = [acc2score[acc].split(';')[-1] for acc in predicted['Accession']]
    # Update 'Method' column where 'Host' is not '-'
    df.loc[predicted.index, 'Method'] = 'AAI-based'


    groups = unpredicted.groupby('cluster')
    cluster_idx = 0
    for cluster, group in groups:
        idx = group['Crispr_score'].idxmax()
        host = group.loc[idx, 'Crispr']
        df.loc[group.index, 'Host'] = host.replace(' ', '_')
        df.loc[group.index, 'Score'] = group.loc[idx, 'Crispr_score']
        if host == '-':
            df.loc[group.index, 'Method'] = '-'
        else:
            df.loc[group.index, 'Method'] = 'CRISPR-based'

    df = df.sort_values('Accession', ascending=False)

    df = df[['Accession', 'Length', 'Host', 'Score', 'Method']]
    df.to_csv(f'{rootpth}/{midfolder}/cherry_prediction.tsv', index=False, sep='\t')


    ###############################################################
    ####################### dump results ##########################
    ###############################################################\
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
            unpredicted_contig.append(record.id)
            unpredicted_length.append(len(record.seq))



    # Create lists by combining existing data with new entries
    all_contigs = df['Accession'].tolist() + filtered_contig + unpredicted_contig
    all_pred = df['Host'].tolist() + ['filtered'] * len(filtered_contig) + ['-'] * len(unpredicted_contig)
    all_score = df['Score'].tolist() + [0] * len(filtered_contig) + [0] * len(unpredicted_contig)
    all_length = df['Length'].tolist() + filtered_lenth + unpredicted_length
    all_method = df['Method'].tolist() + ['-'] * len(filtered_contig) + ['-'] * len(unpredicted_contig)

    # Create DataFrame directly from lists
    contig_to_pred = pd.DataFrame({
        'Accession': all_contigs,
        'Length': all_length,
        'Host': all_pred,
        'CHERRYScore': all_score,
        'Method': all_method
    })

    # Filter genus for a specific condition
    # all_pie = contig_to_pred['Genus'][contig_to_pred['Genus'] != '-']

    # Save DataFrame to CSV
    contig_to_pred.to_csv(f"{rootpth}/{out_dir}/cherry_prediction.tsv", index=False, sep='\t')
    query2host = {acc: genus for acc, genus in zip(df['Accession'], df['Host'])}


    cherry_node = pd.DataFrame({
        'Accession': list(refacc2host.keys()) + list(query2host.keys()), 
        'Host': list(refacc2host.values()) + list(query2host.values()), 
        'TYPE': ['Ref']*len(refacc2host) + ['Query']*len(query2host)
        })
    
    cherry_node.to_csv(f"{rootpth}/{out_dir}/cherry_network_nodes.tsv", index=False, sep='\t')

    if inputs.draw == 'Y':
        draw_network(f'{rootpth}/{out_dir}', f'{rootpth}/{out_dir}', 'cherry')
    
    if inputs.task != 'end_to_end':
        genes = {}
        for record in SeqIO.parse(f'{rootpth}/{midfolder}/query_protein.fa', 'fasta'):
            gene = Gene()
            rec_info = record.description.split()
            gene.id = rec_info[0]
            gene.start = int(rec_info[2])
            gene.end = int(rec_info[4])
            gene.strand = int(rec_info[6])
            gene.genome_id = gene.id.rsplit("_", 1)[0]
            gene.gc = float(rec_info[-1].split('gc_cont=')[-1])
            gene.anno = 'hypothetical protein'
            genes[gene.id] = gene
            genomes[gene.genome_id].genes.append(gene.id)


        _ = os.system(f"cp {rootpth}/{midfolder}/query_protein.fa {rootpth}/{out_dir}/all_predicted_protein.fa")
        _ = os.system(f"cp {rootpth}/{midfolder}/db_results.tab {rootpth}/{out_dir}/alignment_results.tab")
        _ = os.system(f'sed -i "1i\qseqid\tsseqid\tpident\tlength\tmismatch\tgapopen\tqstart\tqend\tsstart\tsend\tevalue" {rootpth}/{out_dir}/alignment_results.tab')

        anno_df = pkl.load(open(f'{db_dir}/RefVirus_anno.pkl', 'rb'))
        # protein annotation
        for ORF in ORF2hits:
            annotations = []
            for hit, _ in ORF2hits[ORF]:
                try:
                    annotations.append(anno_df[hit])
                except:
                    pass
            if annotations:
                genes[ORF].anno = Counter(annotations).most_common()[0][0]

        # write the gene annotation by genomes
        with open(f'{rootpth}/{out_dir}/gene_annotation.tsv', 'w') as f:
            f.write('Genome\tORF\tStart\tEnd\tStrand\tGC\tAnnotation\n')
            for genome in genomes:
                for gene in genomes[genome].genes:
                    f.write(f'{genome}\t{gene}\t{genes[gene].start}\t{genes[gene].end}\t{genes[gene].strand}\t{genes[gene].gc}\t{genes[gene].anno}\n')


    
    logger.info("Run time: %s seconds\n" % round(time.time() - program_start, 2))