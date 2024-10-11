#!/usr/bin/env python
import os
import time
import pandas as pd
import numpy as np
import pickle as pkl

from scripts.ulity import *
from scripts.preprocessing import *
from scripts.parallel_prodigal_gv import main as parallel_prodigal_gv
from Bio import SeqIO
from scipy.special import softmax





def run(inputs):
    logger = get_logger()
    logger.info("Running program: PhaTYP (Lifestyle prediction)")
    program_start = time.time()

    contigs   = inputs.contigs
    midfolder = inputs.midfolder
    out_dir   = 'final_prediction/'
    rootpth   = inputs.outpth
    db_dir    = inputs.dbdir
    parampth  = inputs.dbdir
    threads   = inputs.threads

    if not os.path.isfile(contigs):
        print(f"cannot find the files {contigs}")
        exit()

    if not os.path.exists(db_dir):
        print(f'Database directory {db_dir} missing or unreadable')
        exit(1)

    check_path(os.path.join(rootpth, out_dir))
    check_path(os.path.join(rootpth, midfolder))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("running with cpu")
        torch.set_num_threads(inputs.threads)


    ###############################################################
    #######################  Filter length ########################
    ###############################################################
    genomes = {}
    if os.path.exists(f'{rootpth}/filtered_contigs.fa'):
        logger.info("[1/7] reusing existing filtered contigs...")
        for record in SeqIO.parse(f'{rootpth}/filtered_contigs.fa', 'fasta'):
            genome = Genome()
            genome.id = record.id
            genome.length = len(record.seq)
            genome.genes = []
            genome.viral_hits = {}
            genome.regions = None
            genomes[genome.id] = genome
    else:
        logger.info("[1/7] filtering the length of contigs...")
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
            with open(f'{rootpth}/{out_dir}/phatyp_prediction.tsv', 'w') as file_out:
                file_out.write("Accession\tLength\tTYPE\tPhaTYPScore\n")
                for record in SeqIO.parse(contigs, 'fasta'):
                    file_out.write(f'{record.id}\t{len({record.seq})}\tfiltered\t0\n')
            logger.info(f"PhaTYP finished! please check the results in {os.path.join(rootpth,out_dir, 'phatyp_prediction.tsv')}")
            exit()

        SeqIO.write(rec, f'{rootpth}/filtered_contigs.fa', 'fasta')


    ###############################################################
    ##########################  PhaTYP ############################
    ###############################################################
    if os.path.exists(f'{inputs.proteins}'):
        logger.info("[2/7] using provided protein file...")
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
            logger.info("[2/7] calling genes with prodigal...")
            parallel_prodigal_gv(f'{rootpth}/filtered_contigs.fa', f'{rootpth}/{midfolder}/query_protein.fa', threads)
    elif os.path.exists(f'{rootpth}/{midfolder}/query_protein.fa'):
        logger.info("[2/7] reusing existing protein file...")
    else:
        logger.info("[2/7] calling genes with prodigal...")
        parallel_prodigal_gv(f'{rootpth}/filtered_contigs.fa', f'{rootpth}/{midfolder}/query_protein.fa', threads)
    

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
        try:
            genomes[gene.genome_id].genes.append(gene.id)
        except:
            pass
        

    if os.path.exists(f'{rootpth}/{midfolder}/db_results.tab'):
        logger.info("[3/7] using existing all-against-all alignment results...")
        #_ = os.system(f"awk '{{print $1,$2,$3,$12}}' {rootpth}/{midfolder}/db_results.tab > {rootpth}/{midfolder}/db_results.abc")
    else:
        logger.info("[3/7] running all-against-all alignment...")
        # align to the database
        _ = os.system(f"diamond blastp --db {db_dir}/RefVirus.dmnd --query {rootpth}/{midfolder}/query_protein.fa --out {rootpth}/{midfolder}/db_results.tab --outfmt 6 --threads {threads} --evalue 1e-5 --max-target-seqs 10000 --query-cover 50 --subject-cover 50 --quiet")
        _ = os.system(f"awk '{{print $1,$2,$3,$12}}' {rootpth}/{midfolder}/db_results.tab > {rootpth}/{midfolder}/db_results.abc")
    

    # FLAGS: no proteins aligned to the database
    if os.path.getsize(os.path.join(rootpth, midfolder, 'db_results.abc')) == 0:
        Accession = []
        Length_list = []
        Pred_tmp = []
        for record in SeqIO.parse(f'{contigs}', 'fasta'):
            Accession.append(record.id)
            Length_list.append(len(record.seq))
            Pred_tmp.append('unknown')

        df = pd.DataFrame({"Accession": Accession, "Length": Length_list, "TYPE":Pred_tmp, "PhaTYPScore":[0]*len(Accession)})
        df.to_csv(os.path.join(rootpth, out_dir, "phatyp_prediction.tsv"), index = None, sep='\t')
        exit()

    logger.info("[4/7] converting sequences to sentences for language model...")
    sentence, id2contig, _, pc2wordsid = contig2sentence(db_dir, os.path.join(rootpth, midfolder), genomes)
    generate_bert_input(os.path.join(rootpth, midfolder), sentence, pc2wordsid)

    logger.info("[5/7] Predicting the lifestyle...")
    #id2contig  = pkl.load(open(f'{rootpth}/{midfolder}/phatyp_sentence_id2contig.dict', 'rb'))

    trainer, tokenized_data = init_bert(rootpth, midfolder, parampth)

    with torch.no_grad():
        pred, _, _ = trainer.predict(tokenized_data["test"])


    logger.info("[6/7] summarizing the results...")
    prediction_value = []
    for item in pred:
        prediction_value.append(softmax(item))
    prediction_value = np.array(prediction_value)



    all_pred = []
    all_score = []
    for score in prediction_value:
        pred = np.argmax(score)
        if pred == 1:
            all_pred.append('temperate')
            all_score.append(float('{:.2f}'.format(score[1])))
        else:
            all_pred.append('virulent')
            all_score.append(float('{:.2f}'.format(score[0])))


    contigs_list = {item:1 for item in list(id2contig.values())}
    contigs_add = []
    length_dict = {}
    seq_dict = {}
    for record in SeqIO.parse(f'{contigs}', 'fasta'):
        seq_dict[record.id] = str(record.seq)
        length_dict[record.id] = len(record.seq)
        try:
            _ = contigs_list[record.id]
        except:
            if len(record.seq) < inputs.len:
                contigs_add.append(record.id)
                all_pred.append('filtered')
                all_score.append('0')
                continue
            contigs_add.append(record.id)
            all_pred.append('unpredicted')
            all_score.append('0')

    contigs_list = list(contigs_list.keys()) + contigs_add
    length_list = [length_dict[item] for item in contigs_list]

    logger.info("[7/7] writing the results...")
    pred_csv = pd.DataFrame({"Accession":contigs_list, "Length":length_list, "TYPE":all_pred, "PhaTYPScore":all_score})
    if inputs.task == 'end_to_end':
        # only predict for the prokaryotic group
        ProkaryoticGroup = pkl.load(open(f'{db_dir}/ProkaryoticGroup.pkl', 'rb'))
        taxonomy_df = pd.read_csv(f'{rootpth}/{out_dir}/phagcn_prediction.tsv', sep='\t')
        merged_df = pred_csv.merge(taxonomy_df[['Accession', 'Lineage']], on='Accession', how='left')
        rows_to_update = ~merged_df['Lineage'].apply(lambda lineage: any(group in lineage for group in ProkaryoticGroup))
        # Update the pred_csv dataframe
        pred_csv.loc[rows_to_update, 'TYPE'] = 'unpredicted'
        pred_csv.loc[rows_to_update, 'PhaTYPScore'] = '0'

    pred_csv.to_csv(f'{rootpth}/{out_dir}/phatyp_prediction.tsv', index = False, sep='\t')

    if inputs.task != 'end_to_end':
        _ = os.system(f"cp {rootpth}/{midfolder}/query_protein.fa {rootpth}/{out_dir}/all_predicted_protein.fa")
        _ = os.system(f"cp {rootpth}/{midfolder}/db_results.tab {rootpth}/{out_dir}/alignment_results.tab")
        _ = os.system(f"sed -i '1i\qseqid\tsseqid\tpident\tlength\tmismatch\tgapopen\tqstart\tqend\tsstart\tsend\tevalue' {rootpth}/{out_dir}/alignment_results.tab")
        
        anno_df = pkl.load(open(f'{db_dir}/RefVirus_anno.pkl', 'rb'))
            # protein annotation
        ORF2hits, _ = parse_alignment(f'{rootpth}/{midfolder}/db_results.abc')
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


