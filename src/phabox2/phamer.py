
#!/usr/bin/env python
import os
import pandas as pd
import pickle as pkl

from .scripts.ulity import *
from .scripts.preprocessing import *
from .scripts.parallel_prodigal_gv import main as parallel_prodigal_gv
from .models.phamer import Transformer
import time
from tqdm import tqdm






def run(inputs):
    logger = get_logger()
    if inputs.skip == 'Y' and inputs.task == 'phamer':
        logger.info("Error: parameter 'skip' should be 'Y' when running the task 'phamer'")
        exit()
    if inputs.skip == 'N':
        logger.info("Running program: PhaMer (virus identification)")
    else:
        logger.info("Running program: Data preprocessing")
    program_start = time.time()

    contigs   = inputs.contigs
    midfolder = inputs.midfolder
    out_dir   = 'final_prediction/'
    rootpth   = inputs.outpth
    db_dir    = inputs.dbdir
    parampth  = inputs.dbdir
    threads   = inputs.threads
    reject    = inputs.reject/100

    if not os.path.exists(db_dir):
        print(f'Database directory {db_dir} missing or unreadable')
        exit(1)

    if inputs.skip == 'N':
        supplementary = 'phamer_supplementary'
    else:
        supplementary = 'preprocessing_supplementary'
    check_path(os.path.join(rootpth, out_dir))
    check_path(os.path.join(rootpth, out_dir, supplementary))
    check_path(os.path.join(rootpth, midfolder))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        torch.set_num_threads(inputs.threads)

    ###############################################################
    #######################  Filter length ########################
    ###############################################################
    logger.info("[1/7] filtering the length of contigs...")
    genomes = {}
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
            genome.proportion = 0
            genomes[genome.id] = genome

    if not rec:
        with open(f'{rootpth}/{out_dir}/phamer_prediction.tsv', 'w') as file_out:
            file_out.write("Accession\tLength\tPred\tPhaMerScore\tPhaMerConfidence\n")
            for record in SeqIO.parse(contigs, 'fasta'):
                file_out.write(f'{record.id}\t{len(record.seq)}\tfiltered\t0\trejected\n')
        if inputs.skip == 'N':
            logger.info(f"PhaMer finished! please check the results in {os.path.join(rootpth,out_dir, 'phamer_prediction.tsv')}")
        else:
            logger.info(f"Data preprocessing finished! please check the results in {os.path.join(rootpth,out_dir, 'phamer_prediction.tsv')}")
        exit()

    _ = SeqIO.write(rec, f'{rootpth}/filtered_contigs.fa', 'fasta')


    ###############################################################
    ##########################  PhaMer ############################
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
            _ = SeqIO.write(rec, f'{rootpth}/{midfolder}/query_protein.fa', 'fasta')
        else:
            logger.info("WARNING: no proteins found in the provided file.\nPlease check whether the genes is called by the prodigal.")
            logger.info("[2/7] calling genes with prodigal...")
            parallel_prodigal_gv(f'{rootpth}/filtered_contigs.fa', f'{rootpth}/{midfolder}/query_protein.fa', threads)
    else:
        logger.info("[2/7] calling genes with prodigal...")
        parallel_prodigal_gv(f'{rootpth}/filtered_contigs.fa', f'{rootpth}/{midfolder}/query_protein.fa', threads)
    logger.info("[3/7] running all-against-all alignment...")



    genes = load_gene_info(f'{rootpth}/{midfolder}/query_protein.fa', genomes)

    

    # align to the database
    run_command(f"diamond blastp --db {db_dir}/RefVirus.dmnd --query {rootpth}/{midfolder}/query_protein.fa --out {rootpth}/{midfolder}/db_results.tab --outfmt 6 --threads {threads} --evalue 1e-5 --max-target-seqs 10000 --query-cover 50 --subject-cover 50 --quiet")
    run_command(f"awk '{{print $1,$2,$3,$12}}' {rootpth}/{midfolder}/db_results.tab > {rootpth}/{midfolder}/db_results.abc")
    
    # FLAGS: no proteins aligned to the database
    if os.path.getsize(f'{rootpth}/{midfolder}/db_results.abc') == 0:
        Accession = []
        Length_list = []
        Pred_tmp = []
        for record in SeqIO.parse(f'{contigs}', 'fasta'):
            Accession.append(record.id)
            Length_list.append(len(record.seq))
            Pred_tmp.append('-')

        df = pd.DataFrame({"Accession": Accession, "Length": Length_list, "Pred":['non-virus']*len(Accession), "PhaMerScore":[0]*len(Accession), "PhaMerConfidence":['rejected']*len(Accession)})
        df.to_csv(f"{rootpth}/{out_dir}/phamer_prediction.tsv", index = None, sep='\t')
        exit()
    
    logger.info("[4/7] converting sequences to sentences for language model...")
    sentence, id2contig, proportion, pcs2idx = contig2sentence(db_dir, os.path.join(rootpth, midfolder), genomes)

    if inputs.skip == 'N':
        logger.info("[5/7] Predicting the viruses...")
    else:
        logger.info("[5/7] Analyzing the seuqences...")
    #pcs2idx = pkl.load(open(f'{rootpth}/{midfolder}/phamer_pc2wordsid.dict', 'rb'))
    num_pcs = len(set(pcs2idx.keys()))
    src_vocab_size = num_pcs+1
    model, _, _ = reset_model(Transformer, src_vocab_size, device)
    try:
        pretrained_dict=torch.load(f'{parampth}/transformer.pth', map_location=device)
        model.load_state_dict(pretrained_dict)
    except:
        print('cannot find pre-trained model')
        exit(1)


    all_pred = []
    all_score = []
    all_proportion = []
    all_confidence = []
    with torch.no_grad():
        _ = model.eval()
        with tqdm(total=len(sentence)) as pbar:
            for idx in range(0, len(sentence), 500):
                try:
                    batch_x = sentence[idx: idx+500]
                    weight  = proportion[idx: idx+500]
                except:
                    batch_x = sentence[idx:]
                    weight  = proportion[idx:]
                batch_x = return_tensor(batch_x, device).long()
                logit = model(batch_x)
                logit = torch.sigmoid(logit.squeeze(1))
                for score, pro in zip(logit, weight):
                    if score < 0.5 or pro < reject:
                        all_pred.append('non-virus')
                        if score < 0.5:
                            all_score.append(float('{:.2f}'.format(score)))
                        else:
                            all_score.append(float('{:.2f}'.format(pro)))
                        all_proportion.append(float('{:.2f}'.format(pro)))
                        if pro < reject:
                            all_confidence.append('lower than reject threshold')
                        else:
                            all_confidence.append('lower than viral score threshold; proteinal prophage, please run contamination detection task')
                    else:
                        all_pred.append('virus')
                        all_score.append(float('{:.2f}'.format(score)))
                        all_proportion.append(float('{:.2f}'.format(pro)))
                        if pro < reject:
                            all_confidence.append('lower than reject threshold')
                        elif (pro+score)/2 > 0.8:
                            all_confidence.append('high-confidence')
                        elif (pro+score)/2 > 0.6:
                            all_confidence.append('medium-confidence')
                        else:
                            all_confidence.append('low-confidence; please run contamination detection task')
                    
                _ = pbar.update(len(batch_x))


    logger.info("[6/7] summarizing the results...")


    contigs_list = {item:1 for item in list(id2contig.values())}
    contigs_add = []
    length_add = []
    seq_dict = {}
    for record in SeqIO.parse(f'{contigs}', 'fasta'):
        seq_dict[record.id] = str(record.seq)
        try:
            _ = contigs_list[record.id]
        except:
            if len(record.seq) < inputs.len:
                contigs_add.append(record.id)
                length_add.append(len(record.seq))
                all_pred.append('filtered')
                all_score.append(0)
                all_proportion.append(0)
                all_confidence.append('rejected')
                continue
            if genomes[record.id].proportion < reject:
                contigs_add.append(record.id)
                length_add.append(len(record.seq))
                all_pred.append('non-virus')
                all_score.append(0)
                all_proportion.append(0)
                all_confidence.append('lower than reject threshold')
            else:
                contigs_add.append(record.id)
                length_add.append(len(record.seq))
                all_pred.append('virus')
                all_score.append(float('{:.2f}'.format(genomes[record.id].proportion)))
                all_proportion.append(float('{:.2f}'.format(genomes[record.id].proportion)))
                if genomes[record.id].proportion > 0.75:
                    all_confidence.append('high-confidence')
                elif genomes[record.id].proportion > 0.25:
                    all_confidence.append('medium-confidence')
                else:
                    all_confidence.append('low-confidence; please run contamination detection task')


    length_list = [genomes[item].length for item in contigs_list] + length_add
    contigs_list = list(contigs_list.keys()) + contigs_add

    logger.info("[7/7] writing the results...")
    if inputs.skip == 'N':
        pred_csv = pd.DataFrame({"Accession":contigs_list, "Length":length_list, "Pred":all_pred, "Proportion":all_proportion, "PhaMerScore":all_score, "PhaMerConfidence":all_confidence})
        pred_csv.to_csv(f'{rootpth}/{out_dir}/phamer_prediction.tsv', index = False, sep='\t')
        virus_list = {item:1 for item in pred_csv[pred_csv['Pred'] == 'virus']['Accession'].values}

        virus_rec = []
        low_confidence = {item:1 for item in pred_csv[pred_csv['PhaMerConfidence'] == 'low-confidence; please run contamination detection task']['Accession'].values}
        low_confidence = {**low_confidence, **{item:1 for item in pred_csv[pred_csv['PhaMerConfidence'] == 'lower than viral score threshold; proteinal prophage, please run contamination detection task']['Accession'].values}}
        low_virus_rec = []
        for record in SeqIO.parse(f'{contigs}', 'fasta'):
            try:
                _ = low_confidence[record.id]
                low_virus_rec.append(record)
            except:
                pass
            try:
                _ = virus_list[record.id]
                virus_rec.append(record)
            except:
                pass
        
            

        SeqIO.write(virus_rec, f'{rootpth}/{out_dir}/{supplementary}/predicted_virus.fa', 'fasta')
        SeqIO.write(low_virus_rec, f'{rootpth}/{out_dir}/{supplementary}/uncertain_sequences_for_contamination_task.fa', 'fasta')
        virus_protein_rec = []
        check = {item: 1 for item in virus_list}
        for record in SeqIO.parse(f'{rootpth}/{midfolder}/query_protein.fa', 'fasta'):
            try:
                _ = check[record.id.rsplit('_', 1)[0]]
                virus_protein_rec.append(record)
            except:
                pass
    
        SeqIO.write(virus_protein_rec, f'{rootpth}/{out_dir}/{supplementary}/predicted_virus_protein.fa', 'fasta')  
    
        run_command(f"cp {rootpth}/filtered_contigs.fa {rootpth}/{out_dir}/{supplementary}/all_predicted_contigs.fa")      
        run_command(f"cp {rootpth}/{out_dir}/{supplementary}/predicted_virus.fa {rootpth}/filtered_contigs.fa")

    else:
        run_command(f"cp {rootpth}/filtered_contigs.fa {rootpth}/{out_dir}/{supplementary}/all_predicted_contigs.fa")      

    run_command(f"cp {rootpth}/{midfolder}/query_protein.fa {rootpth}/{out_dir}/{supplementary}/all_predicted_protein.fa")
    run_command(f"cp {rootpth}/{midfolder}/db_results.tab {rootpth}/{out_dir}/{supplementary}/alignment_results.tab")
    tmp = pd.read_csv(f'{rootpth}/{out_dir}/{supplementary}/alignment_results.tab', sep='\t', names=['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'])
    tmp.to_csv(f'{rootpth}/{out_dir}/{supplementary}/alignment_results.tab', sep='\t', index=False)
    #run_command(f"sed -i '1i\qseqid\tsseqid\tpident\tlength\tmismatch\tgapopen\tqstart\tqend\tsstart\tsend\tevalue\tbitscore' {rootpth}/{out_dir}/{supplementary}/alignment_results.tab")
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
        gene.inference = db2ref.get(row['sseqid'], '-')
        
    # write the gene annotation by genomes
    with open(f'{rootpth}/{out_dir}/{supplementary}/gene_annotation.tsv', 'w') as f:
        f.write('Genome\tORF\tStart\tEnd\tStrand\tGC\tAnnotation\tpident\tcoverage\tcloest_gene\n')
        for genome in genomes:
            for gene in genomes[genome].genes:
                f.write(f'{genome}\t{gene}\t{genes[gene].start}\t{genes[gene].end}\t{genes[gene].strand}\t{genes[gene].gc}\t{genes[gene].anno}\t{genes[gene].pident:.2f}\t{genes[gene].coverage:.2f}\t{genes[gene].inference}\n')


    phavip_dump_result(genomes, rootpth, out_dir, logger, supplementary = supplementary)
    
    logger.info("Run time: %s seconds\n" % round(time.time() - program_start, 2))

    if inputs.skip == 'N':
        if not virus_rec: 
            logger.info("PhaMer finished! No virus found in the contigs!")
            logger.info(f"Please check the results in {os.path.join(rootpth,out_dir, 'phamer_prediction.tsv')}")
            exit()





