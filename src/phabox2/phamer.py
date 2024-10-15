
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
    logger.info("Running program: PhaMer (virus identification)")
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

    check_path(os.path.join(rootpth, out_dir))
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
        with open(f'{rootpth}/{out_dir}/phamer_prediction.tsv', 'w') as file_out:
            file_out.write("Accession\tLength\tPred\tPhaMerScore\tPhaMerConfidence\n")
            for record in SeqIO.parse(contigs, 'fasta'):
                file_out.write(f'{record.id}\t{len({record.seq})}\tfiltered\t0\trejected\n')
        logger.info(f"PhaMer finished! please check the results in {os.path.join(rootpth,out_dir, 'phamer_prediction.tsv')}")
        exit()

    SeqIO.write(rec, f'{rootpth}/filtered_contigs.fa', 'fasta')


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
    

    # align to the database
    _ = os.system(f"diamond blastp --db {db_dir}/RefVirus.dmnd --query {rootpth}/{midfolder}/query_protein.fa --out {rootpth}/{midfolder}/db_results.tab --outfmt 6 --threads {threads} --evalue 1e-5 --max-target-seqs 10000 --query-cover 50 --subject-cover 50 --quiet")
    _ = os.system(f"awk '{{print $1,$2,$3,$12}}' {rootpth}/{midfolder}/db_results.tab > {rootpth}/{midfolder}/db_results.abc")
    
    # FLAGS: no proteins aligned to the database
    if os.path.getsize(f'{rootpth}/{midfolder}/db_results.abc') == 0:
        Accession = []
        Length_list = []
        Pred_tmp = []
        for record in SeqIO.parse(f'{contigs}', 'fasta'):
            Accession.append(record.id)
            Length_list.append(len(record.seq))
            Pred_tmp.append('unknown')

        df = pd.DataFrame({"Accession": Accession, "Length": Length_list, "Pred":['non-virus']*len(Accession), "PhaMerScore":[0]*len(Accession), "PhaMerConfidence":['rejected']*len(Accession)})
        df.to_csv(f"{rootpth}/{out_dir}/phamer_prediction.tsv", index = None, sep='\t')
        exit()
    
    logger.info("[4/7] converting sequences to sentences for language model...")
    sentence, id2contig, proportion, pcs2idx = contig2sentence(db_dir, os.path.join(rootpth, midfolder), genomes)

    logger.info("[5/7] Predicting the viruses...")
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


    #sentence   = pkl.load(open(f'{rootpth}/{midfolder}/phamer_sentence.feat', 'rb'))
    #id2contig  = pkl.load(open(f'{rootpth}/{midfolder}/phamer_sentence_id2contig.dict', 'rb'))
    #proportion = pkl.load(open(f'{rootpth}/{midfolder}/phamer_sentence_proportion.feat', 'rb'))

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
                    if score < 0.5 and pro < reject:
                        all_pred.append('non-virus')
                        all_score.append(float('{:.2f}'.format(pro)))
                        all_proportion.append(float('{:.2f}'.format(pro)))
                    else:
                        all_pred.append('virus')
                        all_score.append(float('{:.2f}'.format(score)))
                        all_proportion.append(float('{:.2f}'.format(pro)))
                    if pro > 0.75:
                        all_confidence.append('high-confidence')
                    elif pro > 0.25:
                        all_confidence.append('medium-confidence')
                    elif pro > reject:
                        all_confidence.append('low-confidence')
                    else:
                        all_confidence.append('lower than reject threshold')
                _ = pbar.update(len(batch_x))


    logger.info("[6/7] summarizing the results...")


    contigs_list = {item:1 for item in list(id2contig.values())}
    contigs_add = []
    seq_dict = {}
    for record in SeqIO.parse(f'{contigs}', 'fasta'):
        seq_dict[record.id] = str(record.seq)
        try:
            _ = contigs_list[record.id]
        except:
            if len(record.seq) < inputs.len:
                contigs_add.append(record.id)
                all_pred.append('filtered')
                all_score.append(0)
                all_proportion.append(0)
                all_confidence.append('rejected')
                continue
            if genomes[record.id].proportion < reject:
                contigs_add.append(record.id)
                all_pred.append('non-virus')
                all_score.append(0)
                all_proportion.append(0)
                all_confidence.append('non-virus')
            else:
                contigs_add.append(record.id)
                all_pred.append('virus')
                all_score.append(float('{:.2f}'.format(genomes[record.id].proportion)))
                all_proportion.append(float('{:.2f}'.format(genomes[record.id].proportion)))
                if genomes[record.id].proportion > 0.75:
                    all_confidence.append('high-confidence')
                elif genomes[record.id].proportion > 0.25:
                    all_confidence.append('medium-confidence')
                else:
                    all_confidence.append('low-confidence')



    contigs_list = list(contigs_list.keys()) + contigs_add
    length_list = [genomes[item].length for item in contigs_list]

    logger.info("[7/7] writing the results...")
    pred_csv = pd.DataFrame({"Accession":contigs_list, "Length":length_list, "Pred":all_pred, "Proportion":all_proportion, "PhaMerScore":all_score, "PhaMerConfidence":all_confidence})
    pred_csv.to_csv(f'{rootpth}/{out_dir}/phamer_prediction.tsv', index = False, sep='\t')
    virus_list = pred_csv[pred_csv['Pred'] == 'virus']['Accession'].values

    virus_rec = []
    for record in SeqIO.parse(f'{contigs}', 'fasta'):
        if record.id in virus_list:
            virus_rec.append(record)

    SeqIO.write(virus_rec, f'{rootpth}/{out_dir}/predicted_virus.fa', 'fasta')
    _ = os.system(f"cp {rootpth}/{out_dir}/predicted_virus.fa {rootpth}/filtered_contigs.fa")


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





