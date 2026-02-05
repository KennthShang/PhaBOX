#!/usr/bin/env python
import os
import time
import pandas as pd
import pickle as pkl


from .scripts.ulity import *
from .scripts.preprocessing import *
from .scripts.parallel_prodigal_gv import main as parallel_prodigal_gv
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

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
    pident    = inputs.cpident
    cov       = inputs.ccov/100
    aai       = inputs.aai
    pcov      = inputs.pcov
    share     = inputs.share
    blast     = inputs.blast
    bfolder   = inputs.bfolder
    magonly   = inputs.magonly
    bgtdb     = inputs.bgtdb
    prolen    = inputs.prolen

    if magonly == 'Y':
        jy = 4
    else:
        jy = 10

    if bfolder == 'None' and magonly == 'Y':
        print('In consistent input, please provide the MAGs folder or set the magonly flag to False.')
        exit(1)

    if bfolder == 'None' and bgtdb != 'None':
        print('In consistent input, please provide the MAGs folder (--bfolder) corresponding to your gtdb file.')
        exit(1)

    if not os.path.isfile(contigs):
        print('cannot find the input contigs file')
        exit(1)

    if not os.path.exists(db_dir):
        print(f'Database directory {db_dir} missing or unreadable')
        exit(1)

    if bgtdb != 'None':
        if not os.path.isfile(bgtdb):
            print(f'GTDB file {bgtdb} missing or unreadable')
            exit(1)

    supplementary = 'cherry_supplementary'
    check_path(os.path.join(rootpth, out_dir))
    check_path(os.path.join(rootpth, out_dir, supplementary))
    check_path(os.path.join(rootpth, midfolder))



    ###############################################################
    #######################  Filter length ########################
    ###############################################################
    
    genomes = {}
    if os.path.exists(f'{rootpth}/filtered_contigs.fa'):
        logger.info(f"[1/{jy}] reusing existing filtered contigs...")
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
            with open(f'{rootpth}/{out_dir}/cherry_prediction.tsv', 'w') as file_out:
                file_out.write("Accession\tLength\tHost\tCHERRYScore\tMethod\n")
                for record in SeqIO.parse(contigs, 'fasta'):
                    file_out.write(f'{record.id},{len(record.seq)},filtered,0,-\n')
            logger.info(f"Cherry finished! please check the results in {os.path.join(rootpth,out_dir, 'cherry_prediction.tsv')}")
            exit()
    else:
        logger.info(f"[1/{jy}] filtering the length of contigs...")
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
            with open(f'{rootpth}/{out_dir}/cherry_prediction.tsv', 'w') as file_out:
                file_out.write("Accession\tLength\tHost\tCHERRYScore\tMethod\n")
                for record in SeqIO.parse(contigs, 'fasta'):
                    file_out.write(f'{record.id},{len(record.seq)},filtered,0,-\n')
            logger.info(f"Cherry finished! please check the results in {os.path.join(rootpth,out_dir, 'cherry_prediction.tsv')}")
            exit()

        _ = SeqIO.write(rec, f'{rootpth}/filtered_contigs.fa', 'fasta')



    ###############################################################
    ##############  CRISPRs (MAG/BLASTN/tRNA/Kmer)  ###############
    ###############################################################
    blast_pred_mag = {}
    crispr_pred_mag = {}
    tRNA_pred_mag = {}
    virus2host_kmer_reorg = {}
    crispr_pred_mag = {}
    if bfolder != 'None':
        # running CRT
        logger.info(f"[2/{jy}] finding CRISPRs from MAGs...")
        check_path(f"{rootpth}/{midfolder}/crispr_tmp")
        check_path(f"{rootpth}/{midfolder}/crispr_fa")

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {
                executor.submit(
                    run_crt,
                    db_dir,
                    os.path.join(bfolder, bfile),
                    os.path.join(rootpth, midfolder, 'crispr_tmp', bfile.rsplit('.', 1)[0] + '.crispr'),
                    bfile
                ): bfile for bfile in os.listdir(bfolder)
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    logger.info(f"Error in running CRT for MAG: {result} (ignored)")



        # write the CRISPRs to fasta
        total_rec = []
        if os.path.exists(f"{rootpth}/{midfolder}/crispr_mag.tab"):
            logger.info(f"reusing existing CRISPRs file...")
            crispr_df = pd.read_csv(f"{rootpth}/{midfolder}/crispr_mag.tab", sep='\t', names = ['qseqid', 'sseqid', 'evalue', 'pident', 'length', 'slen'])
            crispr_df['cov'] = crispr_df['length']/crispr_df['slen']
            crispr_df = crispr_df[(crispr_df['cov'] > cov) & (crispr_df['pident'] > pident)]
            crispr_df.to_csv(f"{rootpth}/{midfolder}/CRISPRs_alignment_MAG.tsv", sep = '\t', index = False)
            run_command(f"cp {rootpth}/{midfolder}/CRISPRs_alignment_MAG.tsv {rootpth}/{out_dir}/{supplementary}/CRISPRs_alignment_MAG.tsv")
            best_hit = crispr_df.drop_duplicates(subset='qseqid', keep='first')
            crispr_pred_mag = {row['qseqid']: {'pred': row['sseqid'].split('_CRISPR_')[0], 'ident': round(row['pident']/100, 2)} for index, row in best_hit.iterrows()}
        else:
            logger.info(f"run CRISPRs match results...")
            for bfile in os.listdir(f'{bfolder}'):
                prefix = bfile.rsplit('.', 1)[0]
                outputfile = prefix + '.crispr'
                crispr_rec = []
                cnt = 0
                if not os.path.exists(f'{rootpth}/{midfolder}/crispr_tmp/{outputfile}'):
                    logger.info(f"CRISPR file not found for MAG: {prefix}.")
                    continue
                with open(f'{rootpth}/{midfolder}/crispr_tmp/{outputfile}') as file_in:
                    for line in file_in:
                        tmp_list = line.split("\t")
                        try:
                            _ = int(tmp_list[0])
                        except:
                            continue
                        if tmp_list[3] == '\n':
                            continue
                        if not is_valid_dna_sequence(tmp_list[3]):
                            logger.info(f"Invalid DNA sequence found in MAG: {prefix}.")
                            logger.info(f'Please check the format of the MAGs.')
                            continue
                        rec = SeqRecord(Seq(tmp_list[3]), id=f'{prefix}_CRISPR_{cnt}', description='')
                        cnt += 1
                        crispr_rec.append(rec)
                        total_rec.append(rec)
                                
                if crispr_rec:
                    SeqIO.write(crispr_rec, f"{rootpth}/{midfolder}/crispr_fa/{prefix}_CRISPRs.fa", 'fasta')
                
            SeqIO.write(total_rec, f"{rootpth}/{midfolder}/CRISPRs.fa", 'fasta')
            # if CRISPRs found in the MAGs
            if total_rec:
                run_command(f"cp {rootpth}/{midfolder}/CRISPRs.fa {rootpth}/{out_dir}/{supplementary}/CRISPRs_MAGs.fa")
                check_path(os.path.join(rootpth, midfolder, 'crispr_db'))
                run_command(f'makeblastdb -in {rootpth}/{midfolder}/CRISPRs.fa -dbtype nucl -parse_seqids -out {rootpth}/{midfolder}/crispr_db/magCRISPRs')

                if blast == 'blastn-short':
                    run_command(f'blastn -task blastn-short -query {rootpth}/filtered_contigs.fa -db {rootpth}/{midfolder}/crispr_db/magCRISPRs -out {rootpth}/{midfolder}/crispr_mag.tab -outfmt "6 qseqid sseqid evalue pident length slen" -evalue 1 -gapopen 10 -penalty -1 -gapextend 2 -word_size 7 -dust no -max_target_seqs 25 -perc_identity 90 -num_threads {threads}')
                elif blast == 'blastn':
                    run_command(f'blastn -task blastn -query {rootpth}/filtered_contigs.fa -db {rootpth}/{midfolder}/crispr_db/magCRISPRs -out {rootpth}/{midfolder}/crispr_mag.tab -outfmt "6 qseqid sseqid evalue pident length slen" -evalue 1 -max_target_seqs 25 -perc_identity 90 -num_threads {threads}')
                # read the CRISPRs alignment results
                crispr_df = pd.read_csv(f"{rootpth}/{midfolder}/crispr_mag.tab", sep='\t', names = ['qseqid', 'sseqid', 'evalue', 'pident', 'length', 'slen'])
                crispr_df['cov'] = crispr_df['length']/crispr_df['slen']
                crispr_df = crispr_df[(crispr_df['cov'] > cov) & (crispr_df['pident'] > pident)]
                crispr_df.to_csv(f"{rootpth}/{midfolder}/CRISPRs_alignment_MAG.tsv", sep = '\t', index = False)
                run_command(f"cp {rootpth}/{midfolder}/CRISPRs_alignment_MAG.tsv {rootpth}/{out_dir}/{supplementary}/CRISPRs_alignment_MAG.tsv")
                best_hit = crispr_df.drop_duplicates(subset='qseqid', keep='first')
                crispr_pred_mag = {row['qseqid']: {'pred': row['sseqid'].split('_CRISPR_')[0], 'ident': round(row['pident']/100, 2)} for index, row in best_hit.iterrows()}
            
        pkl.dump(crispr_pred_mag, open(f'{rootpth}/{midfolder}/crispr_pred_mag.dict', 'wb'))

        # BLASTN prophage sequences
        logger.info(f"[3/{jy}] predicting MAG link...")
        if os.path.exists(f"{rootpth}/{midfolder}/blastn_MAGs.tsv"):
            logger.info(f"reusing existing prophage results...")
            run_command(f"cp {rootpth}/{midfolder}/blastn_MAGs.tsv {rootpth}/{out_dir}/{supplementary}/BLASTN_alignment_MAG.tsv")
            blast_df = pd.read_csv(f"{rootpth}/{midfolder}/blastn_MAGs.tsv", sep = '\t')
            blast_df.drop_duplicates(subset='qseqid', keep='first', inplace=True)
            blast_pred_mag = {row['qseqid']: {'pred': row['sseqid'], 'ident': round(row['pident']/100, 2)} for index, row in blast_df.iterrows()}
        else:
            logger.info(f"running prophage results...")
            check_path(f"{rootpth}/{midfolder}/blast_db")
            check_path(f"{rootpth}/{midfolder}/blast_out")
            df_list = []
            for bfile in os.listdir(f'{bfolder}'):
                prefix = bfile.rsplit('.', 1)[0]
                run_command(f"makeblastdb -in {bfolder}/{bfile} -dbtype nucl -parse_seqids -out {rootpth}/{midfolder}/blast_db/{prefix}")
                run_command(f"blastn -query {rootpth}/filtered_contigs.fa -db {rootpth}/{midfolder}/blast_db/{prefix} -outfmt '6 qseqid sseqid evalue pident length' -out {rootpth}/{midfolder}/blast_out/{prefix}.blastn -evalue 1 -max_target_seqs 1 -perc_identity 98 -num_threads {threads}")
                blast_df = pd.read_csv(f"{rootpth}/{midfolder}/blast_out/{prefix}.blastn", sep='\t', names = ['qseqid', 'sseqid', 'evalue', 'pident', 'length'])
                blast_df = blast_df[blast_df['length'] >= prolen]
                blast_df['sseqid'] = prefix
                blast_df.sort_values('length', ascending=False, inplace=True)
                blast_df.drop_duplicates(subset='qseqid', keep='first', inplace=True)
                df_list.append(blast_df)
            blast_df = pd.concat(df_list)
            blast_df.sort_values('length', ascending=False, inplace=True)
            blast_df.to_csv(f"{rootpth}/{midfolder}/blastn_MAGs.tsv", sep = '\t', index = False)
            run_command(f"cp {rootpth}/{midfolder}/blastn_MAGs.tsv {rootpth}/{out_dir}/{supplementary}/BLASTN_alignment_MAG.tsv")
            blast_df.drop_duplicates(subset='qseqid', keep='first', inplace=True)
            blast_pred_mag = {row['qseqid']: {'pred': row['sseqid'], 'ident': round(row['pident']/100, 2)} for index, row in blast_df.iterrows()}
        
        # tRNA prediction
        if os.path.exists(f"{rootpth}/{midfolder}/tRNA_MAGs.tsv"):
            logger.info(f"reusing existing tRNA results...")
            run_command(f"cp {rootpth}/{midfolder}/tRNA_MAGs.tsv {rootpth}/{out_dir}/{supplementary}/tRNA_MAGs.tsv")
            tRNA_df = pd.read_csv(f"{rootpth}/{midfolder}/tRNA_MAGs.tsv", sep = '\t')
            tRNA_df.drop_duplicates(subset=['virus', 'mag'], inplace=True)
            tRNA_pred_mag = {row['virus']: {'pred': row['mag'], 'ident': round(row['pident']/100, 2)} for index, row in tRNA_df.iterrows()}
        else:
            logger.info(f"running tRNA results...")
            check_path(f"{rootpth}/{midfolder}/tRNA")
            check_path(f"{rootpth}/{midfolder}/tRNA_db")
            for bfile in os.listdir(f'{bfolder}'):
                prefix = bfile.rsplit('.', 1)[0]
                run_command(f'aragorn -t -gcbact -o {rootpth}/{midfolder}/tRNA/{prefix}.fa -fo {bfolder}/{bfile}')
            tRNA_rec = []
            for file in os.listdir(f'{rootpth}/{midfolder}/tRNA'):
                cnt = 0
                for record in SeqIO.parse(f'{rootpth}/{midfolder}/tRNA/{file}', 'fasta'):
                    record.id = f'{file.rsplit(".", 1)[0]}_{cnt}'
                    record.description = ''
                    tRNA_rec.append(record)
                    cnt += 1
            # write all tRNA to a single fasta file
            _ = SeqIO.write(tRNA_rec, f'{rootpth}/{midfolder}/tRNA.fa', 'fasta')
            if tRNA_rec:
                run_command(f'makeblastdb -in {rootpth}/{midfolder}/tRNA.fa -dbtype nucl -parse_seqids -out {rootpth}/{midfolder}/tRNA_db/tRNAs')
                run_command(f"blastn -query {rootpth}/filtered_contigs.fa -db {rootpth}/{midfolder}/tRNA_db/tRNAs -outfmt '6 qseqid sseqid evalue pident length slen' -out {rootpth}/{midfolder}/blastn_tRNA.tsv -evalue 1 -max_target_seqs 25 -perc_identity 90 -num_threads {threads}")
                tRNA_df = pd.read_csv(f'{rootpth}/{midfolder}/blastn_tRNA.tsv', sep='\t', header=None)
                tRNA_df.columns = ['virus', 'sseqid', 'evalue', 'pident', 'length', 'slen']
                tRNA_df['coverage'] = tRNA_df['length'] / tRNA_df['slen']
                tRNA_df.loc[tRNA_df['coverage'] > 1, 'coverage'] = 1
                tRNA_df = tRNA_df[tRNA_df['coverage'] >= 0.95]
                tRNA_df['mag'] = tRNA_df['sseqid'].apply(lambda x: x.rsplit('_', 1)[0])
                tRNA_df = tRNA_df[['virus', 'mag', 'evalue', 'pident', 'coverage']]
                tRNA_df.sort_values('pident', ascending=False, inplace=True)
                tRNA_df.to_csv(f'{rootpth}/{midfolder}/tRNA_MAGs.tsv', sep='\t', index=False)
                run_command(f"cp {rootpth}/{midfolder}/tRNA_MAGs.tsv {rootpth}/{out_dir}/{supplementary}/tRNA_MAGs.tsv")
                tRNA_df.drop_duplicates(subset=['virus', 'mag'], inplace=True)
                tRNA_pred_mag = {row['virus']: {'pred': row['mag'], 'ident': round(row['pident']/100, 2)} for index, row in tRNA_df.iterrows()}

        # Kmer prediction
        getPKmer(f'{rootpth}/filtered_contigs.fa', f'{rootpth}/{midfolder}/pkmer', threads)
        getBKmer(f'{bfolder}', f'{rootpth}/{midfolder}/bkmer', threads)
        virus2host_kmer = predictVirusHost(db_dir, f'{rootpth}/{midfolder}/', genomes)
        pkl.dump(virus2host_kmer, open(f'{rootpth}/{midfolder}/virus2host_kmer.pkl', 'wb'))

        # align virus protien against MAGs protein
        ## tranlate the contigs/MAGs to protein sequences
        if os.path.exists(f'{rootpth}/{midfolder}/query_protein.fa'):
            logger.info(f"reusing existing protein file...")
        else:
            logger.info(f"calling genes with prodigal...")
            parallel_prodigal_gv(f'{rootpth}/filtered_contigs.fa', f'{rootpth}/{midfolder}/query_protein.fa', threads)
        
        check_path(os.path.join(rootpth, midfolder, 'bfolder_protein'))
        check_path(os.path.join(rootpth, midfolder, 'bfolder_protein_db'))
        check_path(os.path.join(rootpth, midfolder, 'bfolder_protein_align'))
        Flag, _ = translate_MAGs(f'{bfolder}', f'{rootpth}/{midfolder}/bfolder_protein/', threads)
        assert Flag == True, "Error in translating MAGs to protein sequences, please check the MAGs."

        if os.path.exists(f'{rootpth}/{midfolder}/bfolder_protein_align.tsv'):
            logger.info(f"reusing existing protein alignment file...")
            align = pd.read_csv(f'{rootpth}/{midfolder}/bfolder_protein_align.tsv', sep='\t')
        else:
            # run diamond blastp
            for file in os.listdir(f'{rootpth}/{midfolder}/bfolder_protein/'):
                name = file.rsplit('.', 1)[0]
                _ = os.system(f'diamond makedb --in {rootpth}/{midfolder}/bfolder_protein/{file} -d {rootpth}/{midfolder}/bfolder_protein_db/{name} --quiet')


            for db in os.listdir(f'{rootpth}/{midfolder}/bfolder_protein_db/'):
                db_name = db.split('.dmnd')[0]
                _ = os.system(f'diamond blastp -d {rootpth}/{midfolder}/bfolder_protein_db/{db} -q {rootpth}/{midfolder}/query_protein.fa -o {rootpth}/{midfolder}/bfolder_protein_align/{db_name}.tsv --threads {threads}  --query-cover 50 --subject-cover 50 --quiet')

            df_list = []
            for file in os.listdir(f'{rootpth}/{midfolder}/bfolder_protein_align/'):
                try:
                    df = pd.read_csv(f'{rootpth}/{midfolder}/bfolder_protein_align/{file}', sep='\t', header=None)
                except pd.errors.EmptyDataError:
                    continue
                df.columns = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore']
                df['MAG'] = file.rsplit('.', 1)[0]
                df = df[['qseqid', 'MAG', 'pident']]
                df_list.append(df)

            align = pd.concat(df_list)
            align['phage'] = align['qseqid'].apply(lambda x: x.rsplit('_', 1)[0])
            align = align.groupby(['phage', 'MAG'], as_index=False)['pident'].max()
            align.to_csv(f'{rootpth}/{midfolder}/bfolder_protein_align.tsv', sep='\t', index=False)
        align = align[align['phage'].isin(set(virus2host_kmer.keys()))]

        phage2mag2pident = {phage: {} for phage in virus2host_kmer.keys()}
        phage2mean = {phage: [] for phage in virus2host_kmer.keys()}
        for index, row in align.iterrows():
            phage = row['phage']
            mag = row['MAG']
            pident = row['pident']/100
            phage2mag2pident[phage][mag] = pident
            phage2mean[phage].append(pident)

        for phage in phage2mean.keys():
            if phage2mean[phage]:
                phage2mean[phage] = sum(phage2mean[phage]) / len(phage2mean[phage])
            else:
                phage2mean[phage] = 0



        

        if bgtdb != 'None':
            gtdb_df = pd.read_csv(f'{bgtdb}', sep='\t')
            # check if all mag in the gtdb file
            mag_set = set([item.rsplit('.', 1)[0] for item in os.listdir(f'{bfolder}')])
            assert (mag_set - set(gtdb_df['user_genome'].unique())) == set(), "Seems not all MAGs has annotation in the gtdb file, please check\nPlease also ensure there is no suffix like .fa in the user_genome in your gtdb tsv file\nOr please do not use the --bgtdb option"

            gtdb_df['Species'] = gtdb_df['classification'].apply(lambda x: "species:" + x.split('s__')[-1].split(';')[0] if (isinstance(x, str) and x.split('s__')[-1].split(';')[0] != '') else 'NaN')
            gtdb_df['Genus'] = gtdb_df['classification'].apply(lambda x: "genus:" + x.split('g__')[-1].split(';')[0] if (isinstance(x, str) and x.split('g__')[-1].split(';')[0] != '') else 'NaN')
            gtdb_df['Family'] = gtdb_df['classification'].apply(lambda x: "family:" +x.split('f__')[-1].split(';')[0] if (isinstance(x, str) and x.split('f__')[-1].split(';')[0] != '') else 'NaN')

            mag2taxonomy = {row['user_genome']: row['classification'] for _, row in gtdb_df.iterrows()}
            mag2species = {row['user_genome']: row['Species'] for _, row in gtdb_df.iterrows()}
            mag2genus = {row['user_genome']: row['Genus'] for _, row in gtdb_df.iterrows()}
            mag2family = {row['user_genome']: row['Family'] for _, row in gtdb_df.iterrows()}
            species2linage = {row['Species']: row['classification'] for _, row in gtdb_df.iterrows()}
            genus2linage = {row['Genus']: row['classification'].split(';s__')[0] for _, row in gtdb_df.iterrows()}
            family2linage = {row['Family']: row['classification'].split(';g__')[0] for _, row in gtdb_df.iterrows()}
        
        # initialize a dictionary to store candidate predictions for each phage
        align2pred = {}
        for phage, mag in zip(align['phage'], align['MAG']):
            try:
                align2pred[phage]['Exact'].append(mag)
            except KeyError:
                align2pred[phage] = {'Exact': [mag]}

        if bgtdb != 'None':
            for phage, mag in zip(align['phage'], align['MAG']):
                if mag2species[mag] != 'NaN':
                    align2pred[phage]['Species'] = mag2species[mag]
                if mag2genus[mag] != 'NaN':
                    align2pred[phage]['Genus'] = mag2genus[mag]
                if mag2family[mag] != 'NaN':
                    align2pred[phage]['Family'] = mag2family[mag]


        # reorganize the predictions based on the kmer results
        if bgtdb != 'None':
            for phage, mag in virus2host_kmer.items():
                if phage not in align2pred:
                    continue
                if mag in align2pred[phage]['Exact']:
                    virus2host_kmer_reorg[phage] = {}
                    virus2host_kmer_reorg[phage]['pred'] = mag
                    virus2host_kmer_reorg[phage]['ident'] = round(phage2mag2pident[phage][mag], 2)
                elif mag2species[mag] in align2pred[phage]['Species']:
                    virus2host_kmer_reorg[phage] = {}
                    virus2host_kmer_reorg[phage]['pred'] = mag2species[mag]
                    virus2host_kmer_reorg[phage]['ident'] = round(phage2mean[phage], 2)
                elif mag2genus[mag] in align2pred[phage]['Genus']:
                    virus2host_kmer_reorg[phage] = {}
                    virus2host_kmer_reorg[phage]['pred'] = mag2genus[mag]
                    virus2host_kmer_reorg[phage]['ident'] = round(phage2mean[phage], 2)
                elif mag2family[mag] in align2pred[phage]['Family']:
                    virus2host_kmer_reorg[phage] = {}
                    virus2host_kmer_reorg[phage]['pred'] = mag2family[mag]
                    virus2host_kmer_reorg[phage]['ident'] = round(phage2mean[phage], 2)
        else:
            for phage, mag in virus2host_kmer.items():
                if phage not in align2pred:
                    continue
                if mag in align2pred[phage]['Exact']:
                    virus2host_kmer_reorg[phage] = {}
                    virus2host_kmer_reorg[phage]['pred'] = mag
                    virus2host_kmer_reorg[phage]['ident'] = round(phage2mag2pident[phage][mag], 2)

                

        pkl.dump(virus2host_kmer_reorg, open(f'{rootpth}/{midfolder}/virus2host_kmer_reorg.pkl', 'wb'))
        

        if not crispr_pred_mag and not blast_pred_mag and not virus2host_kmer_reorg and not tRNA_pred_mag:
            logger.info('No CRISPRs/BLASTN/tRNA/Kmer links found in the MAGs.')
            if magonly == 'Y':
                logger.info('Please check the MAGs or set the magonly flag to F.')
                exit(1)
        else:
            crispr_set = set(crispr_pred_mag.keys())
            blast_set = set(blast_pred_mag.keys())
            tRNA_set = set(tRNA_pred_mag.keys())
            kmer_set = set(virus2host_kmer_reorg.keys())

            blast_set = blast_set - crispr_set
            tRNA_set = tRNA_set - crispr_set - blast_set
            kmer_set = kmer_set - crispr_set - blast_set - tRNA_set

            blast_set = list(blast_set)
            tRNA_set = list(tRNA_set)
            kmer_set = list(kmer_set)
                            

            unpredicted = []
            filtered_contig = []
            filtered_lenth = []
            unpredicted_contig = []
            unpredicted_length = []
            for record in SeqIO.parse(f'{contigs}', 'fasta'):
                try:
                    _ = crispr_pred_mag[record.id]
                except:
                    try:
                        _ = blast_pred_mag[record.id]
                    except:
                        try:
                            _ = tRNA_pred_mag[record.id]
                        except:
                            try:
                                _ = virus2host_kmer_reorg[record.id]
                            except:
                                if len(record.seq) < inputs.len:
                                    filtered_contig.append(record.id)
                                    filtered_lenth.append(len(record.seq))
                                unpredicted.append(record)
                                unpredicted_contig.append(record.id)
                                unpredicted_length.append(len(record.seq))
        
            if magonly == 'Y' or not unpredicted:
                logger.info(f"[4/{jy}] writing the results...")
                # Create lists by combining existing data with new entries
                all_contigs = list(crispr_pred_mag.keys())                               + blast_set                                                  + tRNA_set                                                    + kmer_set                                                    + filtered_contig                     + unpredicted_contig
                all_pred = [crispr_pred_mag[item]['pred'] for item in crispr_pred_mag]   + [blast_pred_mag[item]['pred'] for item in blast_set]       + [tRNA_pred_mag[item]['pred'] for item in tRNA_set]          + [virus2host_kmer_reorg[item]['pred'] for item in kmer_set]  + ['filtered'] * len(filtered_contig) + ['-'] * len(unpredicted_contig)
                all_score = [crispr_pred_mag[item]['ident'] for item in crispr_pred_mag] + [blast_pred_mag[item]['ident'] for item in blast_set]      + [tRNA_pred_mag[item]['ident'] for item in tRNA_set]         + [virus2host_kmer_reorg[item]['ident'] for item in kmer_set] + [0] * len(filtered_contig)          + [0] * len(unpredicted_contig)
                all_length = [genomes[item].length for item in crispr_pred_mag]          + [genomes[item].length for item in blast_set]               + [genomes[item].length for item in tRNA_set]                 + [genomes[item].length for item in kmer_set]                 + filtered_lenth                      + unpredicted_length
                all_method = ['CIRPSR-based (MAG)']*len(crispr_pred_mag)                 + ['BLASTN-based (MAG)']*len(blast_set)                      + ['tRNA-based (MAG)']*len(tRNA_set)                          + ['Kmer-based (MAG)']*len(kmer_set)                          + ['-'] * len(filtered_contig)        + ['-'] * len(unpredicted_contig)

                if bgtdb != 'None':
                    all_linage = []
                    for item in all_pred:
                        try:
                            all_linage.append(mag2taxonomy[item])
                        except:
                            try:
                                all_linage.append(species2linage[item])
                            except:
                                try:
                                    all_linage.append(genus2linage[item])
                                except:
                                    try:
                                        all_linage.append(family2linage[item])
                                    except:
                                        all_linage.append('-')

                    contig_to_pred = pd.DataFrame({
                        'Accession': all_contigs,
                        'Length': all_length,
                        'Host': all_pred,
                        'CHERRYScore': all_score,
                        'Method': all_method,
                        'Host_GTDB_lineage': all_linage
                    })
               
                else:
                    contig_to_pred = pd.DataFrame({
                        'Accession': all_contigs,
                        'Length': all_length,
                        'Host': all_pred,
                        'CHERRYScore': all_score,
                        'Method': all_method
                    })
                contig_to_pred.to_csv(f"{rootpth}/{out_dir}/cherry_prediction.tsv", index=False, sep='\t')


                if blast_pred_mag != {}:
                    blast_df = pd.read_csv(f'{rootpth}/{midfolder}/blastn_MAGs.tsv', sep='\t')
                    blast_df.rename(columns={'qseqid': 'Accession', 'sseqid': 'Host', 'pident': 'CHERRYScore'}, inplace=True)
                    blast_df['Method'] = 'BLASTN-based (MAG)'
                    blast_df['Length'] = blast_df['Accession'].apply(lambda x: genomes[x].length if x in genomes else '-')
                    if bgtdb != 'None':
                        blast_df['Host_GTDB_lineage'] = blast_df['Host'].apply(lambda x: mag2taxonomy[x] if x in mag2taxonomy else '-')
                    else:
                        blast_df['Host_GTDB_lineage'] = '-'
                    blast_df = blast_df[['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_GTDB_lineage']]
                else:
                    blast_df = pd.DataFrame(columns=['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_GTDB_lineage'])
                if crispr_pred_mag != {}:    
                    crispr_df = pd.read_csv(f'{rootpth}/{midfolder}/CRISPRs_alignment_MAG.tsv', sep='\t')
                    crispr_df.rename(columns={'qseqid': 'Accession', 'sseqid': 'Host', 'pident': 'CHERRYScore'}, inplace=True)
                    crispr_df['Method'] = 'CRISPR-based (MAG)'
                    crispr_df['Host'] = crispr_df['Host'].apply(lambda x: x.split('_CRISPR_')[0])
                    crispr_df['Length'] = crispr_df['Accession'].apply(lambda x: genomes[x].length if x in genomes else '-')
                    if bgtdb != 'None':
                        crispr_df['Host_GTDB_lineage'] = crispr_df['Host'].apply(lambda x: mag2taxonomy[x] if x in mag2taxonomy else '-')
                    else:
                        crispr_df['Host_GTDB_lineage'] = '-'
                else:
                    crispr_df = pd.DataFrame(columns=['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_GTDB_lineage'])
                if tRNA_pred_mag != {}:
                    tRNA_df = pd.read_csv(f'{rootpth}/{midfolder}/tRNA_MAGs.tsv', sep='\t')
                    tRNA_df.rename(columns={'virus': 'Accession', 'mag': 'Host', 'pident': 'CHERRYScore'}, inplace=True)
                    tRNA_df['Method'] = 'tRNA-based (MAG)'
                    tRNA_df['Length'] = tRNA_df['Accession'].apply(lambda x: genomes[x].length if x in genomes else '-')
                    if bgtdb != 'None':
                        tRNA_df['Host_GTDB_lineage'] = tRNA_df['Host'].apply(lambda x: mag2taxonomy[x] if x in mag2taxonomy else '-')
                    else:
                        tRNA_df['Host_GTDB_lineage'] = '-'
                    tRNA_df = tRNA_df[['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_GTDB_lineage']]
                else:
                    tRNA_df = pd.DataFrame(columns=['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_GTDB_lineage'])
                if virus2host_kmer_reorg != {}:
                    virus2host = pkl.load(open(f'{rootpth}/{midfolder}/virus2host_kmer_reorg.pkl', 'rb'))
                    virus = []
                    mag = []
                    score = []
                    for v, pred in virus2host.items():
                        h = pred['pred']
                        s = pred['ident']
                        virus.append(v)
                        mag.append(h)
                        score.append(s)
                    kmer_df = pd.DataFrame({'Accession': virus, 'Host': mag, 'CHERRYScore': score})
                    kmer_df['Method'] = 'Kmer-based (MAG)'
                    kmer_df['Length'] = kmer_df['Accession'].apply(lambda x: genomes[x].length if x in genomes else '-')
                    if bgtdb != 'None':
                        kmer_df['Host_GTDB_lineage'] = kmer_df['Host'].apply(lambda x: mag2taxonomy[x] if x in mag2taxonomy else '-')
                    else:
                        kmer_df['Host_GTDB_lineage'] = '-'
                    kmer_df = kmer_df[['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_GTDB_lineage']]
                else:
                    kmer_df = pd.DataFrame(columns=['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_GTDB_lineage'])
                multi_df = pd.concat([blast_df, crispr_df, tRNA_df, kmer_df, contig_to_pred])
                multi_df.drop_duplicates(subset=['Accession', 'Host', 'Method'], keep='first', inplace=True)
                multi_df = multi_df.sort_values(['Accession', 'CHERRYScore'], ascending=[True, False])
                multi_df.to_csv(f"{rootpth}/{out_dir}/cherry_multi_prediction.tsv", index=False, sep='\t')

                logger.info("Run time: %s seconds\n" % round(time.time() - program_start, 2))
                return()
    else:
        logger.info(f"[2/{jy}] no MAGs provided, skipping MAG link prediction...")
        logger.info(f"[3/{jy}] will continue running CHERRY with database version...")
    

    ###############################################################
    ######################## CRISPRs DB  ##########################
    ###############################################################
    logger.info(f"[4/{jy}] predicting DB CRISPRs...")

    if blast == 'blastn-short':
        run_command(f'blastn -task blastn-short -query {rootpth}/filtered_contigs.fa -db {db_dir}/crispr_db/allCRISPRs -out {rootpth}/{midfolder}/crispr_out.tab -outfmt "6 qseqid sseqid evalue pident length slen" -evalue 1 -gapopen 10 -penalty -1 -gapextend 2 -word_size 7 -dust no -max_target_seqs 25 -perc_identity 90 -num_threads {threads}')
    elif blast == 'blastn':
        run_command(f'blastn -task blastn -query {rootpth}/filtered_contigs.fa -db {db_dir}/crispr_db/allCRISPRs -out {rootpth}/{midfolder}/crispr_out.tab -outfmt "6 qseqid sseqid evalue pident length slen" -evalue 1 -max_target_seqs 25 -perc_identity 90 -num_threads {threads}')


    crispr_df = pd.read_csv(f"{rootpth}/{midfolder}/crispr_out.tab", sep='\t', names = ['qseqid', 'sseqid', 'evalue', 'pident', 'length', 'slen'])
    crispr_df['cov'] = crispr_df['length']/crispr_df['slen']
    crispr_df = crispr_df[(crispr_df['cov'] > cov) & (crispr_df['pident'] > pident)]
    crispr_df.to_csv(f"{rootpth}/{midfolder}/CRISPRs_alignment_DB.tsv", sep = '\t', index = False)
    run_command(f"cp {rootpth}/{midfolder}/CRISPRs_alignment_DB.tsv {rootpth}/{out_dir}/{supplementary}/CRISPRs_alignment_DB.tsv")
    best_hit = crispr_df.drop_duplicates(subset='qseqid', keep='first')
    #crispr_pred_mag = {row['qseqid']: {'pred': row['sseqid'].split('_CRISPR_')[0], 'ident': round(row['pident']/100, 2)} for index, row in best_hit.iterrows()}
    crispr_pred_db = {}
    for index, row in best_hit.iterrows():
        for item in row['sseqid'].split('|'):
            if 'GC' in item:
                prokaryote = item.split('.')[0]
                crispr_pred_db[row['qseqid']] = {'pred': prokaryote, 'ident': round(row['pident']/100, 2)}
                break
    pkl.dump(crispr_pred_db, open(f'{rootpth}/{midfolder}/crispr_pred_db.dict', 'wb'))


    

    if os.path.exists(f'{inputs.proteins}'):
        logger.info(f"[5/{jy}] using provided protein file...")
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
        logger.info(f"[5/{jy}] reusing existing protein file...")
    else:
        logger.info(f"[5/{jy}] calling genes with prodigal...")
        parallel_prodigal_gv(f'{rootpth}/filtered_contigs.fa', f'{rootpth}/{midfolder}/query_protein.fa', threads)
    
    if os.path.exists(f'{rootpth}/{midfolder}/self_results.abc') and os.path.exists(f'{rootpth}/{midfolder}/db_results.abc'):
        logger.info(f"[6/{jy}] reusing all-against-all alignment from PhaGCN...")
    else:
        logger.info(f"[6/{jy}] running all-against-all alignment...")
        # combine the database with the predicted proteins
        run_command(f"cat {db_dir}/RefVirus.faa {rootpth}/{midfolder}/query_protein.fa > {rootpth}/{midfolder}/ALLprotein.fa")
        # generate the diamond database
        run_command(f"diamond makedb --in {rootpth}/{midfolder}/query_protein.fa -d {rootpth}/{midfolder}/query_protein.dmnd --threads {threads} --quiet")
        # align to the database
        run_command(f"diamond blastp --db {db_dir}/RefVirus.dmnd --query {rootpth}/{midfolder}/query_protein.fa --out {rootpth}/{midfolder}/db_results.tab --outfmt 6 --threads {threads} --evalue 1e-5 --max-target-seqs 10000 --query-cover 50 --subject-cover 50 --quiet")
        run_command(f"awk '{{print $1,$2,$3,$12}}' {rootpth}/{midfolder}/db_results.tab > {rootpth}/{midfolder}/db_results.abc")
        # align to itself
        run_command(f"diamond blastp --db {rootpth}/{midfolder}/query_protein.dmnd --query {rootpth}/{midfolder}/query_protein.fa --out {rootpth}/{midfolder}/self_results.tab --outfmt 6 --threads {threads} --evalue 1e-5 --max-target-seqs 10000 --query-cover 50 --subject-cover 50 --quiet")
        run_command(f"awk '{{print $1,$2,$3,$12}}' {rootpth}/{midfolder}/self_results.tab > {rootpth}/{midfolder}/self_results.abc")


    logger.info(f"[7/{jy}] generating cherry networks...")
    # FLAGS: no proteins aligned to the database
    if os.path.getsize(f'{rootpth}/{midfolder}/db_results.abc') == 0:
        if len(crispr_pred_db) == 0 and len(crispr_pred_mag) == 0 and len(blast_pred_mag) == 0 and len(tRNA_pred_mag) == 0 and len(virus2host_kmer_reorg) == 0:
            Accession = []
            Length_list = []
            for record in SeqIO.parse(f'{contigs}', 'fasta'):
                Accession.append(record.id)
                Length_list.append(len(record.seq))
            contig_to_pred = pd.DataFrame({"Accession": Accession, "Length":Length_list, "Host":['-']*len(Accession), "CHERRYScore":[0]*len(Accession), "Method":['-']*len(Accession)})
            contig_to_pred.to_csv(f"{rootpth}/{out_dir}/cherry_prediction.tsv", index = None, sep='\t')
            exit()
        else:
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
                    Pred.append(crispr_pred_mag[record.id]['pred'])
                    Score.append(crispr_pred_mag[record.id]['ident'])
                    Method.append('CRISPR-based (MAG)')
                except:
                    try:
                        Pred.append(blast_pred_mag[record.id]['pred'])
                        Score.append(blast_pred_mag[record.id]['ident'])
                        Method.append('BLASTN-based (MAG)')
                    except:
                        try:
                            Pred.append(tRNA_pred_mag[record.id]['pred'])
                            Score.append(tRNA_pred_mag[record.id]['ident'])
                            Method.append('tRNA-based (MAG)')
                        except:
                            try:
                                Pred.append(virus2host_kmer_reorg[record.id]['pred'])
                                Score.append(virus2host_kmer_reorg[record.id]['ident'])
                                Method.append('Kmer-based (MAG)')
                            except:
                                try:
                                    Pred.append(CRISPRs_acc2host[crispr_pred_db[record.id]['pred']])
                                    Score.append(crispr_pred_db[record.id]['ident'])
                                    Method.append('CRISPR-based (DB)')
                                except:
                                    Pred.append('-')
                                    Score.append(0.0)
                                    Method.append('-')

            if bgtdb != 'None':
                all_linage = []
                for item in Pred:
                    try:
                        all_linage.append(mag2taxonomy[item])
                    except:
                        try:
                            all_linage.append(species2linage[item])
                        except:
                            try:
                                all_linage.append(genus2linage[item])
                            except:
                                try:
                                    all_linage.append(family2linage[item])
                                except:
                                    all_linage.append('-')
                contig_to_pred = pd.DataFrame({"Accession": Accession, "Length": Length_list, "Host":Pred, "CHERRYScore": Score, "Method": Method, 'Host_GTDB_lineage': all_linage})
            else:
                contig_to_pred = pd.DataFrame({"Accession": Accession, "Length": Length_list, "Host":Pred, "CHERRYScore": Score, "Method": Method})
            contig_to_pred.to_csv(f"{rootpth}/{out_dir}/cherry_prediction.tsv", index = None, sep='\t')


            ############################################
            ######### multi-host prediction ############
            ############################################
            if blast_pred_mag != {}:
                blast_df = pd.read_csv(f'{rootpth}/{midfolder}/blastn_MAGs.tsv', sep='\t')
                blast_df.rename(columns={'qseqid': 'Accession', 'sseqid': 'Host', 'pident': 'CHERRYScore'}, inplace=True)
                blast_df['Method'] = 'BLASTN-based (MAG)'
                blast_df['Length'] = blast_df['Accession'].apply(lambda x: genomes[x].length if x in genomes else '-')
                if bgtdb != 'None':
                    blast_df['Host_GTDB_lineage'] = blast_df['Host'].apply(lambda x: mag2taxonomy[x] if x in mag2taxonomy else '-')
                else:
                    blast_df['Host_GTDB_lineage'] = '-'
                blast_df = blast_df[['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_GTDB_lineage']]
            else:
                blast_df = pd.DataFrame(columns=['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_GTDB_lineage'])
            if crispr_pred_mag != {}:    
                crispr_df = pd.read_csv(f'{rootpth}/{midfolder}/CRISPRs_alignment_MAG.tsv', sep='\t')
                crispr_df.rename(columns={'qseqid': 'Accession', 'sseqid': 'Host', 'pident': 'CHERRYScore'}, inplace=True)
                crispr_df['Method'] = 'CRISPR-based (MAG)'
                crispr_df['Host'] = crispr_df['Host'].apply(lambda x: x.split('_CRISPR_')[0])
                crispr_df['Length'] = crispr_df['Accession'].apply(lambda x: genomes[x].length if x in genomes else '-')
                if bgtdb != 'None':
                    crispr_df['Host_GTDB_lineage'] = crispr_df['Host'].apply(lambda x: mag2taxonomy[x] if x in mag2taxonomy else '-')
                else:
                    crispr_df['Host_GTDB_lineage'] = '-'
            else:
                crispr_df = pd.DataFrame(columns=['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_GTDB_lineage'])
            if tRNA_pred_mag != {}:
                tRNA_df = pd.read_csv(f'{rootpth}/{midfolder}/tRNA_MAGs.tsv', sep='\t')
                tRNA_df.rename(columns={'virus': 'Accession', 'mag': 'Host', 'pident': 'CHERRYScore'}, inplace=True)
                tRNA_df['Method'] = 'tRNA-based (MAG)'
                tRNA_df['Length'] = tRNA_df['Accession'].apply(lambda x: genomes[x].length if x in genomes else '-')
                if bgtdb != 'None':
                    tRNA_df['Host_GTDB_lineage'] = tRNA_df['Host'].apply(lambda x: mag2taxonomy[x] if x in mag2taxonomy else '-')
                else:
                    tRNA_df['Host_GTDB_lineage'] = '-'
                tRNA_df = tRNA_df[['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_GTDB_lineage']]
            else:
                tRNA_df = pd.DataFrame(columns=['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_GTDB_lineage'])
            if virus2host_kmer_reorg != {}:
                virus2host = pkl.load(open(f'{rootpth}/{midfolder}/virus2host_kmer_reorg.pkl', 'rb'))
                virus = []
                mag = []
                score = []
                for v, pred in virus2host.items():
                    h = pred['pred']
                    s = pred['ident']
                    virus.append(v)
                    mag.append(h)
                    score.append(s)
                kmer_df = pd.DataFrame({'Accession': virus, 'Host': mag, 'CHERRYScore': score})
                kmer_df['Method'] = 'Kmer-based (MAG)'
                kmer_df['Length'] = kmer_df['Accession'].apply(lambda x: genomes[x].length if x in genomes else '-')
                if bgtdb != 'None':
                    kmer_df['Host_GTDB_lineage'] = kmer_df['Host'].apply(lambda x: mag2taxonomy[x] if x in mag2taxonomy else '-')
                else:
                    kmer_df['Host_GTDB_lineage'] = '-'
                kmer_df = kmer_df[['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_GTDB_lineage']]
            else:
                kmer_df = pd.DataFrame(columns=['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_GTDB_lineage'])
            multi_df = pd.concat([blast_df, crispr_df, tRNA_df, kmer_df, contig_to_pred])
            multi_df.drop_duplicates(subset=['Accession', 'Host', 'Method'], keep='first', inplace=True)
            multi_df = multi_df.sort_values(['Accession', 'CHERRYScore'], ascending=[True, False])
            multi_df.to_csv(f"{rootpth}/{out_dir}/cherry_multi_prediction.tsv", index=False, sep='\t')
            logger.info("Run time: %s seconds\n" % round(time.time() - program_start, 2))
            exit()

    # add the genome size
    genome_size = defaultdict(int)
    for index, r in enumerate(SeqIO.parse(f'{rootpth}/{midfolder}/ALLprotein.fa', 'fasta')):
        genome_id = r.id.rsplit('_', 1)[0]
        genome_size[genome_id] += 1


    # filter the network
    if os.path.exists(f'{rootpth}/{midfolder}/phagcn_network.tsv'):
        cherry_network = f'phagcn_network.tsv'
        run_command(f'cp {rootpth}/{out_dir}/phagcn_supplementary/phagcn_network_edges.tsv {rootpth}/{out_dir}/{supplementary}/cherry_network_edges.tsv')
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
        sub_df.to_csv(f"{rootpth}/{out_dir}/{supplementary}/cherry_network_edges.tsv", index=False, sep='\t')
    
    run_command(f'mcl {rootpth}/{midfolder}/{cherry_network} -te {threads} -I 2.0 --abc -o {rootpth}/{midfolder}/cherry_genus_clusters.txt > /dev/null 2>&1')


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

    df.to_csv(f'{rootpth}/{midfolder}/cherry_clustering.tsv', index=False, sep='\t')

    
    logger.info(f"[8/{jy}] predicting the host...")
    #query_df = pd.read_csv(f'{rootpth}/{midfolder}/cherry_clustering.tsv')
    query_df = df.copy()
    ref_df = pd.read_csv(f'{db_dir}/RefVirus.csv')
    genus2score = {genus: score for genus, score in zip(query_df['Genus'], query_df['Score']) if genus != '-' }
    query_df = query_df[['Accession', 'Lineage', 'Length', 'Genus']]
    ref_df = ref_df[['Accession', 'Lineage', 'Length', 'Genus']]
    cluster_df = pd.concat([query_df, ref_df])
    genus2host = pkl.load(open(f'{db_dir}/genus2host.pkl', 'rb'))
    genus2hostlineage = pkl.load(open(f'{db_dir}/genus2hostlineage.pkl', 'rb'))

    
    cluster_df['Host'] = cluster_df['Genus'].apply(lambda x: genus2host[x] if x in genus2host else '-')
    cluster_df['Host_NCBI'] = cluster_df['Host'].apply(lambda x: genus2hostlineage[x] if x in genus2hostlineage else '-')
    cluster_df['Host_GTDB'] = cluster_df['Host'].apply(lambda x: '-')
    cluster_df['Score'] = 0


    ref_df = cluster_df[cluster_df['Accession'].isin(ref_df['Accession'])]
    refacc2host = {acc:host for acc, host in zip(ref_df['Accession'], ref_df['Host'])}

    #prokaryote_df = pd.read_csv(f'{db_dir}/prokaryote.csv')
    CRISPRs_acc2host = pkl.load(open(f'{db_dir}/CRISPRs_acc2host.pkl', 'rb'))
    hostacc2ncbi = pkl.load(open(f'{db_dir}/hostacc2ncbi.pkl', 'rb'))
    hostacc2gtdb = pkl.load(open(f'{db_dir}/hostacc2gtdb.pkl', 'rb'))
    crispr2host = {}
    crispr2host_NCBI_lineage = {}
    crispr2host_GTDB_lineage = {}
    for acc in crispr_pred_db:
        host = CRISPRs_acc2host[crispr_pred_db[acc]['pred']]
        try:
            ncbi_lineage = hostacc2ncbi[crispr_pred_db[acc]['pred']]
        except:
            ncbi_lineage = 'Not found'
        try:
            gtdb_lineage = hostacc2gtdb[crispr_pred_db[acc]['pred']]
        except:
            gtdb_lineage = 'Not found'
        crispr2host[acc] = host
        crispr2host_NCBI_lineage[acc] = ncbi_lineage
        crispr2host_GTDB_lineage[acc] = gtdb_lineage

    # add crispr information to df['Crispr']
    try:
        crispr_pred_mag = pkl.load(open(f'{rootpth}/{midfolder}/crispr_pred_mag.dict', 'rb'))
    except:
        crispr_pred_mag = {}
    cluster_df['Crispr_db'] = cluster_df['Accession'].apply(lambda x: crispr2host[x] if x in crispr2host else '-')
    cluster_df['Crispr_score_db'] = cluster_df['Accession'].apply(lambda x: crispr_pred_db[x]['ident'] if x in crispr_pred_db else 0.0)
    cluster_df['Crispr_NCBI_db'] = cluster_df['Accession'].apply(lambda x: crispr2host_NCBI_lineage[x] if x in crispr2host_NCBI_lineage else '-')
    cluster_df['Crispr_GTDB_db'] = cluster_df['Accession'].apply(lambda x: crispr2host_GTDB_lineage[x] if x in crispr2host_GTDB_lineage else '-')
    cluster_df['Crispr_mag'] = cluster_df['Accession'].apply(lambda x: crispr_pred_mag[x]['pred'] if x in crispr_pred_mag else '-')
    cluster_df['Crispr_score_mag'] = cluster_df['Accession'].apply(lambda x: crispr_pred_mag[x]['ident'] if x in crispr_pred_mag else 0.0)
    cluster_df['BLASTN_mag'] = cluster_df['Accession'].apply(lambda x: blast_pred_mag[x]['pred'] if x in blast_pred_mag else '-')
    cluster_df['BLASTN_score_mag'] = cluster_df['Accession'].apply(lambda x: blast_pred_mag[x]['ident'] if x in blast_pred_mag else 0.0)
    cluster_df['tRNA_mag'] = cluster_df['Accession'].apply(lambda x: tRNA_pred_mag[x]['pred'] if x in tRNA_pred_mag else '-')
    cluster_df['tRNA_score_mag'] = cluster_df['Accession'].apply(lambda x: tRNA_pred_mag[x]['ident'] if x in tRNA_pred_mag else 0.0)
    cluster_df['Kmer_mag'] = cluster_df['Accession'].apply(lambda x: virus2host_kmer_reorg[x]['pred'] if x in virus2host_kmer_reorg else '-')
    cluster_df['Kmer_score_mag'] = cluster_df['Accession'].apply(lambda x: virus2host_kmer_reorg[x]['ident'] if x in virus2host_kmer_reorg else 0.0)



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
        # check whether  NCBI accession exist
        if any('phabox' in acc for acc in acc_list):
            continue
        idx = group['Length'].idxmax()
        Lineage = group.loc[idx, 'Lineage']
        cluster_df.loc[group.index, 'Lineage'] = Lineage
        Genus = group.loc[idx, 'Genus']
        cluster_df.loc[group.index, 'Genus'] = Genus
        Host = group.loc[idx, 'Host']
        cluster_df.loc[group.index, 'Host'] = Host
        ncbi = group.loc[idx, 'Host_NCBI']
        cluster_df.loc[group.index, 'Host_NCBI'] = ncbi
        gtdb = group.loc[idx, 'Host_GTDB']
        cluster_df.loc[group.index, 'Host_GTDB'] = gtdb

    # assign the Lineage according to the entry in database
    for cluster, group in groups:
        acc_list = group['Accession']
        # check whether  NCBI accession exist
        if not any('phabox' in acc for acc in acc_list):
            continue
        ref_group = group[group['Accession'].apply(lambda x: 'phabox' in x)]
        query_group = group[group['Accession'].isin(query_df['Accession'])]
        if len(ref_group[ref_group['Genus'] != '-']['Genus'].unique()) == 1:
            Lineage = ref_group['Lineage'].values[0]
            cluster_df.loc[query_group.index, 'Lineage'] = Lineage
            Genus = ref_group['Genus'].values[0]
            cluster_df.loc[group.index, 'Genus'] = Genus
            Host = ref_group['Host'].values[0]
            cluster_df.loc[group.index, 'Host'] = Host
            ncbi = ref_group['Host_NCBI'].values[0]
            cluster_df.loc[group.index, 'Host_NCBI'] = ncbi
            gtdb = ref_group['Host_GTDB'].values[0]
            cluster_df.loc[group.index, 'Host_GTDB'] = gtdb
        else:
            # assign the host for the query group accoding to the genus
            for idx, row in query_group[query_group['Genus'] != '-'].iterrows():
                try:
                    cluster_df.loc[idx, 'Host'] = genus2host[row['Genus']]
                except:
                    pass


    df = cluster_df[cluster_df['Accession'].isin(genomes.keys())].copy()
    df = df.reset_index(drop=True)

    logger.info(f"[9/{jy}] summarizing the results...")
    predicted = df[df['Host'] != '-']
    unpredicted = df[df['Host'] == '-']

    df['Score'] = df['Score'].astype(str)
    df.loc[unpredicted.index, 'Score'] = '-'
    df.loc[predicted.index, 'Score'] = [genus2score[acc].split(';')[-1] if acc in genus2score else 1 for acc in predicted['Genus']]
    # Update 'Method' column where 'Host' is not '-'
    df.loc[predicted.index, 'Method'] = 'AAI-based'

    for index, row in unpredicted.iterrows():
        if row['Crispr_mag'] != '-':
            df.loc[index, 'Host'] = row['Crispr_mag']
            df.loc[index, 'Score'] = row['Crispr_score_mag']
            df.loc[index, 'Method'] = 'CRISPR-based (MAG)'
            df.loc[index, 'Host_NCBI'] = '-'
            df.loc[index, 'Host_GTDB'] = '-'
        elif row['BLASTN_mag'] != '-':
            df.loc[index, 'Host'] = row['BLASTN_mag']
            df.loc[index, 'Score'] = row['BLASTN_score_mag']
            df.loc[index, 'Method'] = 'BLASTN-based (MAG)'
            df.loc[index, 'Host_NCBI'] = '-'
            df.loc[index, 'Host_GTDB'] = '-'
        elif row['tRNA_mag'] != '-':
            df.loc[index, 'Host'] = row['tRNA_mag']
            df.loc[index, 'Score'] = row['tRNA_score_mag']
            df.loc[index, 'Method'] = 'tRNA-based (MAG)'
            df.loc[index, 'Host_NCBI'] = '-'
            df.loc[index, 'Host_GTDB'] = '-'
        elif row['Kmer_mag'] != '-':
            df.loc[index, 'Host'] = row['Kmer_mag']
            df.loc[index, 'Score'] = row['Kmer_score_mag']
            df.loc[index, 'Method'] = 'Kmer-based (MAG)'
            df.loc[index, 'Host_NCBI'] = '-'
            df.loc[index, 'Host_GTDB'] = '-'
        elif row['Crispr_db'] != '-':
            df.loc[index, 'Host'] = row['Crispr_db']
            df.loc[index, 'Score'] = row['Crispr_score_db']
            df.loc[index, 'Method'] = 'CRISPR-based (DB)'
            df.loc[index, 'Host_NCBI'] = row['Crispr_NCBI_db']
            df.loc[index, 'Host_GTDB'] = row['Crispr_GTDB_db']

    for index, row in predicted.iterrows():
        if row['Crispr_mag'] != '-':
            df.loc[index, 'Host'] = row['Crispr_mag']
            df.loc[index, 'Score'] = row['Crispr_score_mag']
            df.loc[index, 'Method'] = 'CRISPR-based (MAG)'
            df.loc[index, 'Host_NCBI'] = '-'
            df.loc[index, 'Host_GTDB'] = '-'
        elif row['BLASTN_mag'] != '-':
            df.loc[index, 'Host'] = row['BLASTN_mag']
            df.loc[index, 'Score'] = row['BLASTN_score_mag']
            df.loc[index, 'Method'] = 'BLASTN-based (MAG)'
            df.loc[index, 'Host_NCBI'] = '-'
            df.loc[index, 'Host_GTDB'] = '-'
        elif row['tRNA_mag'] != '-':
            df.loc[index, 'Host'] = row['tRNA_mag']
            df.loc[index, 'Score'] = row['tRNA_score_mag']
            df.loc[index, 'Method'] = 'tRNA-based (MAG)'
            df.loc[index, 'Host_NCBI'] = '-'
            df.loc[index, 'Host_GTDB'] = '-'
        elif row['Kmer_mag'] != '-':
            df.loc[index, 'Host'] = row['Kmer_mag']
            df.loc[index, 'Score'] = row['Kmer_score_mag']
            df.loc[index, 'Method'] = 'Kmer-based (MAG)'
            df.loc[index, 'Host_NCBI'] = '-'
            df.loc[index, 'Host_GTDB'] = '-'

    """
    groups = df.groupby('cluster')
    for cluster, group in groups:
        idx_mag = group['Crispr_score_mag'].idxmax()
        mag_crispr_host = group.loc[idx_mag, 'Crispr_mag']
        mag_blast_host = group.loc[idx_mag, 'BLASTN_mag']
        idx_db = group['Crispr_score_db'].idxmax()
        db_host  = group.loc[idx_db, 'Crispr_db']
        idx_genus = group['Length'].idxmax()
        genus_host = group.loc[idx_genus, 'Genus']
        if mag_crispr_host == '-' and db_host == '-' and genus_host != '-':
            unpredicted = group[group['Host'] == '-']
            df.loc[unpredicted.index, 'Method'] = '-'
        elif mag_crispr_host != '-':
            host = mag_crispr_host
            df.loc[group.index, 'Host'] = host
            df.loc[group.index, 'Score'] = group.loc[idx_mag, 'Crispr_score_mag']
            df.loc[group.index, 'Method'] = 'CRISPR-based (MAG)'
        elif db_host != '-' and genus_host == '-':
            host = db_host
            df.loc[group.index, 'Host'] = host
            df.loc[group.index, 'Score'] = group.loc[idx_db, 'Crispr_score_db']
            df.loc[group.index, 'Method'] = 'CRISPR-based (DB)'
            df.loc[group.index, 'Host_NCBI'] = group.loc[idx_db, 'Crispr_NCBI_db']
            df.loc[group.index, 'Host_GTDB'] = group.loc[idx_db, 'Crispr_GTDB_db']
    """

    df = df.sort_values('Accession', ascending=False)
    #test
    df.to_csv(f'{rootpth}/{midfolder}/test.tsv', index=False, sep='\t')
    df = df[['Accession', 'Length', 'Host', 'Host_NCBI', 'Host_GTDB', 'Score', 'Method']]
    df.to_csv(f'{rootpth}/{midfolder}/cherry_prediction.tsv', index=False, sep='\t')


    ###############################################################
    ####################### dump results ##########################
    ###############################################################
    logger.info(f"[10/{jy}] writing the results...")
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
    all_pred = df['Host'].tolist() + ['filtered'] * len(filtered_contig) + ['-'] * len(unpredicted_contig)
    all_score = df['Score'].tolist() + [0] * len(filtered_contig) + [0] * len(unpredicted_contig)
    all_length = df['Length'].tolist() + filtered_lenth + unpredicted_length
    all_method = df['Method'].tolist() + ['-'] * len(filtered_contig) + ['-'] * len(unpredicted_contig)
    all_ncbi = df['Host_NCBI'].tolist() + ['-'] * len(filtered_contig) + ['-'] * len(unpredicted_contig)
    all_gtdb = df['Host_GTDB'].tolist() + ['-'] * len(filtered_contig) + ['-'] * len(unpredicted_contig)

    """
    for item, linage in zip(all_pred, all_ncbi):
        all_linage = []
        if linage == '-':
            try:
                all_linage.append(genus2hostlineage[item]) 
            except:
                all_linage.append('-')
        else:
            all_linage.append(linage)
        all_ncbi = all_linage
    """
            

    if bgtdb != 'None':
        all_linage = []
        for item, linage in zip(all_pred, all_gtdb):
            try:
                all_linage.append(mag2taxonomy[item])
            except:
                try:
                    all_linage.append(species2linage[item])
                except:
                    try:
                        all_linage.append(genus2linage[item])
                    except:
                        try:
                            all_linage.append(family2linage[item])
                        except:
                            all_linage.append(linage)
        all_gtdb = all_linage
    
    # Create DataFrame directly from lists
    contig_to_pred = pd.DataFrame({
        'Accession': all_contigs,
        'Length': all_length,
        'Host': all_pred,
        'CHERRYScore': all_score,
        'Method': all_method,
        'Host_NCBI_lineage': all_ncbi,
        'Host_GTDB_lineage': all_gtdb
    })

    # Save DataFrame to CSV
    contig_to_pred.to_csv(f"{rootpth}/{out_dir}/cherry_prediction.tsv", index=False, sep='\t')
    query2host = {acc: genus for acc, genus in zip(df['Accession'], df['Host'])}


    cherry_node = pd.DataFrame({
        'Accession': list(refacc2host.keys()) + list(query2host.keys()), 
        'Host': list(refacc2host.values()) + list(query2host.values()), 
        'TYPE': ['Ref']*len(refacc2host) + ['Query']*len(query2host)
        })
    
    cherry_node.to_csv(f"{rootpth}/{out_dir}/{supplementary}/cherry_network_nodes.tsv", index=False, sep='\t')

    # add host nodes to the network
    df = pd.read_csv(f'{rootpth}/{out_dir}/cherry_prediction.tsv', sep='\t')
    df = df[df['Host'] != '-']

    ref_df = pd.read_csv(f'{rootpth}/{out_dir}/{supplementary}/cherry_network_nodes.tsv', sep='\t')
    ref_df = ref_df[(ref_df['Host'] != '-')&(ref_df['TYPE'] == 'Ref')]
    
    # rewrite the edges file
    Source = df['Accession'].tolist()+ref_df['Accession'].tolist()
    Target = df['Host'].tolist()+ref_df['Host'].tolist()
    Weight = df['CHERRYScore'].tolist()+[1]*len(ref_df['Accession'].tolist())
    host_edges = pd.DataFrame({"Source": Source, "Target": Target, "Weight": Weight})

    # rewrite the nodes file
    Accession = df['Host'].tolist() + df['Accession'].tolist() + list(set(ref_df['Host'].tolist()))
    Host = df['Host'].tolist() + df['Host'].tolist() + list(set(ref_df['Host'].tolist()))
    TYPE = ['Host'] * len(df['Host'].tolist()) + ['Query'] * len(df['Accession'].tolist()) + ['Host'] * len(set(ref_df['Host'].tolist()))
    host_nodes = pd.DataFrame({"Accession": Accession, "Host": Host, "TYPE": TYPE})

    edges_df = pd.concat((pd.read_csv(f'{rootpth}/{out_dir}/{supplementary}/cherry_network_edges.tsv', sep='\t'), host_edges))
    nodes_df = pd.concat((pd.read_csv(f'{rootpth}/{out_dir}/{supplementary}/cherry_network_nodes.tsv', sep='\t'), host_nodes))


    nodes_df.drop_duplicates(subset=['Accession'], keep='first', inplace=True)
    edges_df.drop_duplicates(subset=['Source', 'Target'], keep='first', inplace=True)
    edges_df.to_csv(f'{rootpth}/{out_dir}/{supplementary}/cherry_network_edges.tsv', sep='\t', index=False)
    nodes_df.to_csv(f'{rootpth}/{out_dir}/{supplementary}/cherry_network_nodes.tsv', sep='\t', index=False)


    #################################################
    ########     multi-host prediction      #########
    #################################################
    if bfolder != 'None':
        if blast_pred_mag != {}:
            blast_df = pd.read_csv(f'{rootpth}/{midfolder}/blastn_MAGs.tsv', sep='\t')
            blast_df.rename(columns={'qseqid': 'Accession', 'sseqid': 'Host', 'pident': 'CHERRYScore'}, inplace=True)
            blast_df['Method'] = 'BLASTN-based (MAG)'
            blast_df['Length'] = blast_df['Accession'].apply(lambda x: genomes[x].length if x in genomes else '-')
            blast_df['Host_NCBI_lineage'] = '-'
            if bgtdb != 'None':
                blast_df['Host_GTDB_lineage'] = blast_df['Host'].apply(lambda x: mag2taxonomy[x] if x in mag2taxonomy else '-')
            else:
                blast_df['Host_GTDB_lineage'] = '-'
            blast_df = blast_df[['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_NCBI_lineage', 'Host_GTDB_lineage']]
        else:
            blast_df = pd.DataFrame(columns=['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_NCBI_lineage', 'Host_GTDB_lineage'])
        if crispr_pred_mag != {}:    
            crispr_df = pd.read_csv(f'{rootpth}/{midfolder}/CRISPRs_alignment_MAG.tsv', sep='\t')
            crispr_df.rename(columns={'qseqid': 'Accession', 'sseqid': 'Host', 'pident': 'CHERRYScore'}, inplace=True)
            crispr_df['Method'] = 'CRISPR-based (MAG)'
            crispr_df['Host'] = crispr_df['Host'].apply(lambda x: x.split('_CRISPR_')[0])
            crispr_df['Length'] = crispr_df['Accession'].apply(lambda x: genomes[x].length if x in genomes else '-')
            crispr_df['Host_NCBI_lineage'] = '-'
            if bgtdb != 'None':
                crispr_df['Host_GTDB_lineage'] = crispr_df['Host'].apply(lambda x: mag2taxonomy[x] if x in mag2taxonomy else '-')
            else:
                crispr_df['Host_GTDB_lineage'] = '-'
        else:
            crispr_df = pd.DataFrame(columns=['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_NCBI_lineage', 'Host_GTDB_lineage'])
        if tRNA_pred_mag != {}:
            tRNA_df = pd.read_csv(f'{rootpth}/{midfolder}/tRNA_MAGs.tsv', sep='\t')
            tRNA_df.rename(columns={'virus': 'Accession', 'mag': 'Host', 'pident': 'CHERRYScore'}, inplace=True)
            tRNA_df['Method'] = 'tRNA-based (MAG)'
            tRNA_df['Length'] = tRNA_df['Accession'].apply(lambda x: genomes[x].length if x in genomes else '-')
            tRNA_df['Host_NCBI_lineage'] = '-'
            if bgtdb != 'None':
                tRNA_df['Host_GTDB_lineage'] = tRNA_df['Host'].apply(lambda x: mag2taxonomy[x] if x in mag2taxonomy else '-')
            else:
                tRNA_df['Host_GTDB_lineage'] = '-'
            tRNA_df = tRNA_df[['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_NCBI_lineage', 'Host_GTDB_lineage']]
        else:
            tRNA_df = pd.DataFrame(columns=['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_NCBI_lineage', 'Host_GTDB_lineage'])
        if virus2host_kmer_reorg != {}:
            virus2host = pkl.load(open(f'{rootpth}/{midfolder}/virus2host_kmer_reorg.pkl', 'rb'))
            virus = []
            mag = []
            score = []
            for v, pred in virus2host.items():
                h = pred['pred']
                s = pred['ident']
                virus.append(v)
                mag.append(h)
                score.append(s)
            kmer_df = pd.DataFrame({'Accession': virus, 'Host': mag, 'CHERRYScore': score})
            kmer_df['Method'] = 'Kmer-based (MAG)'
            kmer_df['Length'] = kmer_df['Accession'].apply(lambda x: genomes[x].length if x in genomes else '-')
            kmer_df['Host_NCBI_lineage'] = '-'
            if bgtdb != 'None':
                kmer_df['Host_GTDB_lineage'] = kmer_df['Host'].apply(lambda x: mag2taxonomy[x] if x in mag2taxonomy else '-')
            else:
                kmer_df['Host_GTDB_lineage'] = '-'
            kmer_df = kmer_df[['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_NCBI_lineage', 'Host_GTDB_lineage']]
        else:
            kmer_df = pd.DataFrame(columns=['Accession', 'Length', 'Host', 'CHERRYScore', 'Method', 'Host_NCBI_lineage', 'Host_GTDB_lineage'])
        multi_df = pd.concat([blast_df, crispr_df, tRNA_df, kmer_df, contig_to_pred])
        multi_df.drop_duplicates(subset=['Accession', 'Host', 'Method'], keep='first', inplace=True)
        multi_df = multi_df.sort_values(['Accession', 'CHERRYScore'], ascending=[True, False])
        multi_df.to_csv(f"{rootpth}/{out_dir}/cherry_multi_prediction.tsv", index=False, sep='\t')



    if inputs.draw == 'Y':
        draw_network(f'{rootpth}/{out_dir}/{supplementary}/', f'{rootpth}/{out_dir}/{supplementary}', 'cherry')
    
    if inputs.task != 'end_to_end':
        genes = load_gene_info(f'{rootpth}/{midfolder}/query_protein.fa', genomes)
        run_command(f"cp {rootpth}/filtered_contigs.fa {rootpth}/{out_dir}/{supplementary}/all_predicted_contigs.fa")
        run_command(f"cp {rootpth}/{midfolder}/query_protein.fa {rootpth}/{out_dir}/{supplementary}/all_predicted_protein.fa")
        run_command(f"cp {rootpth}/{midfolder}/db_results.tab {rootpth}/{out_dir}/{supplementary}/alignment_results.tab")
        tmp = pd.read_csv(f'{rootpth}/{out_dir}/{supplementary}/alignment_results.tab', sep='\t', names=['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'])
        tmp.to_csv(f'{rootpth}/{out_dir}/{supplementary}/alignment_results.tab', sep='\t', index=False)
        #run_command(f'sed -i "1i\qseqid\tsseqid\tpident\tlength\tmismatch\tgapopen\tqstart\tqend\tsstart\tsend\tevalue\tbitscore" {rootpth}/{out_dir}/{supplementary}/alignment_results.tab')

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

        phavip_dump_result(genomes, rootpth, out_dir, logger, supplementary = 'cherry_supplementary')

    
    logger.info("Run time: %s seconds\n" % round(time.time() - program_start, 2))
