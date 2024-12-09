import os
import re
import sys
import pandas as pd
import numpy as np
import pickle as pkl
import subprocess
from shutil import which
from collections import Counter
from Bio import SeqIO
from xml.etree import ElementTree as ElementTree


def blastxml_to_tabular(in_file, out_file):
    output_handle = open(out_file, "w")
    if not os.path.isfile(in_file):
        sys.exit("Input BLAST XML file not found: %s" % in_file)


    re_default_query_id = re.compile(r"^Query_\d+$")
    assert re_default_query_id.match(r"Query_101")
    assert not re_default_query_id.match(r"Query_101a")
    assert not re_default_query_id.match(r"MyQuery_101")
    re_default_subject_id = re.compile(r"^Subject_\d+$")
    assert re_default_subject_id.match(r"Subject_1")
    assert not re_default_subject_id.match(r"Subject_")
    assert not re_default_subject_id.match(r"Subject_12a")
    assert not re_default_subject_id.match(r"TheSubject_1")

    # get an iterable
    try:
        parser = ElementTree.XMLParser(encoding="utf-8")
        context = ElementTree.iterparse(in_file, events=("start", "end"), parser=parser)
    except Exception:
        sys.exit("Invalid data format.")
    # turn it into an iterator
    context = iter(context)
    # get the root element
    try:
        event, root = next(context)
    except Exception:
        sys.exit("Invalid data format.")
    for event, elem in context:
        if event == "end" and elem.tag == "BlastOutput_program":
            blast_program = elem.text
        # for every <Iteration> tag
        if event == "end" and elem.tag == "Iteration":
            qseqid = elem.findtext("Iteration_query-ID")
            if re_default_query_id.match(qseqid):
                # Place holder ID, take the first word of the query definition
                qseqid = elem.findtext("Iteration_query-def").split(None, 1)[0]
            qlen = int(elem.findtext("Iteration_query-len"))

            # for every <Hit> within <Iteration>
            for hit in elem.findall("Iteration_hits/Hit"):
                sseqid = hit.findtext("Hit_id").split(None, 1)[0]
                hit_def = sseqid + " " + hit.findtext("Hit_def")
                if re_default_subject_id.match(sseqid) and sseqid == hit.findtext(
                    "Hit_accession"
                ):
                    # Place holder ID, take the first word of the subject definition
                    hit_def = hit.findtext("Hit_def")
                    sseqid = hit_def.split(None, 1)[0]
                if sseqid.startswith(
                    "gnl|BL_ORD_ID|"
                ) and sseqid == "gnl|BL_ORD_ID|" + hit.findtext("Hit_accession"):
                    # Alternative place holder ID, again take the first word of hit_def
                    hit_def = hit.findtext("Hit_def")
                    sseqid = hit_def.split(None, 1)[0]
                # for every <Hsp> within <Hit>
                for hsp in hit.findall("Hit_hsps/Hsp"):
                    nident = hsp.findtext("Hsp_identity")
                    length = hsp.findtext("Hsp_align-len")
                    # As of NCBI BLAST+ 2.4.0 this is given to 3dp (not 2dp)
                    pident = "%0.3f" % (100 * float(nident) / float(length))

                    q_seq = hsp.findtext("Hsp_qseq")
                    h_seq = hsp.findtext("Hsp_hseq")
                    m_seq = hsp.findtext("Hsp_midline")
                    assert len(q_seq) == len(h_seq) == len(m_seq) == int(length)
                    gapopen = str(
                        len(q_seq.replace("-", " ").split())
                        - 1
                        + len(h_seq.replace("-", " ").split())
                        - 1
                    )

                    mismatch = (
                        m_seq.count(" ")
                        + m_seq.count("+")
                        - q_seq.count("-")
                        - h_seq.count("-")
                    )
                    expected_mismatch = len(q_seq) - sum(
                        1
                        for q, h in zip(q_seq, h_seq)
                        if q == h or q == "-" or h == "-"
                    )
                    xx = sum(1 for q, h in zip(q_seq, h_seq) if q == "X" and h == "X")
                    if not (
                        expected_mismatch - q_seq.count("X")
                        <= int(mismatch)
                        <= expected_mismatch + xx
                    ):
                        sys.exit(
                            "%s vs %s mismatches, expected %i <= %i <= %i"
                            % (
                                qseqid,
                                sseqid,
                                expected_mismatch - q_seq.count("X"),
                                int(mismatch),
                                expected_mismatch,
                            )
                        )

                    expected_identity = sum(1 for q, h in zip(q_seq, h_seq) if q == h)
                    if not (
                        expected_identity - xx
                        <= int(nident)
                        <= expected_identity + q_seq.count("X")
                    ):
                        sys.exit(
                            "%s vs %s identities, expected %i <= %i <= %i"
                            % (
                                qseqid,
                                sseqid,
                                expected_identity,
                                int(nident),
                                expected_identity + q_seq.count("X"),
                            )
                        )

                    evalue = hsp.findtext("Hsp_evalue")
                    if evalue == "0":
                        evalue = "0.0"
                    else:
                        evalue = "%0.0e" % float(evalue)

                    bitscore = float(hsp.findtext("Hsp_bit-score"))
                    if bitscore < 100:
                        # Seems to show one decimal place for lower scores
                        bitscore = "%0.1f" % bitscore
                    else:
                        # Note BLAST does not round to nearest int, it truncates
                        bitscore = "%i" % bitscore

                    values = [
                        qseqid,
                        sseqid,
                        pident,
                        length,  # hsp.findtext("Hsp_align-len")
                        str(mismatch),
                        gapopen,
                        hsp.findtext("Hsp_query-from"),  # qstart,
                        hsp.findtext("Hsp_query-to"),  # qend,
                        hsp.findtext("Hsp_hit-from"),  # sstart,
                        hsp.findtext("Hsp_hit-to"),  # send,
                        evalue,  # hsp.findtext("Hsp_evalue") in scientific notation
                        bitscore,  # hsp.findtext("Hsp_bit-score") rounded
                    ]
                    output_handle.write("\t".join(values) + "\n")
            # prevents ElementTree from growing large datastructure
            root.clear()
            elem.clear()
    output_handle.close()
    return



#############################################################
####################  DIAMOND BLASTP  #######################
#############################################################

def run_diamond(diamond_db, outpth, infile, tool, threads):
    try:
        # running alignment
        diamond_cmd = f'diamond blastp --outfmt 5 --threads {threads} -d {diamond_db} -q {outpth}/{infile} -o {outpth}/{tool}_results.xml -k 25 --quiet'
        #print("Running Diamond...")
        _ = subprocess.check_call(diamond_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        content = open(f'{outpth}/{tool}_results.xml', 'r').read()
        content = content.replace('&', '')
        with open(f'{outpth}/{tool}_results.xml', 'w') as file:
            file.write(content)
    except:
        print(diamond_cmd)
        print("diamond blastp failed")
        exit(1)

def convert_xml(outpth, tool):
    blastxml_to_tabular(f'{outpth}/{tool}_results.xml', f'{outpth}/{tool}_results.tab')
    try:
        diamond_out_fp = f"{outpth}/{tool}_results.tab"
        database_abc_fp = f"{outpth}/{tool}_results.abc"
        _ = subprocess.check_call("awk '{{print $1,$2,$12}}' {0} > {1}".format(diamond_out_fp, database_abc_fp), shell=True)
    except:
        print("convert xml failed")
        exit(1)


def parse_evalue(blast_df, outpth, tool):
    protein2evalue = {}
    for protein, evalue in zip(blast_df['query'].values, blast_df['evalue'].values):
        try:
            protein2evalue[protein]
        except:
            protein2evalue[protein] = evalue        

    return protein2evalue

def parse_coverage(blast_df):
    with open(f'{blast_df}') as file_out:
        check_name_single = {}
        for line in file_out.readlines():
            parse = line.replace("\n", "").split("\t")
            virus = parse[0]
            qstart = float(parse[-5])
            tqend = float(parse[-4])
            sstart   = float(parse[-3]) 
            ssend = float(parse[-2])     
            tmp_score = np.abs((qstart-tqend)/(sstart-ssend))
            if tmp_score < 0.7:
                continue
            if virus in check_name_single:
                continue
            check_name_single[virus] = tmp_score
            
    return check_name_single

def parse_xml(protein2id, outpth, tool):
    xml_files = ['']*len(protein2id)
    flag = 0 # 0 for common, 1 for specific item
    with open(f'{outpth}/{tool}_results.xml', 'r') as file:
        content = file.readlines()
        for i in range(len(content)):
            line = content[i]
            if '<Iteration>\n' == line:
                item = content[i+3]
                qseqid = item.split('>', 1)[1]
                qseqid = qseqid.split(' ', 1)[0]
                try:
                    idx = protein2id[qseqid]
                    xml_files[idx]+=line
                    flag = 1
                except:
                    flag = 2
            elif line == '</Iteration>\n':
                if flag == 1:
                    xml_files[idx]+=line
                flag = 0
            elif flag == 0:
                for j in range(len(xml_files)):
                    xml_files[j]+=line
            elif flag == 1:
                xml_files[idx]+=line
            elif flag == 2:
                continue
    return xml_files


def parse_xml2(protein2id, outpth, tool):
    #protein2id = {'YP_009984512.1': 0, 'YP_009984889.1':1}
    xml_files = ['<?xml version="1.0"?>\n<!DOCTYPE BlastOutput PUBLIC "">\n']*len(protein2id)
    flag = 0 # 0 for common, 1 for specific item
    start = 0 # start position to writein
    context = ElementTree.iterparse(f'{outpth}/{tool}_results.xml', events=("start", "end"))
    for event, elem in context:
        if elem.tag == 'Iteration' and event == 'start':
            try:
                qseqid = elem.findtext("Iteration_query-def").split(" ", 1)[0]
                idx = protein2id[qseqid]
                xml_files[idx]+=f'<{elem.tag}>{elem.text}'
                flag = 1
            except:
                flag = 2
        elif elem.tag == 'Iteration' and event == 'end':
            if flag == 1:
                xml_files[idx]+=f'</{elem.tag}>\n'
            elif flag == 2:
                pass
            flag = 0
        elif flag == 0 and event =='start':
            for i in range(len(xml_files)):
                xml_files[i]+=f'<{elem.tag}>{elem.text}'
        elif flag == 0 and event =='end':
            for i in range(len(xml_files)):
                xml_files[i]+=f'</{elem.tag}>\n'
        elif flag == 1 and event =='start':
            xml_files[idx]+=f'<{elem.tag}>{elem.text}'
        elif flag == 1 and event =='end':
            xml_files[idx]+=f'</{elem.tag}>\n'
        elif flag == 2:
            continue
    return xml_files

def parse_position(outpth):
    protein2start = {}
    protein2end   = {}
    for record in SeqIO.parse(f'{outpth}/test_protein.fa', 'fasta'):
        description = str(record.description)
        description = description.split(' # ')
        start = description[1]
        end   = description[2]
        protein2start[record.id] = f'{start}'
        protein2end[record.id] =f'{end}'
    return protein2start, protein2end




#############################################################
####################  Contig2Sentence  ######################
#############################################################

def contig2sentence(db_dir, outpth, genomes):
    # Load dictonary and BLAST results
    #proteins_df = pd.read_csv(f'{db_dir}/proteins.csv')
    #proteins_df.dropna(axis=0, how='any', inplace=True)
    #pc2wordsid = {pc: idx for idx, pc in enumerate(sorted(set(proteins_df['cluster'].values)))}
    #protein2pc = {protein: pc for protein, pc in zip(proteins_df['protein_id'].values, proteins_df['cluster'].values)}
    protein2pc = pkl.load(open(f'{db_dir}/protein2token.pkl', 'rb'))
    pc2wordsid = pkl.load(open(f'{db_dir}/pc2wordsid.pkl', 'rb'))
    blast_df = pd.read_csv(f"{outpth}/db_results.abc", sep=' ', names=['query', 'ref', 'pident', 'bitscore'])
    max_bitscore_per_query = blast_df.groupby('query')['bitscore'].transform('max')
    blast_df = blast_df[blast_df['bitscore'] > max_bitscore_per_query * 0.9]
    blast_df['genome'] = blast_df['query'].apply(lambda x: x.rsplit('_', 1)[0])
    blast_df = blast_df[blast_df['genome'].isin(genomes.keys())]

    # Parse the DIAMOND results
    contig2pcs = {}
    check = {}
    for query, ref, evalue in zip(blast_df['query'].values, blast_df['ref'].values, blast_df['bitscore'].values):
        try:
            _ = check[query]
        except KeyError:
            try:
                pc = pc2wordsid[protein2pc[ref]]
            except:
                continue
            conitg = query.rsplit('_', 1)[0]
            idx    = query.rsplit('_', 1)[1]
            try:
                contig2pcs[conitg].append((idx, pc, evalue))
            except:
                contig2pcs[conitg] = [(idx, pc, evalue)]
            check[query] = 1

    # Sorted by position
    for contig in contig2pcs:
        contig2pcs[contig] = sorted(contig2pcs[contig], key=lambda tup: tup[0])

    # Contigs2sentence
    contig2id = {contig:idx for idx, contig in enumerate(contig2pcs.keys())}
    id2contig = {idx:contig for idx, contig in enumerate(contig2pcs.keys())}
    sentence = np.zeros((len(contig2id.keys()), 300))
    sentence_weight = np.ones((len(contig2id.keys()), 300))
    for row in range(sentence.shape[0]):
        contig = id2contig[row]
        pcs = contig2pcs[contig]
        for col in range(len(pcs)):
            try:
                _, sentence[row][col], sentence_weight[row][col] = pcs[col]
                sentence[row][col] += 1
            except:
                break


    # propotion
    rec = []
    for key in blast_df['query'].unique():
        name = key.rsplit('_', 1)[0]
        rec.append(name)
    counter = Counter(rec)

    for genome in counter.keys():
        genomes[genome].proportion = counter[genome]/len(genomes[genome].genes)


    mapped_num = np.array([counter[item] for item in id2contig.values()])

    total_num = np.array([len(genomes[item].genes) for item in id2contig.values()])
    proportion = mapped_num/total_num


    # Store the parameters
    #pkl.dump(sentence,        open(f'{outpth}/sentence.feat', 'wb'))
    #pkl.dump(id2contig,       open(f'{outpth}/sentence_id2contig.dict', 'wb'))
    #pkl.dump(proportion,      open(f'{outpth}/sentence_proportion.feat', 'wb'))
    #pkl.dump(pc2wordsid,      open(f'{outpth}/pc2wordsid.dict', 'wb'))

    return sentence, id2contig, proportion, pc2wordsid




#############################################################
#################  Convert2BERT input  ######################
#############################################################

def generate_bert_input(outpth, feat, pcs):
    #feat = pkl.load(open(f'{inpth}/sentence.feat', 'rb'))
    #pcs = pkl.load(open(f'{inpth}/pc2wordsid.dict', 'rb'))
    id2pcs = {item: key for key, item in pcs.items()}
    text = []
    label = []
    for line in feat:
        sentence = ""
        flag = 0
        for i in range(len(line)-2):
            if line[i]-1 == -1:
                flag = 1
                sentence = sentence[:-1]
                break
            sentence = sentence + id2pcs[line[i]-1] + ' '
        if flag == 0:
            sentence = sentence[:-1]
        text.append(sentence)
        label.append(1)

    feat_df = pd.DataFrame({'label':label, 'text':text})
    feat_df.to_csv(f'{outpth}/bert_feat.csv', index=None)


