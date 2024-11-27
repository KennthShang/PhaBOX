#!/usr/bin/env python
from phabox2 import phamer, phatyp, phagcn, cherry, contamination, votu, tree, phavip
import argparse
import os
import pandas as pd
from  phabox2.scripts.ulity import *


__version__ = "2.1.6"
description = """
                                  \033[1m\033[96mPhaBOX v2.1.6\033[0m\033[0m                  
               \033[1m\033[96mJiayu SHANG, Cheng Peng, and Yanni SUN Otc. 2024\033[0m\033[0m 


\033[1mDocumentation, support and updates available at https://github.com/KennthShang/PhaBOX/wiki \033[0m


Syntax: phabox2 [--help] [--task TASK] [--dbdir DBDIR] [--outpth OUTPTH] 
                    [--contigs CONTIGS] [--proteins PROTEINS]
                    [--midfolder MIDFOLDER] [--threads THREADS]
                    [--len LEN] [--reject REJECT] [--aai AAI] 
                    [--share SHARE] [--pcov PCOV]
                    [--pident PIDENT] [--cov COV] 
                    [--blast BLAST] [--sensitive SENSITIVE] 
                    [--draw DRAW] [--marker MARKER [MARKER ...]]
                    [--tree TREE] [--bfolder BFOLDER] [--magonly MAGONLY]
                    [--mode MODE] [--ani ANI]



\033[93mGeneral options:\033[0m

\033[94m--task\033[0m    
    Select a program to run:
    end_to_end    || Run phamer, phagcn, phatyp, cherry, and phavip once (default)
    phamer        || Virus identification
    phagcn        || Taxonomy classification
    phatyp        || Lifestyle prediction
    cherry        || Host prediction
    phavip        || Protein annotation
    contamination || Contamination/proviurs detection
    votu          || vOTU grouping (ANI-based or AAI-based)
    tree          || Build phylogenetic trees based on marker genes

    \033[93mFor more options in specific tasks, please run:\033[0m
        \033[93mphabox2 --task [task] -h\033[0m
        \033[93mExample:\033[0m
            \033[93mphabox2 --task phamer -h\033[0m
            \033[93mphabox2 --task phagcn -h\033[0m
    \033[93mend_to_end task will not show the options but use all the parameters\033[0m
    \033[93mYou can also check the parameters via: https://github.com/KennthShang/PhaBOX/wiki/Command-line-options \033[0m
            


\033[94m--dbdir\033[0m 
    Path of downloaded phabox2 database directory (required)

\033[94m--outpth\033[0m         
    Rootpth for the output folder (required)
    All the results, including intermediate files and final predictions, are stored in this folder. 

\033[94m--contigs\033[0m
    Path of the input FASTA file (required)

\033[94m--proteins\033[0m  
    FASTA file of predicted proteins. (optional)

\033[94m--midfolder\033[0m 
    Midfolder for intermediate files. (optional)
    This folder will be created within the --outpth to store intermediate files.

\033[94m--len\033[0m
    Filter the length of contigs || default: 3000
    Contigs with length smaller than this value will not proceed 

\033[94m--threads\033[0m   
    Number of threads to use || default: all available threads
"""

overal_description = """


\033[93mGeneral options:\033[0m

\033[94m--dbdir\033[0m    
    Path of downloaded phabox2 database directory (required)

\033[94m--outpth\033[0m         
    Rootpth for the output folder (required)
    All the results, including intermediate files and final predictions, are stored in this folder. 

\033[94m--contigs\033[0m
    Path of the input FASTA file (required)

\033[94m--proteins\033[0m  
    FASTA file of predicted proteins. (optional)

\033[94m--midfolder\033[0m 
    Midfolder for intermediate files. (optional)
    This folder will be created within the --outpth to store intermediate files.

\033[94m--len\033[0m
    Filter the length of contigs || default: 3000
    Contigs with length smaller than this value will not proceed 

\033[94m--threads\033[0m   
    Number of threads to use || default: all available threads
"""

phamer_description = """PhaMer: Virus identification

Usage: phabox2 --task phamer [options]


\033[93mIn-task options:\033[0m

\033[94m--reject\033[0m    
    Reject sequences in which the percent proteins aligned to known phages is smaller than the value. 
    Default: 10
    Range from 0 to 20
"""

phatyp_description = """PhaTYP: Lifestyle prediction

Usage: phabox2 --task phatyp [options]


\033[93mIn-task options:\033[0m

There are no additional options for lifestyle prediction. Only need to follow the general options.
"""

phagcn_description = """PhaGCN: Taxonomy classification

Usage: phabox2 --task phagcn [options]


\033[93mIn-task options:\033[0m


The options below are used to generate a network for virus-virus connections. 
The current parameters are optimized for the ICTV 2024 and are highly accurate for grouping genus-level vOTUs. 
When making changes, make sure you understand what they are.

\033[94m--aai\033[0m 
    Average amino acids identity  || default: 75 || range from 0 to 100 

\033[94m--share\033[0m
    Minimum shared number of proteins || default: 15 || range from 0 to 100

\033[94m--pcov\033[0m  
    Protein-based coverage || default: 80 || range from 0 to 100

\033[94m--draw\033[0m  
    Draw network examples for the query virus relationship. || default: N || Y or N 

    
--draw is used to plot sub-networks containing the query virus. We use it to generate visualization for our web server. 
However, it will only print the top 10 largest sub-networks, so we do not recommend that users use it. 
We have provided the complete network for visualization (network_edges.tsv and network_nodes.tsv file)
please check it out via: https://github.com/KennthShang/PhaBOX/wiki/Outputs#-outputs-for-specific-task
"""

cherry_description = """Cherry: Host prediction

Usage: phabox2 --task cherry [options]


\033[93mIn-task options:\033[0m

The options below are used to generate a network for virus-virus connections. 
The current parameters are optimized for the ICTV 2024 and are highly accurate for grouping genus-level vOTUs. 
When making changes, make sure you understand 100% what they are.

\033[94m--aai\033[0m 
    Average amino acids identity  || default: 75 || range from 0 to 100 

\033[94m--share\033[0m
    Minimum shared number of proteins || default: 15 || range from 0 to 100

\033[94m--pcov\033[0m  
    Protein-based coverage || default: 80 || range from 0 to 100

\033[94m--draw\033[0m  
    Draw network examples for the query virus relationship. || default: N || Y or N 

    
--draw is used to plot sub-networks containing the query virus. We use it to generate visualization for our web server. 
However, it will only print the top 10 largest sub-networks, so we do not recommend that users use it. 
We have provided the complete network for visualization (network_edges.tsv and network_nodes.tsv file).
Please check it out via: https://github.com/KennthShang/PhaBOX/wiki/Outputs#-outputs-for-specific-task


The options below are used to predict CRISPRs based on MAGs.

\033[94m--bfolder\033[0m
    Path to the folder that contains MAGs || default: None

The options below are used to align contigs to CRISPRs.

\033[94m--cpident\033[0m
    Alignment identity for CRISPRs || default: 90 || range from 90 to 100

\033[94m--ccov\033[0m
    Alignment coverage for CRISPRs || default: 90 || range from 0 to 100

\033[94m--blast\033[0m
    BLAST program for CRISPRs || default: blastn || blastn or blastn-short
    blastn-short will lead to more sensitive results but require more time to execute the program 

The default parameters are optimized for predicting prokaryotic hosts (data from the NCBI RefSeq database). 
When making changes, make sure you understand what they are.

\033[94m--magonly\033[0m
    Only predicting host based on the provided MAGs: Y or N || default: N
    Y will only predict the host based on the provided MAGs
    N will predict the host based on the MAGs and the reference database
"""

phavip_description = """PhaVIP: Virus annotation

Usage: phabox2 --task phavip [options]

\033[93mIn-task options:\033[0m

There are no additional options for lifestyle prediction. Only need to follow the general options.
"""


contamination_description = """Contamination: Contamination/proviurs detection

Usage: phabox2 --task contamination [options]


\033[93mIn-task options:\033[0m

\033[94m--sensitive\033[0m    
    Sensitive when search for the prokaryotic genes || default: N ||  Y or N
    Y will lead to more sensitive results but require more time to execute the program
"""

votu_description = """vOTU: vOTU groupping

Usage: phabox2 --task votu [options]

\033[93mIn-task options:\033[0m

\033[94m--mode\033[0m 
    Mode for clustering ANI based or AAI based || default: ANI || ANI or AAI

AAI-based options:

\033[94m--aai\033[0m 
    Average amino acids identity for AAI based genus grouping || default: 75 || range from 0 to 100

\033[94m--pcov\033[0m 
    Protein-level coverage for AAI based genus grouping || default: 80 || range from 0 to 100

\033[94m--share\033[0m 
    Minimum shared percent of proteins for AAI based genus grouping || default: 15 || range from 0 to 100

ANI-based options:

\033[94m--ani\033[0m
    Alignment identity for ANI-based clustering  || default: 95 || range from 0 to 100

\033[94m--tcov\033[0m
    Alignment coverage for ANI-based clustering || default: 85 || range from 0 to 100
"""


tree_description = """Tree: Build pylogenetic tree based on marker genes

Usage: phabox2 --task tree [options]

\033[93mIn-task options:\033[0m

\033[94m--marker\033[0m
    A list of markers used to generate tree || default: terl portal
    You can choose more than one marker to generate the tree from below:
    
    The marker genes were obtained from the RefSeq 2024:
        endolysin      || 91 percent prokaryotic virus have endolysin
        holin          || 75 percent prokaryotic virus have holin
        head           || 77 percent prokaryotic virus have marjor head
        portal         || 84 percent prokaryotic viruses have portal
        terl           || 92 percent prokaryotic viruses have terminase large subunit

        Using combinations of these markers can improve the accuracy of the tree 
        But will decrease the number of sequences in the tree.

\033[94m--mcov\033[0m
    Alignment coverage for matching marker genes || default: 50 || range from 0 to 100

\033[94m--mpident\033[0m
    Alignment identity for matching marker genes || default: 25 || range from 0 to 100

\033[94m--msa\033[0m
    Whether run msa || default: N || Y or N
    Y will run msa for the marker genes using mafft
    But this will require more time to execute the program

\033[94m--tree\033[0m
    Whether build a tree || default: N || Y or N
    Y will generate the tree based on the marker genes using FastTree
    But this will require more time to execute the program
"""



def get_task_description(task):
    descriptions = {
        'phamer': phamer_description,
        'phatyp': phatyp_description,
        'phagcn': phagcn_description,
        'cherry': cherry_description,
        'contamination': contamination_description,
        'votu': votu_description,
        'tree': tree_description,
        'phavip': phavip_description
    }
    return descriptions.get(task, f'No description for {task}.\nPlease check the description for subprogram.')



def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='store_true', help='Show this help message and exit')
    parser.add_argument('--task', help='Select a program to run  || (default end_to_end)',  default = 'end_to_end')
    parser.add_argument('-d', '--dbdir', help='Path of database directory || (required)',  default = 'database/')
    parser.add_argument('-o', '--outpth', help='Rootpth for the output folder || (required)', default='test_out/')
    parser.add_argument('--contigs', help='Path of the input FASTA file || (required)',  default = 'test_contigs.fa')
    parser.add_argument('--proteins', help='FASTA file of predicted proteins || (optional)')
    parser.add_argument('--midfolder', help='Midfolder for intermediate files || (optional)', default='midfolder/')
    parser.add_argument('--threads', help='Number of threads to use || (default use all the threads)', type=int, default=int(os.cpu_count()))
    parser.add_argument('--len', help='Filter the length of contigs || (default 3000)', type=int, default=3000)
    parser.add_argument('--reject', help='Reject sequences that < 10 percent proteins aligned to known phages: 0-20  || (default 10)',  type=float, default = 10)
    parser.add_argument('--aai', help='Average amino acids identity for AAI based genus grouping: 0-100  || (default 75)',  type=float, default = 75)
    parser.add_argument('--share', help='Minimum shared number of proteins for AAI based genus grouping: 0-100  || (default 15)',  type=float, default = 15)
    parser.add_argument('--pcov', help='Protein-level coverage for AAI based genus grouping: 0-100  || (default 80)',  type=float, default = 80)
    parser.add_argument('--cpident', help='Alignment identity for CRISPRs: 90-100 || (default 90)',  type=float, default = 90)
    parser.add_argument('--ccov', help='Alignment coverage for CRISPRs: 0-100 || (default 90)',  type=float, default = 90)
    parser.add_argument('--blast', help='BLAST program for CRISPRs: blastn or blastn-short || (default blastn)', default = 'blastn')
    parser.add_argument('--bfolder', help='path to the folder that contains MAGs || (default None)', default = 'None')
    parser.add_argument('--magonly', help='Only predicting host based on the MAGs: Y or N || (default N)', default = 'N')
    parser.add_argument('--sensitive', help='Sensitive search for the prokaryotic genes: Y or N (contamination) || (default N)', default = 'N')
    parser.add_argument('--mode', help='Mode for clustering ANI based or AAI based || (default ANI)', default = 'ANI')
    parser.add_argument('--ani', help='Alignment identity for ANI-based clustering: 0-100 || (default 95)', type=float, default = 95)
    parser.add_argument('--tcov', help='Alignment coverage for ANI-based clustering: 0-100 || (default 85)',  type=float, default = 85)
    parser.add_argument('--draw', help='Draw network examples for the query virus relationship: Y or N || (default N)', default = 'N')
    parser.add_argument('--marker', nargs='+', type=str, help='A list of marker used to generate tree', default=['terl', 'portal'])
    parser.add_argument('--mpident', help='Alignment identity for matching marker genes || default: 25 || range from 0 to 100',  type=float, default = 25)
    parser.add_argument('--mcov', help='Alignment coverage for matching marker genes || default: 50 || range from 0 to 100',  type=float, default = 50)
    parser.add_argument('--msa', type=str, help='Whether run msa: Y or N ||default N', default='N')
    parser.add_argument('--tree', type=str, help='Whether build a tree: Y or N ||default N', default='N')
    inputs = parser.parse_args()

    if inputs.help:
        if inputs.task in ['phamer', 'phagcn', 'phatyp', 'cherry', 'contamination', 'votu', 'tree', 'phavip']:
            print(get_task_description(inputs.task))
            print(overal_description)
            return
        else:
            print(description)
            return

    logger = get_logger()
    logger.info(f"PhaBOX2 is running with: {inputs.threads} threads!")
    if inputs.task == "end_to_end":
        phamer.run(inputs)
        logger.info(f"PhaMer finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction')}")
        phagcn.run(inputs)
        logger.info(f"PhaGCN finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction')}")
        cherry.run(inputs)
        logger.info(f"Cherry finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction')}")
        phatyp.run(inputs)
        logger.info(f"PhaTYP finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction')}")
        df1 = pd.read_csv(os.path.join(inputs.outpth, 'final_prediction', 'phamer_prediction.tsv'), sep='\t')
        df2 = pd.read_csv(os.path.join(inputs.outpth, 'final_prediction', 'phagcn_prediction.tsv'), sep='\t')
        df3 = pd.read_csv(os.path.join(inputs.outpth, 'final_prediction', 'phatyp_prediction.tsv'), sep='\t')
        df4 = pd.read_csv(os.path.join(inputs.outpth, 'final_prediction', 'cherry_prediction.tsv'), sep='\t')
        df  = df1.merge(df2, on=['Accession', 'Length'], how='outer') \
               .merge(df3, on=['Accession', 'Length'], how='outer') \
               .merge(df4, on=['Accession', 'Length'], how='outer')
        df.fillna('NA', inplace=True)
        df.to_csv(f'{inputs.outpth}/final_prediction/final_prediction_summary.tsv', index=False, sep='\t')
        logger.info(f"Summarized finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction', 'final_prediction_summary.tsv')}\n\n")
    elif inputs.task == "phamer":
        phamer.run(inputs)
        logger.info(f"PhaMer finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction')}\n\n")
    elif inputs.task == "phatyp":
        phatyp.run(inputs)
        logger.info(f"PhaTYP finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction')}\n\n")
    elif inputs.task == "phagcn":
        phagcn.run(inputs)
        logger.info(f"PhaGCN finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction')}\n\n")
    elif inputs.task == "phavip":
        phavip.run(inputs)
        logger.info(f"PhaVIP finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction')}\n\n")
        logger.info('The annotation file is named as protein_annotation.tsv in phavip_supplementary folder')
    elif inputs.task == "cherry":
        cherry.run(inputs)
        logger.info(f"Cherry finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction')}\n\n")
    elif inputs.task == "contamination":
        contamination.run(inputs)
        logger.info(f"Contamination finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction')}\n\n")
    elif inputs.task == "votu":
        votu.run(inputs)
        logger.info(f"vOTU groupping finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction')}\n\n")
    elif inputs.task == "tree":
        tree.run(inputs)
        logger.info(f"Building tree finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction')}\n\n")
    else:
        logger.error(f"Task {inputs.task} is not supported, please check the help message.")
    

    return


if __name__ == "__main__":
    main()
