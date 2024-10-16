#!/usr/bin/env python
from phabox2 import phamer, phatyp, phagcn, cherry, contamination
import argparse
import os
import pandas as pd
from  phabox2.scripts.ulity import *

__version__ = "2.0.0"


def main():
    parser = argparse.ArgumentParser(description="""Main script of PhaBOX.""")
    parser.add_argument('--task', help='Select a program to run (end_to_end, phamer, phagcn, phatyp, cherry, contamination) || (default end_to_end)',  default = 'end_to_end')
    parser.add_argument('--dbdir', help='Path of database directory || (required)',  default = 'database/')
    parser.add_argument('--outpth', help='Rootpth for the output folder || (required)', default='test_out/')
    parser.add_argument('--contigs', help='Path of the input FASTA file || (required)',  default = 'test_contigs.fa')
    parser.add_argument('--proteins', help='FASTA file of predicted proteins || (optional)')
    parser.add_argument('--midfolder', help='Midfolder for intermediate files || (optional)', default='midfolder/')
    parser.add_argument('--threads', help='Number of threads to use || (default use all the threads)', type=int, default=int(os.cpu_count()))
    parser.add_argument('--len', help='Filter the length of contigs || (default 3000)', type=int, default=3000)
    parser.add_argument('--reject', help='Reject sequences that < 10 percent proteins aligned to known phages: 0-20 (phamer) || (default 10)',  type=float, default = 10)
    parser.add_argument('--aai', help='Average amino acids identity for genus grouping: 0-100 (phagcn and cherry) || (default 75)',  type=float, default = 75)
    parser.add_argument('--share', help='Minimum shared number of proteins for genus grouping: 0-100 (phagcn and cherry) || (default 45)',  type=float, default = 45)
    parser.add_argument('--pcov', help='Protein-based coverage for genus grouping: 0-100 (phagcn and cherry) || (default 80)',  type=float, default = 80)
    parser.add_argument('--pident', help='Alignment identity for CRISPRs: 90-100 (cherry) || (default 90)',  type=float, default = 90)
    parser.add_argument('--cov', help='Alignment coverage for CRISPRs: 0-100 (cherry) || (default 90)',  type=float, default = 90)
    parser.add_argument('--blast', help='BLAST program for CRISPRs: blastn or blastn-short (cherry) || (default blastn)', default = 'blastn')
    parser.add_argument('--sensitive', help='Sensitive search for the prokaryotic genes: Y or N (contamination) || (default N)', default = 'N')
    parser.add_argument('--draw', help='Draw network examples for the query virus relationship: Y or N (phagcn and cherry) || (default N)', default = 'N')
    inputs = parser.parse_args()
    logger = get_logger()
    logger.info(f"PhaBOX2 is running with: {inputs.threads} threads!")
    
    if inputs.task == "end_to_end":
        phamer.run(inputs)
        phagcn.run(inputs)
        cherry.run(inputs)
        phatyp.run(inputs)
        logger.info(f"PhaMer finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction', 'phamer_prediction.tsv')}")
        logger.info(f"PhaGCN finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction', 'phagcn_prediction.tsv')}")
        logger.info(f"Cherry finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction', 'cherry_prediction.tsv')}")
        logger.info(f"PhaTYP finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction', 'phatyp_prediction.tsv')}")
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
        logger.info(f"PhaMer finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction', 'phamer_prediction.tsv')}\n\n")
    elif inputs.task == "phatyp":
        phatyp.run(inputs)
        logger.info(f"PhaTYP finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction', 'phatyp_prediction.tsv')}\n\n")
    elif inputs.task == "phagcn":
        phagcn.run(inputs)
        logger.info(f"PhaGCN finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction', 'phagcn_prediction.tsv')}\n\n")
    elif inputs.task == "cherry":
        cherry.run(inputs)
        logger.info(f"Cherry finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction', 'cherry_prediction.tsv')}\n\n")
    elif inputs.task == "contamination":
        contamination.run(inputs)
        logger.info(f"Contamination finished! please check the results in {os.path.join(inputs.outpth, 'final_prediction', 'contamination.tsv')}\n\n")
    else:
        logger.error(f"Task {inputs.task} is not supported, please check the help message.")

    return


if __name__ == "__main__":
    main()