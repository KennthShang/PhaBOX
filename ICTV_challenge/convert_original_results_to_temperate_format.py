import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--infile', type=str, default = 'phagcn_prediction.tsv')
parser.add_argument('--outfile', type=str, default = 'ICTV_classification.csv')
inputs = parser.parse_args()

infile = inputs.infile
outfile = inputs.outfile



df = pd.read_csv(infile, sep='\t')
df.rename(columns={'Accession': 'SequenceID'}, inplace=True)
df.drop(columns=['Length', 'Genus', 'Prokaryotic virus (Bacteriophages and Archaeal virus)'], inplace=True)
df['Realm (-viria)'] = 'NA'
df['Realm_score'] = 'NA'
df['Subrealm (-vira)'] = 'NA'
df['Subrealm_score'] = 'NA'
df['Kingdom (-virae)'] = 'NA'
df['Kingdom_score'] = 'NA'
df['Subkingdom (-virites)'] = 'NA'
df['Subkingdom_score'] = 'NA'
df['Phylum (-viricota)'] = 'NA'
df['Phylum_score'] = 'NA'
df['Subphylum (-viricotina)'] = 'NA'
df['Subphylum_score'] = 'NA'
df['Class (-viricetes)'] = 'NA'
df['Class_score'] = 'NA'
df['Subclass (-viricetidae)'] = 'NA'
df['Subclass_score'] = 'NA'
df['Order (-virales)'] = 'NA'
df['Order_score'] = 'NA'
df['Suborder (-virineae)'] = 'NA'
df['Suborder_score'] = 'NA'
df['Family (-viridae)'] = 'NA'
df['Family_score'] = 'NA'
df['Subfamily (-virinae)'] = 'NA'
df['Subfamily_score'] = 'NA'
df['Genus (-virus)'] = 'NA'
df['Genus_score'] = 'NA'
df['Subgenus (-virus)'] = 'NA'
df['Subgenus_score'] = 'NA'
df['Species (binomial)'] = 'NA'
df['Species_score'] = 'NA'



for idx, row in df.iterrows():
    lineage = row['Lineage']
    scores = row['PhaGCNScore']
    if scores == -1:
        continue
    for taxon, score in zip(lineage.split(';'), scores.split(';')):
        if taxon.endswith('viria'):
            df.loc[idx, 'Realm (-viria)'] = taxon.split(':')[1]
            df.loc[idx, 'Realm_score'] = score
        if taxon.endswith('vira'):
            df.loc[idx, 'Subrealm (-vira)'] = taxon.split(':')[1]
            df.loc[idx, 'Subrealm_score'] = score
        if 'Kingdom'.lower() in taxon:
            df.loc[idx, 'Kingdom (-virae)'] = taxon.split(':')[1]
            df.loc[idx, 'Kingdom_score'] = score
        if 'Subkingdom'.lower() in taxon:
            df.loc[idx, 'Subkingdom (-virites)'] = taxon.split(':')[1]
            df.loc[idx, 'Subkingdom_score'] = score
        if 'Phylum'.lower() in taxon:
            df.loc[idx, 'Phylum (-viricota)'] = taxon.split(':')[1]
            df.loc[idx, 'Phylum_score'] = score
        if 'Subphylum'.lower() in taxon:
            df.loc[idx, 'Subphylum (-viricotina)'] = taxon.split(':')[1]
            df.loc[idx, 'Subphylum_score'] = score
        if 'Class'.lower() in taxon:
            df.loc[idx, 'Class (-viricetes)'] = taxon.split(':')[1]
            df.loc[idx, 'Class_score'] = score
        if 'Subclass'.lower() in taxon:
            df.loc[idx, 'Subclass (-viricetidae)'] = taxon.split(':')[1]
            df.loc[idx, 'Subclass_score'] = score
        if 'Order'.lower() in taxon:
            df.loc[idx, 'Order (-virales)'] = taxon.split(':')[1]
            df.loc[idx, 'Order_score'] = score
        if 'Suborder'.lower() in taxon:
            df.loc[idx, 'Suborder (-virineae)'] = taxon.split(':')[1]
            df.loc[idx, 'Suborder_score'] = score
        if 'Family'.lower() in taxon:
            df.loc[idx, 'Family (-viridae)'] = taxon.split(':')[1]
            df.loc[idx, 'Family_score'] = score
        if 'Subfamily'.lower() in taxon:
            df.loc[idx, 'Subfamily (-virinae)'] = taxon.split(':')[1]
            df.loc[idx, 'Subfamily_score'] = score
        if 'Genus'.lower() in taxon:
            df.loc[idx, 'Genus (-virus)'] = taxon.split(':')[1]
            df.loc[idx, 'Genus_score'] = score
        if 'Subgenus'.lower() in taxon:
            df.loc[idx, 'Subgenus (-virus)'] = taxon.split(':')[1]
            df.loc[idx, 'Subgenus_score'] = score
        if 'Species'.lower() in taxon:
            df.loc[idx, 'Species (binomial)'] = taxon.split(':')[1]
            df.loc[idx, 'Species_score'] = score



cluster = df['GenusCluster']
df.drop(columns=['Lineage', 'PhaGCNScore', 'GenusCluster'], inplace=True)
df['GenusCluster'] = cluster
df.to_csv(outfile, index=False)
