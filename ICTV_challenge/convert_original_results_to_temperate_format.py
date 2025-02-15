import pandas as pd
import argparse
import json

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--infile', help="Path to the folder containing original results", type=str, default = 'phagcn_prediction.tsv')
parser.add_argument('--outfile', help="Path to save the required formats", type=str, default = 'ICTV_classification.csv')
parser.add_argument('--json_file', help="Path to the json file containing valid names", type=str, default = 'MSL39v4_valid_names_per_taxa.json')
inputs = parser.parse_args()

infile = inputs.infile
outfile = inputs.outfile
json_file = inputs.json_file


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



with open(json_file, "r") as f:
    valid_names = json.load(f)

def check_validity(name, taxa, valid_names):
    if name not in valid_names[taxa]:
        return False
    return True

def find_validity(name, taxa, valid_names):
    if taxa == 'Realm':
        name = name.split('viria')[0]
    if taxa == 'Subrealm':
        name = name.split('vira')[0]
    if taxa == 'Kingdom':
        name = name.split('virae')[0]
    if taxa == 'Subkingdom':
        name = name.split('virites')[0]
    if taxa == 'Phylum':
        name = name.split('viricota')[0]
    if taxa == 'Subphylum':
        name = name.split('viricotina')[0]
    if taxa == 'Class':
        name = name.split('viricetes')[0]
    if taxa == 'Subclass':
        name = name.split('viricetidae')[0]
    if taxa == 'Order':
        name = name.split('virales')[0]
    if taxa == 'Suborder':
        name = name.split('virineae')[0]
    if taxa == 'Family':
        name = name.split('viridae')[0]
    if taxa == 'Subfamily':
        name = name.split('virinae')[0]
    if taxa == 'Genus':
        name = name.split('virus')[0]
    if taxa == 'Subgenus':
        name = name.split('virus')[0]

    for exist_taxa in valid_names:
        for exist_name in valid_names[exist_taxa]:
            if exist_name.startswith(name):
                return exist_name, exist_taxa
    return 'NA', 'NA'

def rename(valid_name, valid_taxa, score, idx):
    if valid_taxa == 'Realm':
        df.loc[idx, 'Realm (-viria)'] = valid_name
        df.loc[idx, 'Realm_score'] = score
    if valid_taxa == 'Subrealm':
        df.loc[idx, 'Subrealm (-vira)'] = valid_name
        df.loc[idx, 'Subrealm_score'] = score
    if valid_taxa == 'Kingdom':
        df.loc[idx, 'Kingdom (-virae)'] = valid_name
        df.loc[idx, 'Kingdom_score'] = score
    if valid_taxa == 'Subkingdom':
        df.loc[idx, 'Subkingdom (-virites)'] = valid_name
        df.loc[idx, 'Subkingdom_score'] = score
    if valid_taxa == 'Phylum':
        df.loc[idx, 'Phylum (-viricota)'] = valid_name
        df.loc[idx, 'Phylum_score'] = score
    if valid_taxa == 'Subphylum':
        df.loc[idx, 'Subphylum (-viricotina)'] = valid_name
        df.loc[idx, 'Subphylum_score'] = score
    if valid_taxa == 'Class':
        df.loc[idx, 'Class (-viricetes)'] = valid_name
        df.loc[idx, 'Class_score'] = score
    if valid_taxa == 'Subclass':
        df.loc[idx, 'Subclass (-viricetidae)'] = valid_name
        df.loc[idx, 'Subclass_score'] = score
    if valid_taxa == 'Order':
        df.loc[idx, 'Order (-virales)'] = valid_name
        df.loc[idx, 'Order_score'] = score
    if valid_taxa == 'Suborder':
        df.loc[idx, 'Suborder (-virineae)'] = valid_name
        df.loc[idx, 'Suborder_score'] = score
    if valid_taxa == 'Family':
        df.loc[idx, 'Family (-viridae)'] = valid_name
        df.loc[idx, 'Family_score'] = score
    if valid_taxa == 'Subfamily':
        df.loc[idx, 'Subfamily (-virinae)'] = valid_name
        df.loc[idx, 'Subfamily_score'] = score
    if valid_taxa == 'Genus':
        df.loc[idx, 'Genus (-virus)'] = valid_name
        df.loc[idx, 'Genus_score'] = score
    if valid_taxa == 'Subgenus':
        df.loc[idx, 'Subgenus (-virus)'] = valid_name
        df.loc[idx, 'Subgenus_score'] = score

    
    return df

for idx, row in df.iterrows():
    lineage = row['Lineage']
    scores = row['PhaGCNScore']
    if scores == -1:
        continue
    for taxon, score in zip(lineage.split(';'), scores.split(';')):
        if taxon.endswith('viria'):
            if check_validity(taxon.split(':')[1], 'Realm', valid_names):
                df.loc[idx, 'Realm (-viria)'] = taxon.split(':')[1]
                df.loc[idx, 'Realm_score'] = score
            else:
                valid_name, valid_taxa = find_validity(taxon.split(':')[1], 'Realm', valid_names)
                if valid_taxa != 'NA':
                    rename(valid_name, valid_taxa, score, idx)
        
        if taxon.endswith('vira'):
            if check_validity(taxon.split(':')[1], 'Subrealm', valid_names):
                df.loc[idx, 'Subrealm (-vira)'] = taxon.split(':')[1]
                df.loc[idx, 'Subrealm_score'] = score
            else:
                valid_name, valid_taxa = find_validity(taxon.split(':')[1], 'Subrealm', valid_names)
                if valid_taxa != 'NA':
                    rename(valid_name, valid_taxa, score, idx)
        if 'kingdom' == taxon.split(':')[0]:
            if check_validity(taxon.split(':')[1], 'Kingdom', valid_names):
                df.loc[idx, 'Kingdom (-virae)'] = taxon.split(':')[1]
                df.loc[idx, 'Kingdom_score'] = score
            else:
                valid_name, valid_taxa = find_validity(taxon.split(':')[1], 'Kingdom', valid_names)
                if valid_taxa != 'NA':
                    rename(valid_name, valid_taxa, score, idx)
        if 'subkingdom' == taxon.split(':')[0]:
            if check_validity(taxon.split(':')[1], 'Subkingdom', valid_names):
                df.loc[idx, 'Subkingdom (-virites)'] = taxon.split(':')[1]
                df.loc[idx, 'Subkingdom_score'] = score
            else:
                valid_name, valid_taxa = find_validity(taxon.split(':')[1], 'Subkingdom', valid_names)
                if valid_taxa != 'NA':
                    rename(valid_name, valid_taxa, score, idx)
        if 'phylum' == taxon.split(':')[0]:
            if check_validity(taxon.split(':')[1], 'Phylum', valid_names):
                df.loc[idx, 'Phylum (-viricota)'] = taxon.split(':')[1]
                df.loc[idx, 'Phylum_score'] = score
            else:
                valid_name, valid_taxa = find_validity(taxon.split(':')[1], 'Phylum', valid_names)
                if valid_taxa != 'NA':
                    rename(valid_name, valid_taxa, score, idx)
        if 'subphylum' == taxon.split(':')[0]:
            if check_validity(taxon.split(':')[1], 'Subphylum', valid_names):
                df.loc[idx, 'Subphylum (-viricotina)'] = taxon.split(':')[1]
                df.loc[idx, 'Subphylum_score'] = score
            else:
                valid_name, valid_taxa = find_validity(taxon.split(':')[1], 'Subphylum', valid_names)
                if valid_taxa != 'NA':
                    rename(valid_name, valid_taxa, score, idx)
        if 'class' == taxon.split(':')[0]:
            if check_validity(taxon.split(':')[1], 'Class', valid_names):
                df.loc[idx, 'Class (-viricetes)'] = taxon.split(':')[1]
                df.loc[idx, 'Class_score'] = score
            else:
                valid_name, valid_taxa = find_validity(taxon.split(':')[1], 'Class', valid_names)
                if valid_taxa != 'NA':
                    rename(valid_name, valid_taxa, score, idx)
        if 'subclass' == taxon.split(':')[0]:
            if check_validity(taxon.split(':')[1], 'Subclass', valid_names):
                df.loc[idx, 'Subclass (-viricetidae)'] = taxon.split(':')[1]
                df.loc[idx, 'Subclass_score'] = score
            else:
                valid_name, valid_taxa = find_validity(taxon.split(':')[1], 'Subclass', valid_names)
                if valid_taxa != 'NA':
                    rename(valid_name, valid_taxa, score, idx)
        if 'order' == taxon.split(':')[0]:
            if check_validity(taxon.split(':')[1], 'Order', valid_names):
                df.loc[idx, 'Order (-virales)'] = taxon.split(':')[1]
                df.loc[idx, 'Order_score'] = score
            else:
                valid_name, valid_taxa = find_validity(taxon.split(':')[1], 'Order', valid_names)
                if valid_taxa != 'NA':
                    rename(valid_name, valid_taxa, score, idx)
        if 'suborder' == taxon.split(':')[0]:
            if check_validity(taxon.split(':')[1], 'Suborder', valid_names):
                df.loc[idx, 'Suborder (-virineae)'] = taxon.split(':')[1]
                df.loc[idx, 'Suborder_score'] = score
            else:
                valid_name, valid_taxa = find_validity(taxon.split(':')[1], 'Suborder', valid_names)
                if valid_taxa != 'NA':
                    rename(valid_name, valid_taxa, score, idx)
        if 'family' == taxon.split(':')[0]:
            if check_validity(taxon.split(':')[1], 'Family', valid_names):
                df.loc[idx, 'Family (-viridae)'] = taxon.split(':')[1]
                df.loc[idx, 'Family_score'] = score
            else:
                valid_name, valid_taxa = find_validity(taxon.split(':')[1], 'Family', valid_names)
                if valid_taxa != 'NA':
                    rename(valid_name, valid_taxa, score, idx)
        if 'subfamily' == taxon.split(':')[0]:
            if check_validity(taxon.split(':')[1], 'Subfamily', valid_names):
                df.loc[idx, 'Subfamily (-virinae)'] = taxon.split(':')[1]
                df.loc[idx, 'Subfamily_score'] = score
            else:
                valid_name, valid_taxa = find_validity(taxon.split(':')[1], 'Subfamily', valid_names)
                if valid_taxa != 'NA':
                    rename(valid_name, valid_taxa, score, idx)
        if 'genus' == taxon.split(':')[0]:
            if check_validity(taxon.split(':')[1], 'Genus', valid_names):
                df.loc[idx, 'Genus (-virus)'] = taxon.split(':')[1]
                df.loc[idx, 'Genus_score'] = score
            else:
                valid_name, valid_taxa = find_validity(taxon.split(':')[1], 'Genus', valid_names)
                if valid_taxa != 'NA':
                    rename(valid_name, valid_taxa, score, idx)
        if 'subgenus' == taxon.split(':')[0]:
            if check_validity(taxon.split(':')[1], 'Subgenus', valid_names):
                df.loc[idx, 'Subgenus (-virus)'] = taxon.split(':')[1]
                df.loc[idx, 'Subgenus_score'] = score
            else:
                valid_name, valid_taxa = find_validity(taxon.split(':')[1], 'Subgenus', valid_names)
                if valid_taxa != 'NA':
                    rename(valid_name, valid_taxa, score, idx)
        if 'species' == taxon.split(':')[0]:
            df.loc[idx, 'Species (binomial)'] = taxon.split(':')[1]
            df.loc[idx, 'Species_score'] = score





df.drop(columns=['Lineage', 'PhaGCNScore', 'GenusCluster'], inplace=True)
df.to_csv(outfile, index=False)
