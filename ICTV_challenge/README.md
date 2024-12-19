## Reproduce the results

1. Install the phabox2 with the [guildline](https://github.com/KennthShang/PhaBOX/wiki)
2. Run the PhaGCN program using codes: `phabox2 --task phagcn -d phabox_db_v2 -o ICTV_Challenge --contigs ICTV_Challenge.fa`
3. Convert the format using the script **convert_original_results_to_temperate_format.py**

## Brief description of the methodology


The description of the methods can be found via [PhaBOX implementation](https://phage.ee.cityu.edu.hk/implement)
The papers are: PMID 34252923, 36464489, and 37641717


In short, our method is based on protein-similarity network for classification. In the network, the nodes are viruses (both reference and query sequence), and the edges are the similarity between viruses, such as shared common protein cluster, average amino acids identity, and shared sequence fragment (overall coverage). Then, the algorithm will be classified based on the information and give a score.

Since we found many of the viruses, especially phages, do not have genus/family level prediction when processing our real sequence data, in our method, we provided a new column **GenusCluster** showing the potential new genus clustered by the "sequence similarity".  If it is not needed, you can delete the column.
