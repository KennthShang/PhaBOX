# PhaSUIT

## A sever for analysing and visualizing phage contings in metagenomic data

Bacteriophages are viruses infecting bacteria. Being key players in microbial communities, they can regulate the composition/function of microbiome by infecting their bacterial hosts and mediating gene transfer. Recently, metagenomic sequencing, which can sequence all genetic materials from various microbiome, has become a popular means for new phage discovery. However, accurate and comprehensive detection of phages from the metagenomic data remains difficult. High diversity/abundance, and limited reference genomes pose major challenges for recruiting phage fragments from metagenomic data.

In this work, we have developed an ensemble learning-based pipelines, PhaSUIT, to accurately analysis phage contigs in metagenomic data. PhaSUIT employs different types of sequence similairty features, such as protein oragnizations, protein homology, and protein-protein associations, within deep learning framework. PhaSUIT has four main functions: phage recognition, taxonomic classification, phage lifestyle prediction, and phage host prediction, corresponding to four subprograms: PhaMer, PhaGCN, PhaTYP, and CHERRY. We optimized the functions in our subprograms to save computational resourses and time. Thus, users can analyze phage contigs in one-shot mode using PhaSUIT.

More importantly, PhaSUIT does not make predictions in a black box, but interpretable. For each predicted phage contig, we visualized the basic components of PhaSUIT, such as its relationships to other phages and protein homology, to show evidence for generating predictions.



tab(Learn more) tab(Go to use it)

Reminder:

The following browsers are supported/tested by this website:

- Windows: Chrome, Firefox, Edges
- Mac: Chrome, Firefox, Safari
- Linux: Chrome, Firefox



If you find our work useful for your research work, please cite the corresponding subprogram listed below. The citation of PhaSUIT is not yet available.

* PhaMer (phage identification)

```
Jiayu Shang, Xubo Tang, Ruocheng Guo, Yanni Sun, Accurate identification of bacteriophages from metagenomic data using Transformer, Briefings in Bioinformatics, Volume 23, Issue 4, July 2022, bbac258, https://doi.org/10.1093/bib/bbac258
```

* PhaGCN (taxonomy classification)

```
Jiayu Shang, Jingzhe Jiang, Yanni Sun, Bacteriophage classification for assembled contigs using graph convolutional network, Bioinformatics, Volume 37, Issue Supplement_1, July 2021, Pages i25–i33, https://doi.org/10.1093/bioinformatics/btab293
```

* Cherry (host prediction)

```
Jiayu Shang, Yanni Sun, CHERRY: a Computational metHod for accuratE pRediction of virus–pRokarYotic interactions using a graph encoder–decoder model, Briefings in Bioinformatics, Volume 23, Issue 5, September 2022, bbac182, https://doi.org/10.1093/bib/bbac182
```

* PhaTYP (Lifestyle prediction)

```
Shang, J., Tang, X., & Sun, Y. (2022). PhaTYP: Predicting the lifestyle for bacteriophages using BERT. arXiv preprint arXiv:2206.09693.
```


# 
