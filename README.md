<img src='imgs/logo.jpg'>


# Local version of [PhaBOX2](https://phage.ee.cityu.edu.hk) web server

[![PhaBOX Server](https://img.shields.io/badge/PhaBOX-Webserver-brightgreen)](http://phage.ee.cityu.edu.hk/)
[![GitHub License](https://img.shields.io/github/license/KennthShang/PhaBOX)](https://github.com/KennthShang/PhaBOX/blob/main/LICENSE.md)
[![BioConda Install](https://img.shields.io/conda/dn/bioconda/phabox.svg?style=flag&label=BioConda%20install)](https://anaconda.org/bioconda/phabox) 
[![PhaBOXv1](https://img.shields.io/static/v1.svg?label=PhaBOX_v1&message=bioadv/vbad101&color=blue)](https://doi.org/10.1093/bioadv/vbad101)
[![PhaMer](https://img.shields.io/static/v1.svg?label=PhaMer&message=bib/bbac258&color=blue)](https://doi.org/10.1093/bib/bbac258)
[![PhaGCN](https://img.shields.io/static/v1.svg?label=PhaGCN&message=bioinformatics/btab293&color=blue)](https://doi.org/10.1093/bioinformatics/btab293)
[![PhaGCN2](https://img.shields.io/static/v1.svg?label=PhaGCN2&message=bib/bbac505&color=blue)](https://doi.org/10.1093/bioinformatics/btab293)
[![PhaTYP](https://img.shields.io/static/v1.svg?label=PhaTYP&message=bib/bbac487&color=blue)](https://doi.org/10.1093/bib/bbac487)
[![CHERRY](https://img.shields.io/static/v1.svg?label=CHERRY&message=bib/bbac182&color=blue)](https://doi.org/10.1093/bib/bbac182)
[![PhaVIP](https://img.shields.io/static/v1.svg?label=PhaVIP&message=bioinformatics/btad229&color=blue)](https://doi.org/10.1093/bioinformatics/btad229)

This is the source code of our website [PhaBOX2](https://phage.ee.cityu.edu.hk), consisting of the latest updates of our previously released programs: PhaMer, PhaGCN, CHERRY/HostG, and PhaTYP.

Please download this local version for large-scale sequencing data and use more threads to speed up the program.

**If you like our tool, please give us a star on GitHub. This will provide power for our ongoing maintenance, thanks！**

<a name="news"></a>
## ⌛️&nbsp; News

PhaBOX has now been upgraded to the 2.0 version with faster speed!

There are some major components, including:

  🎉 Generalized for all kinds of viruses; more than just bacteriophage

  🎉 Virus identification (latest PhaMer)

  🎉 Taxonomy classification (latest PhaGCN)

  🎉 Host prediction (latest CHERRY/HostG)

  🎉 Lifestyle prediction (latest PhaTYP)

  🎉 Contamination/provirus detection

  🎉 vOTU grouping

  🎉 Phylogenetic tree based on marker genes

  🎉 Viral protein annotation

  🎉 All the databases are updated to the latest ICTV 2024 release

If you have any more suggestions, feel free to let me know! We consider long-term maintenance PhaBOX and adding modules according to your needs


You can post an issue or directly email me (jiayushang@cuhk.edu.hk). We welcome any suggestions.

<a name="quick"></a>
## 🚀&nbsp; Quick Start
> [!IMPORTANT]
> If you are a new user, please check our [WIKI](https://github.com/KennthShang/PhaBOX/wiki) page. We provide a tutorial to help you get started quickly and understand how to use PhaBOX2. We hope you will enjoy it!

If you are familiar with the PhaBOX2, please check our [Update log](https://github.com/KennthShang/PhaBOX/wiki/Update-logs). We may have some updates to the program to make it more useful. If you want to use the latest version, please also [upgrade your PhaBOX2](https://github.com/KennthShang/PhaBOX/wiki#upgrading-phabox)


## 🚀&nbsp; The Most Recent Update Logs
### 2.1.12 June 30, 2025
To use this version, please also update your phabox database as below:
```
wget https://github.com/KennthShang/PhaBOX/releases/download/2.1.0/phabox_db_v2_1.zip
```

> [!IMPORTANT]
>  New functions were added to the taxonomy classification (--phagcn), host prediction (--cherry), and phylogenetic tree (--tree)
>  The taxonomy classification task allows classification of viruses at the species level.
>  The Host prediction task allows the user to provide the GTDB-tk taxonomy file (gtdbtk.tsv) for high-confidence filtering and labeling (--bgtdb gtdbtk.tsv).
>  The phylogenetic tree task allows running MSA and tree construction without genes from the database (--msadb Y or N).
>  A marker_stats.tsv file (information of marker genes in the database) will be copied to the final_prediction folder once the tree task is finished.


### 2.1.11 March 6, 2025

> [!IMPORTANT]
> New functions were added to the cherry host prediction task
> We also adjust the host prediction logic as below:
> 1. CRISPRs from MAGs (if MAGs were provided)
> 2. BLASTN (prophage) from MAGs (if MAGs were provided)
> 3. Protein organization compared to the viruses in the database
> 4. Kmer frequency compared to MAGs (if MAGs were provided)
> 5. CRISPRs from the database


New added parameters in `--task cherry`
```
--prophage
     Minimum alignment length for estimating potential prophage || default: 1000 || range from 0 to 100000
```


### 2.1.10 Dec 26, 2024

<a name="license"></a>

## 📘&nbsp; License
The PhaBOX pipelines are released under the terms of the [Academic Free License v3.0 License](https://choosealicense.com/licenses/afl-3.0/).

