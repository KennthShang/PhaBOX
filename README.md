<img src='imgs/logo.png'>

# Local version of [PhaBOX2](https://phage.ee.cityu.edu.hk) web server

This is the source code of our website [PhaBOX2](https://phage.ee.cityu.edu.hk).

PhaBOX2 is a Python library for virus-related tasks: 
1. virus identification
2. taxonomy classification
3. host prediction
4. lifestyle prediction (for prokaryotic virus). 

Table of Contents
=================
* [ ⌛️&nbsp; News](#news)
* [ 🚀&nbsp; Quick Start](#quick)
* [ ⌛️&nbsp; License ](#license)
* [ 🤵&nbsp; Team ](#team)



<a name="news"></a>
## ⌛️&nbsp; News

PhaBOX has now been upgraded to the 2.0 version!!! There are some major components, including:

  🎉 Generalize for all kinds of viruses with faster speed! But we will call it PhaBOX2, not VirBOX, for a better inheritance ：) 

  🎉 Provide a more comprehensive taxonomy classification (latest ICTV 2024) and complete taxonomy lineage

  🎉 Provide a genus-level clustering for potential new genus (genus-level vOTU)!

  🎉 Provide a protein annotation function!

  🎉 Provide a contamination and prophage detection module!

  🎉 More user-friendly commands!

The web server is still upgrading; please be patient

The following functions will be coming soon!
1. Provide a marker-searching module for the phylogenetic tree
2. Allowing the use of the host prediction function in a more flexible way (only use CRISPRs for prediction; MAGs' CRISPR detection, etc.). Currently, if you want to predict the phage-bacteria interaction via CRISPR using **your own bacterial assemblies**, please try: [https://github.com/KennthShang/CHERRY_crispr_MAG](https://github.com/KennthShang/CHERRY_crispr_MAG)

If you have further suggestions, feel free to let me know! You can post an issue or directly email me (jiayushang@cuhk.edu.hk). We welcome any suggestions.


## 🚀&nbsp; Quick Start



### Please check our [WIKI](https://github.com/KennthShang/PhaBOX/wiki) page. We provide a tutorial for you to get started quickly and understand the usage of phabox2. Hope you will enjoy it!




<a name="license"></a>

## 📘&nbsp; License
The PhaBOX pipelines are released under the terms of the [Academic Free License v3.0 License](https://choosealicense.com/licenses/afl-3.0/).


<a name="team"></a>
## 🤵&nbsp; Team

 * <b>Head of PhaBOX program</b><br/>

 | [Jiayu SHANG](https://kennthshang.github.io/)       | [Cheng PENG](https://github.com/ChengPENG-wolf)       |
|:-------------------------:|:-------------------------:|
| <img width=120/ src="imgs/mine.pic.jpg?raw=true"> | <img width=120/ src="imgs/Wolf.jpg?raw=true"> |


 * <b>Supervisor</b><br/>
 
 | [Yanni SUN](https://yannisun.github.io/)       |
|:-------------------------:|
| <img width=120/ src="imgs/yanni.png?raw=true"> |


Our groupmates also provide many useful tools for bioinformatics analysis. Please check [Yanni's Group](https://yannisun.github.io/tools.html) for further information. Hope you will like them! 
