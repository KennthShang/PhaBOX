PhaSUIT is a python library for phage-related tasks, including phage identification, taxonomy classification, host prediction and lifestyle prediction. We integrate our previous published tools: PhaMer, PhaGCN, Cherry, and PhaTYP into one program and optimized the functions in these program to save computation resourse and time. We also provided an one-shot mode to run all the pipelines using one command. The Websever version of PhaSUIT will coming soon. Hope you will enjoy it.

If you have any suggestion or problem, feel free to contact me via email: jyshang2-c@my.cityu.edu.hk. Also you can open an issue under this GitHub folder.



# Citations
The papers of our PhaSUIT can be found below. If you use PhaSUIT for your research, please use the following citations: 

* PhaMer (phage identification)
```
Jiayu Shang, Xubo Tang, Ruocheng Guo, Yanni Sun, Accurate identification of bacteriophages from metagenomic data using Transformer, Briefings in Bioinformatics, 2022;, bbac258, https://doi.org/10.1093/bib/bbac258
```

* PhaGCN (taxonomy classification)
```
Jiayu Shang, Jingzhe Jiang, Yanni Sun, Bacteriophage classification for assembled contigs using graph convolutional network, Bioinformatics, Volume 37, Issue Supplement_1, July 2021, Pages i25–i33, https://doi.org/10.1093/bioinformatics/btab293
```

* Cherry (host prediction)
```
Jiayu Shang, Yanni Sun, CHERRY: a Computational metHod for accuratE pRediction of virus–pRokarYotic interactions using a graph encoder–decoder model, Briefings in Bioinformatics, 2022;, bbac182, https://doi.org/10.1093/bib/bbac182
```

* PhaTYP (Lifestyle prediction)
```
Shang, J., Tang, X., & Sun, Y. (2022). PhaTYP: Predicting the lifestyle for bacteriophages using BERT. arXiv preprint arXiv:2206.09693.
```


# Overview


## Required Dependencies
Detailed package information can be found in `websever.yaml`

If you want to use the gpu to accelerate the program please install the packages below:
* cuda
* Pytorch-gpu

Search [pytorch](https://pytorch.org/) to find the correct cuda version based on your computer


## Quick install
*Note*: we suggest you to install all the package using conda (both [miniconda](https://docs.conda.io/en/latest/miniconda.html) and [Anaconda](https://anaconda.org/) are ok).

After cloning this respository, you can use anaconda to install the **websever.yaml**. This will install all packages you need with cpu mode. The command is: `conda env create -f websever.yaml -n phasuit`


### Prepare the database and environment
Due to the limited size of the GitHub, we zip the database. Before using phasuit, you need to unpack them using the following commands.

1. When you use PhaMer at the first time
```
cd PhaSUIT/
conda env create -f websever.yaml -n phasuit
conda activate phasuit


# database
fileid="1d_6DGhN4Q-NZPKHEIo4yD4InLkP2U1rI"
filename="database.zip"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}

# inital files
fileid="1d_6DGhN4Q-NZPKHEIo4yD4InLkP2U1rI"
filename="initial_files.zip"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}

unzip database.zip 
unzip initial_files.zip

# generate intial files (this program might takes 30mins to a few hours according to the number of threads)
python init_script.py --threads [num]
```


2. Then, you only need to activate your 'phasuit' environment before using phasuit in the next time.
```
conda activate phasuit
```


## Usage

```
python main.py [--contigs INPUT_FA] [--threads NUM_THREAD][--len MINIMUM_LEN] [--rootpth ROOT_PTH] [--out OUTPUT_PTH]  [--midfolder MID_PTH] [--parampth PARAM_PTH] [--dbdir DR]
```

**Options**


      --contigs INPUT_FA
                            input fasta file
      --threads NUM_THREAD
                            Number of threads to run PhaMer (default 8)
      --len MINIMUM_LEN
                            predict only for sequence >= len bp (default 3000)3223q8                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
      --rootpth ROOT_PTH
                            The folder you want to store the outputs of PhaSUIT (default user_0/)
      --out OUTPUT_PTH
                            The output folder under rootpth. All the prediction will be stored in this folder. (default out/)
      --midfolder MID_PTH
                            The midfolder under rootpth. All the intermediate files will be stored in this folder. (default midfolder/)
      --parampth PARAM_PTH 
                            The pth you store your parameters (default parameters/)
      --dbdir DR
                            An optional path to store the database directory (default database/)


**Example**

Prediction on the example file:

    python main.py test_contigs.fa --threads 8 --len 1000 --rootpth simple_test

Then, PhaSUIT will run all the sub-functions to generate predictions under the `simple_test/out/` foder:  `phamer_prediction.csv` (phage identification), `phagcn_prediction.csv` (taxonomy classification), `cherry_prediction.csv` (host prediction), and `phatyp_prediction.csv` (lifestyle prediction). 



### Contact
If you have any questions, please email us: jyshang2-c@my.cityu.edu.hk


