# Components of the PhaSUIT

The overall workflow of PhaSUIT is presented in terms of data collection and curation, feature encoding, and model prediction as shown in the figure below:

![image-20221112164507467](/Users/jyshang2/Library/Application Support/typora-user-images/image-20221112164507467.png)





### PhaMer

<img src="/Users/jyshang2/Library/Application Support/typora-user-images/image-20221113111709788.png" alt="image-20221113111709788" style="zoom:40%;" />

PhaMer employ a contextualized embedding model from natural language processing (NLP) to learn protein-associated patterns in phages. Specifically, by converting a sequence into a sentence composed of protein-based tokens, we employ the embedding model to learn both the protein composition and also their associations in phage sequences. 

**(Fig. 1A)** First, we construct the vocabulary containing protein-based tokens, which are essentially protein clusters with high similarities. Then, we apply DIAMOND BLASTP [1] to record the presence of tokens in training phage sequences. 

**(Fig. 1B)** Then, the tokens and their positions will be fed into Transformer  for contextual-aware embedding. The embedding layer and the self-attention mechanism in Transformer enable the model to learn the importance of each protein cluster and the protein-protein associations. In addition, by using the phages' host genomes as the negative samples in the training data, the model can learn from the hard cases and thus is more likely to achieve high precision in real data. 

PhaMer can directly use the whole sequences for training, avoiding the bias of segmentation. We rigorously tested PhaMer on multiple independent datasets covering different scenarios including the RefSeq dataset, short contigs, simulated metagenomic data, mock metagenomic data, and the public IMG/VR dataset. We compared PhaMer with four competitive learning-based tools and one alignment-based tool (VirSorter) based on a third-party review [2]. The results show that PhaMer competes favorably against the existing tools. 



### PhaGCN

<img src="/Users/jyshang2/Library/Application Support/typora-user-images/image-20221113112825839.png" alt="image-20221113112825839" style="zoom:40%;" />

Given the enormous diversity of phages and the sheer amount of unlabeled phages, we formulate the phage classification problem as a semi-supervised learning problem. We choose the GCN as the learning model and combine the strength of both the alignment-based and the learning-based methods. 

The input to PhaGCN is a knowledge graph. There are two key components in the knowledge graph: node encoding and edge construction. The node is a numerical vector learned from contigs using a CNN. The edge encodes features from both the sequence similarity and the organization of genes.  

Fig. 1 contains the major components for node and edge construction. 

* **(Fig. 1 A1-A3)** To encode a sequence using a node, a pre-trained convolutional neural network (CNN) is adopted to capture features from the input DNA sequence. The CNN model is trained to convert proximate substrings into vectors of high similarity. 
* **(Fig. 1 B1-B4)** The edge construction consists of several steps. We employ a greedy search algorithm to find the best BLASTP results (E-value less than 1e-5) between the translated proteins from the contigs and the database . 
* **(Fig. 1 B5)** Then the Markov clustering algorithm (MCL) is applied to generate protein clusters from the BLASTP result . 
* **(Fig. 1 B6-B7)** Based on the results of BLASTP (sequence similarity) and MCL (shared proteins), we define the edges between sequences (contigs and reference genomes) using two metrics: P_weight and E_weight. 
* **(Fig. 1 C1)** By combining the node’s features and edges, we construct the knowledge graph  and feed it to the GCN to classify new phage contigs.

We compared PhaGCN with three state-of-the-art models specifically designed for phage classification: Phage Orthologous Groups (POG), vConTACT 2.0, and ClassiPhage. The experimental results demonstrated that PhaGCN outperforms other popular methods in classifying new phage contigs.



### PhaTYP

<img src="/Users/jyshang2/Library/Application Support/typora-user-images/image-20221113114807661.png" alt="image-20221113114807661" style="zoom:40%;" />



PhaTYP is a BERT-based model that learns the protein composition and associations from phage genomes to classify the lifestyles of phages.

To address the difficulties of classifying incomplete genomes with limited training data, we divide the lifestyle classification into two tasks: a self-supervised learning task (Fig. 1 A) and a fine-tuning task (Fig. 1 B). 

* **(Fig. 1A)** To circumvent the problem that only a limited number of phages have lifestyle annotations, we applied self-supervised learning to learn protein association features from all the phage genomes using Masked Language Model (Masked LM), aiming to recover the original protein from the masked protein sentences. This task allows us to utilize all the phage genomes for training regardless of available lifestyle annotations. 
* **(Fig. 1B)** In the second task, we will fine-tune the Masked LM on phages with known lifestyle annotations for classification. To ensure that the model can handle short contigs, we apply data augmentation by generating fragments ranging from 100bp to 10,000bp for training. 

We evaluated PhaTYP on contigs of different lengths and contigs assembled from real metagenomic data. The benchmark results against the state-of-the-art methods show that PhaTYP not only achieves the highest performance on complete genomes but also improves the accuracy on short contigs by over 10%.

### CHERRY

<img src="/Users/jyshang2/Library/Application Support/typora-user-images/image-20221113113131587.png" alt="image-20221113113131587" style="zoom:50%;" />

CHERRY can predict the hosts' taxa (phylum to species) for newly identified viruses based on a multimodal graph. 

**(Fig. 1A)** The multimodal graph incorporates multiple types of interactions, including protein organization information between viruses, the sequence similarity between viruses and prokaryotes, and the CRISPR signals . In addition, we use k-mer frequency as the node features to enhance the learning ability. 

**(Fig. 1B)** Rather than directly using these features for prediction, we design an encoder-decoder structure to learn the best embedding for input sequences and predict the interactions between viruses and prokaryotes. The graph convolutional encoder   utilizes the topological structure of the multimodal graph and thus, features from both training and testing sequences can be incorporated to embed new node features. 

**(Fig. 1C)** Then, a link prediction decoder is adopted to estimate how likely a given virus-prokaryote pair forms a real infection. 

Another feature behind the high accuracy of CHERRY is the construction of the negative training set. The dataset for training is highly imbalanced, with the real host as the positive data and all other prokaryotes as negative data. We carefully addressed this issue using negative sampling. Instead of using a random subset of the negative set for training the model, we apply end-to-end optimization and negative sampling to automatically learn the hard cases during training. 

To demonstrate the reliability of our method, we rigorously tested CHERRY on multiple independent datasets including the RefSeq dataset, simulated short contigs, and metagenomic datasets. We compared CHERRY with WIsH, PHP, HoPhage, VPF-Class, RaFAH, HostG, vHULK, PHIST, DeepHost, PHIAF, and VHM-net. The results show that CHERRY competes favorably against the state-of-the-art tools. 





# Quick start

PhaSUIT allows users to copy-and-paste or upload their interested DNA contigs in FASTA format in the ***Piplines*** pages. When predicting query contigs, PhaSUIT server provides two options for analysing users' sequences:

1. Running PhaSUIT once for all phage tasks (phage identification, taxa classification, lifestyle prediction, and host prediction). 
2. Running subprogram that users are interested in.

However, when running PhaGCN, PhaTYP, and CHERRY in the single-program mode, users should make sure their inputs sequences are all phages. Otherwise, user should run PhaMer to filter the non-phage genomes first. 



### Submitting a job

![image-20221113105459917](/Users/jyshang2/Library/Application Support/typora-user-images/image-20221113105459917.png)

1. Choose a program you want to run in red box.
2. Paste or upload your DNA sequences in the green box.
3. Set the parameters in the blue box. If you want to use the default parameters, let them blank.
4. Choose whether you want to be notified by email. If yes, tun on the button and paste you email address. Otherwise, submit your task directly.



