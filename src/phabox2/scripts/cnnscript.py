import  os
import  sys
import  numpy as np
import  torch
import  torch.utils.data as Data
from    models.CAPCNN import WCNN
from    Bio import SeqIO
from    Bio.SeqRecord import SeqRecord   
from    torch import nn
from    torch.nn import functional as F
from    torch import optim




def create_fragments(inpth, file_name):
    new_record = []
    for record in SeqIO.parse(f'{inpth}/{file_name}', "fasta"):
        seq = record.seq
        if len(seq) > 2000:
            for i in range(0, len(seq), 2000):
                if i + 2000 > len(seq):
                    new_seq = str(seq[-2000:])
                    new_record.append(new_seq)
                    break
                
                new_seq = str(seq[i:i+2000])
                new_record.append(new_seq)
        else:
            print("error length < 2000bp")
            print(record.description)
            
            
    return new_record


def return_kmer_vocab():
    k_list = ["A", "C", "G", "T"]
    nucl_list = ["A", "C", "G", "T"]


    for i in range(2):
        tmp = []
        for item in nucl_list:
            for nucl in k_list:
                tmp.append(nucl+item)
        k_list = tmp


    int_to_vocab = {ii: word for ii, word in enumerate(k_list)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int

def encode(data, vocab_to_int):
    feature = []
    for read in data:
        int_read = []
        for i in range(len(read)):
            if i + 3 > len(read):
                break
                
            # int_read.append(vocab_to_int[read[i:i+3]]) # Nov 16th
            try:
                int_read.append(vocab_to_int[read[i:i+3]])
            except:
                int_read.append(64) 
        
        
        if len(int_read) != 1998:
            print("error length")
            exit(1)
        
        feature.append(int_read)
    return np.array(feature)

def create_cnndataset(data):
    if data.reshape(-1).shape[0] == 1998:
        data = data.reshape(1, 1998)
        
    label = np.zeros(len(data))    
    data = np.c_[data, label]

    return data




def load_cnnmodel(parampth):
    cnn = WCNN(num_token=100,num_class=18,kernel_sizes=[3,7,11,15], kernel_nums=[256, 256, 256, 256])
    pretrained_dict=torch.load(f'{parampth}/CNN_Params.pkl', map_location='cpu')
    cnn.load_state_dict(pretrained_dict)

    torch_embeds = nn.Embedding(65, 100)
    tmp = torch.load(f'{parampth}/CNN_Embed.pkl', map_location='cpu')
    old_weight = tmp['weight']
    padding = torch.zeros((1, 100))
    new_weight = torch.cat([tmp['weight'], padding])
    torch_embeds.weight = torch.nn.Parameter(new_weight)
    torch_embeds.weight.requires_grad=False

    return cnn, torch_embeds


