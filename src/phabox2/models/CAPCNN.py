import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import data_parallel

class WCNN(nn.Module):
    def __init__(self, num_class=20, num_token=4, seq_len=250, kernel_nums=[256, 256, 256, 256],
                 kernel_sizes=[3, 7, 11, 15], dropout=0.5, num_fc=512):
        super(WCNN, self).__init__()

        self.num_token = num_token
        self.seq_len = seq_len
        self.num_class = num_class
        self.channel_in = 1
        self.kernel_nums = kernel_nums
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(self.channel_in, self.kernel_nums[i], (kernel_size, self.num_token)) for i, kernel_size in
             enumerate(self.kernel_sizes)])
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(sum(self.kernel_nums), num_fc)
        self.out = nn.Linear(num_fc, self.num_class)

    def forward(self, x):

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        
        return x
