import  torch
from    torch import nn
from    torch.nn import functional as F
import numpy as np
import random

def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1./ (1-rate))

    return out


def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)

    return res


class GraphConvolution(nn.Module):


    def __init__(self, node_dim,input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False,
                 learn_weight=True):
        super(GraphConvolution, self).__init__()


        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.edge_w = nn.Parameter(torch.rand(node_dim, node_dim))
        self.bias = None
        self.learn_weight=True
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))


    def forward(self, inputs):
        # print('inputs:', inputs)
        x, support = inputs

        if self.training:
            x = F.dropout(x, self.dropout)

        xw = torch.mm(x, self.weight)
        support_w = support
        #support_w = torch.mm(support, self.edge_w)
        #support_w = torch.mul(support, self.edge_w)
        #if self.learn_weight:
            #support_w = torch.mm(support, self.edge_w)
        #    support_w = torch.mul(support, self.edge_w)
        #else:
        #    support_w = support
        out = torch.mm(support_w , xw)

        if self.bias is not None:
            out += self.bias

        return self.activation(out), support




class encoder(nn.Module):
    def __init__(self, node_dim, input_dim, output_dim, num_features_nonzero):
        super(encoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_dim = 1474

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)

        # hidden = 64
        self.layers = nn.Sequential(GraphConvolution(self.node_dim, self.input_dim, self.input_dim, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=0,
                                                     is_sparse_inputs=False,
                                                     learn_weight=True),


                                    )


    def forward(self, inputs):
        x, support = inputs

        x = self.layers((x, support))
        
        return x[0]

    def l2_loss(self):

        layer = self.layers.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss


class decoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim_1, hidden_dim_2):
        super(decoder, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.model = nn.Sequential(
                        nn.Linear(self.feature_dim, self.hidden_dim_1),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim_2, 1)
                    )

        

    def forward(self, inputs):
        logit = self.model(inputs)
        return logit

