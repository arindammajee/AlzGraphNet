# Import necessary libraries
import numpy as np
from sklearn import neighbors
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

config = {
    'height' : 256,
    'width' : 256,
    'depth' : 64
}
patch_config = {
    'height' : 8,
    'width' : 8,
    'depth' : 8
}

#no_of_nodes = int((config['height']/patch_config['height'])*(config['width']/patch_config['width'])*(config['depth']/patch_config['depth']))
no_of_nodes = 64

class MRI3DConvolution(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer1 = self._conv_layer_set(1, 8)
        self.conv_layer2 = self._conv_layer_set(8, 32)
        #self.conv_layer3 = self._conv_layer_set(32, 128)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3,3,3), padding=1, stride=2),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2))
        )
        return conv_layer
    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        print(out.shape)
        #out = self.conv_layer3(out)
        out = out.view(out.size(0), 1, 16, 16, 128)
        return out


class GraphBuilding(nn.Module):
    def __init__(self, config=config, patch_config=patch_config):
        super(GraphBuilding, self).__init__()
        
        self.config = config
        self.patch_config = patch_config


    def forward(self, image, name='knn', n = 8):
        patches = []

        for i in range(int(image.shape[-3]/self.patch_config['height'])):
            for j in range(int(image.shape[-2]/self.patch_config['width'])):
                for k in range(int(image.shape[-1]/self.patch_config['depth'])):
                    patch = image[:, :, i*self.patch_config['height']:(i+1)*self.patch_config['height'], j*self.patch_config['width']:(j+1)*self.patch_config['width'], k*self.patch_config['depth']:(k+1)*self.patch_config['depth']]
                    patch = patch.reshape(image.shape[0], 1, -1)
                    patches.append(patch)

        #print(len(patches))
        
        features = torch.cat(patches, dim=1)
        adjacency = []
        #print(features.shape)
        for i in range(features.shape[0]):
            if name == 'knn':
                feature = features[i].cpu().detach().numpy()
                adj = neighbors.kneighbors_graph(feature, n_neighbors = n).toarray()
                adj = (adj + np.transpose(adj)) / 2
                adj = torch.as_tensor(adj, dtype=torch.float32, device=torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')
                adj = adj.reshape(1, adj.shape[0], adj.shape[1])
                adjacency.append(adj)
        
        adjacency = torch.cat(adjacency, dim=0)
        
        return features, adjacency



class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, device='cuda:0', bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features)).to(device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        #print(x.shape)
        x = torch.matmul(x, self.weight)
        #print(x.shape)
        output = torch.bmm(adj, x)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MRIGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MRIGCN, self).__init__()
        torch.manual_seed(12345)

        #self.conv_layer1 = self._conv_layer_set(1, 8)
        self.graph_input_layer = GraphBuilding()
        self.conv = MRI3DConvolution()
        self.gc1 = GraphConvolutionLayer(nfeat, nhid)
        self.gc2 = GraphConvolutionLayer(nhid, nhid//2)
        self.gc3 = GraphConvolutionLayer(nhid//2, nhid//4)
        self.fc1 = nn.Linear(no_of_nodes*nhid//4, nhid)
        self.fc2 = nn.Linear(nhid, nhid//4)
        self.fc3 = nn.Linear(nhid//4, nclass)
        self.dropout = dropout

        
    def forward(self, image):
        #x = F.relu(self.conv_layer1(image))
        #print(image.shape)
        x = self.conv(image)
        #print(x.shape)
        features, adj = self.graph_input_layer(x)
        #print(features.shape, adj.shape)
        x = F.relu(self.gc1(features, adj))
        x = self.gc2(x, adj)
        x = self.gc3(x, adj)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout) #, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(x, dim=1)
    
    """
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
    """