import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geo_nn


class DirectionalGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DirectionalGraphConvolution, self).__init__()
        self.conv = geo_nn.GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = x.to(torch.float32)
        x = F.relu(self.conv(x, edge_index=edge_index, edge_weight=edge_weight))
        return x


class DirectionalGraphNeuralNetworkBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DirectionalGraphNeuralNetworkBlock, self).__init__()
        self.graph_conv = DirectionalGraphConvolution(
            in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.graph_conv(x, edge_index, edge_weight)
        return x


class DirectionalGraphNeuralNetwork(nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_classes):
        super(DirectionalGraphNeuralNetwork, self).__init__()

        # Directional Graph Convolution Blocks
        self.block1 = DirectionalGraphNeuralNetworkBlock(num_node_features, 64)
        self.block2 = DirectionalGraphNeuralNetworkBlock(64, 64)

        self.block3 = DirectionalGraphNeuralNetworkBlock(64, 128)
        self.block4 = DirectionalGraphNeuralNetworkBlock(128, 128)

        self.block5 = DirectionalGraphNeuralNetworkBlock(128, 256)
        self.block6 = DirectionalGraphNeuralNetworkBlock(256, 256)

        self.block7 = DirectionalGraphNeuralNetworkBlock(256, 512)
        self.block8 = DirectionalGraphNeuralNetworkBlock(512, 512)

        # Global Average Pooling
        self.global_pooling = geo_nn.global_mean_pool

        # Fully connected layers for classification with added dropout
        self.fc1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, edge_index, edge_weight, batch):
        # Directional Graph Convolution Blocks
        x = self.block1(x, edge_index, edge_weight)
        x = self.block2(x, edge_index, edge_weight)

        x = self.block3(x, edge_index, edge_weight)
        x = self.block4(x, edge_index, edge_weight)

        x = self.block5(x, edge_index, edge_weight)
        x = self.block6(x, edge_index, edge_weight)

        x = self.block7(x, edge_index, edge_weight)
        x = self.block8(x, edge_index, edge_weight)

        # Global Average Pooling
        x = self.global_pooling(x, batch=batch)
        x = x.to(torch.float32)

        # Fully connected layers for classification with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
