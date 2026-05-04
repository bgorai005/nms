import math

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


LOG_2PI = math.log(2.0 * math.pi)


def chain_adjacency(num_nodes, bidirectional=True):
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for idx in range(num_nodes - 1):
        adjacency[idx, idx + 1] = 1.0
        if bidirectional:
            adjacency[idx + 1, idx] = 1.0
    return adjacency


def chain_edge_index(num_nodes, bidirectional=True):
    adjacency = chain_adjacency(num_nodes, bidirectional=bidirectional)
    src, dst = torch.nonzero(adjacency, as_tuple=True)
    return torch.stack([src, dst], dim=0).long()


class GCN_GRU(nn.Module):
    def __init__(
        self,
        node_features=4,
        gcn_hidden=12,
        gru_hidden=96,
        num_nodes=6,
        gru_layers=1,
        gcn_dropout=0.2,
        fc_dropout=0.3,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.gcn_hidden = gcn_hidden

        self.gcn1 = GCNConv(node_features, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.bn = nn.BatchNorm1d(gcn_hidden * num_nodes)
        self.gcn_dropout = nn.Dropout(gcn_dropout)
        self.gru = nn.GRU(
            gcn_hidden * num_nodes,
            gru_hidden,
            batch_first=True,
            num_layers=gru_layers,
            dropout=gcn_dropout if gru_layers > 1 else 0.0,
        )
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(gru_hidden, 1)
        self.register_buffer("edge_index", chain_edge_index(num_nodes, bidirectional=True))

    def _batch_edge_index(self, num_graphs):
        edge_count = self.edge_index.size(1)
        edge_index = self.edge_index.repeat(1, num_graphs)
        offsets = (torch.arange(num_graphs, device=self.edge_index.device) * self.num_nodes).repeat_interleave(
            edge_count
        )
        return edge_index + offsets.unsqueeze(0)

    def forward(self, x_seq):
        batch_size, seq_len, num_nodes, num_features = x_seq.shape
        num_graphs = batch_size * seq_len

        x_flat = x_seq.reshape(num_graphs * num_nodes, num_features)
        edge_index = self._batch_edge_index(num_graphs)

        h = torch.relu(self.gcn1(x_flat, edge_index))
        h = self.gcn_dropout(h)
        h = torch.relu(self.gcn2(h, edge_index))
        h = self.gcn_dropout(h)

        h = h.reshape(batch_size * seq_len, num_nodes * self.gcn_hidden)
        h = self.bn(h)
        h = h.reshape(batch_size, seq_len, num_nodes * self.gcn_hidden)

        gru_out, _ = self.gru(h)
        last = self.fc_dropout(gru_out[:, -1, :])
        return self.fc(last).squeeze(-1)


class SequenceProbGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, input_dim)
        self.logvar_head = nn.Linear(hidden_dim, input_dim)

    def conditional_parameters(self, x_seq):
        if x_seq.size(1) < 2:
            raise ValueError("SequenceProbGRU needs at least two timesteps per sequence.")

        context = x_seq[:, :-1, :]
        targets = x_seq[:, 1:, :]
        hidden_seq, _ = self.gru(context)
        batch_size, steps, hidden_dim = hidden_seq.shape
        hidden_seq = self.bn(hidden_seq.reshape(-1, hidden_dim)).reshape(batch_size, steps, hidden_dim)
        mean = self.mean_head(hidden_seq)
        logvar = torch.clamp(self.logvar_head(hidden_seq), min=-6.0, max=4.0)
        return mean, logvar, targets

    def negative_log_likelihood(self, x_seq):
        mean, logvar, targets = self.conditional_parameters(x_seq)
        squared_error = (targets - mean) ** 2
        nll = 0.5 * (LOG_2PI + logvar + squared_error * torch.exp(-logvar))
        stepwise_nll = nll.sum(dim=-1)
        return stepwise_nll.sum(dim=-1), stepwise_nll

    def sequence_log_probability(self, x_seq):
        total_nll, _ = self.negative_log_likelihood(x_seq)
        return -total_nll

    def probability_score(self, x_seq):
        mean_log_prob = self.sequence_log_probability(x_seq) / ((x_seq.size(1) - 1) * self.input_dim)
        return torch.exp(mean_log_prob)

    def forward(self, x_seq):
        return self.sequence_log_probability(x_seq)


class SequenceAnomalyScoreWrapper(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, x_seq):
        return (-self.detector.sequence_log_probability(x_seq)).unsqueeze(-1)
