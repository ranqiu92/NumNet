
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.nn import util


class NumGNN(nn.Module):

    def __init__(self, node_dim, iteration_steps=1):
        super(NumGNN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._dd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dq_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)

        self._dd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dq_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)

    def forward(self, d_node, q_node, d_node_mask, q_node_mask, graph):
        d_node_len = d_node.size(1)
        q_node_len = q_node.size(1)

        diagmat = torch.diagflat(torch.ones(d_node.size(1), dtype=torch.long, device=d_node.device))
        diagmat = diagmat.unsqueeze(0).expand(d_node.size(0), -1, -1)
        dd_graph = d_node_mask.unsqueeze(1) * d_node_mask.unsqueeze(-1) * (1 - diagmat)
        dd_graph_left = dd_graph * graph[:, :d_node_len, :d_node_len]
        dd_graph_right = dd_graph * (1 - graph[:, :d_node_len, :d_node_len])

        diagmat = torch.diagflat(torch.ones(q_node.size(1), dtype=torch.long, device=q_node.device))
        diagmat = diagmat.unsqueeze(0).expand(q_node.size(0), -1, -1)
        qq_graph = q_node_mask.unsqueeze(1) * q_node_mask.unsqueeze(-1) * (1 - diagmat) 
        qq_graph_left = qq_graph * graph[:, d_node_len:, d_node_len:]
        qq_graph_right = qq_graph * (1 - graph[:, d_node_len:, d_node_len:])

        dq_graph = d_node_mask.unsqueeze(-1) * q_node_mask.unsqueeze(1)
        dq_graph_left = dq_graph * graph[:, :d_node_len, d_node_len:]
        dq_graph_right = dq_graph * (1 - graph[:, :d_node_len, d_node_len:])

        qd_graph = q_node_mask.unsqueeze(-1) * d_node_mask.unsqueeze(1)
        qd_graph_left = qd_graph * graph[:, d_node_len:, :d_node_len]
        qd_graph_right = qd_graph * (1 - graph[:, d_node_len:, :d_node_len])


        d_node_neighbor_num = dd_graph_left.sum(-1) + dd_graph_right.sum(-1) + dq_graph_left.sum(-1) + dq_graph_right.sum(-1)
        d_node_neighbor_num_mask = (d_node_neighbor_num >= 1).long()
        d_node_neighbor_num = util.replace_masked_values(d_node_neighbor_num.float(), d_node_neighbor_num_mask, 1)

        q_node_neighbor_num = qq_graph_left.sum(-1) + qq_graph_right.sum(-1) + qd_graph_left.sum(-1) + qd_graph_right.sum(-1)
        q_node_neighbor_num_mask = (q_node_neighbor_num >= 1).long()
        q_node_neighbor_num = util.replace_masked_values(q_node_neighbor_num.float(), q_node_neighbor_num_mask, 1)

        for step in range(self.iteration_steps):
            d_node_weight = torch.sigmoid(self._node_weight_fc(d_node)).squeeze(-1)            
            q_node_weight = torch.sigmoid(self._node_weight_fc(q_node)).squeeze(-1)            

            self_d_node_info = self._self_node_fc(d_node)
            self_q_node_info = self._self_node_fc(q_node)

            dd_node_info_left = self._dd_node_fc_left(d_node)
            qd_node_info_left = self._qd_node_fc_left(d_node)
            qq_node_info_left = self._qq_node_fc_left(q_node)
            dq_node_info_left = self._dq_node_fc_left(q_node)

            dd_node_weight = util.replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                    dd_graph_left,
                    0)

            qd_node_weight = util.replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                    qd_graph_left,
                    0)

            qq_node_weight = util.replace_masked_values(
                    q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                    qq_graph_left,
                    0)

            dq_node_weight = util.replace_masked_values(
                    q_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                    dq_graph_left,
                    0)

            dd_node_info_left = torch.matmul(dd_node_weight, dd_node_info_left)
            qd_node_info_left = torch.matmul(qd_node_weight, qd_node_info_left)
            qq_node_info_left = torch.matmul(qq_node_weight, qq_node_info_left)
            dq_node_info_left = torch.matmul(dq_node_weight, dq_node_info_left)


            dd_node_info_right = self._dd_node_fc_right(d_node)
            qd_node_info_right = self._qd_node_fc_right(d_node)
            qq_node_info_right = self._qq_node_fc_right(q_node)
            dq_node_info_right = self._dq_node_fc_right(q_node)

            dd_node_weight = util.replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                    dd_graph_right,
                    0)

            qd_node_weight = util.replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                    qd_graph_right,
                    0)

            qq_node_weight = util.replace_masked_values(
                    q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                    qq_graph_right,
                    0)

            dq_node_weight = util.replace_masked_values(
                    q_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                    dq_graph_right,
                    0)

            dd_node_info_right = torch.matmul(dd_node_weight, dd_node_info_right)
            qd_node_info_right = torch.matmul(qd_node_weight, qd_node_info_right)
            qq_node_info_right = torch.matmul(qq_node_weight, qq_node_info_right)
            dq_node_info_right = torch.matmul(dq_node_weight, dq_node_info_right)


            agg_d_node_info = (dd_node_info_left + dd_node_info_right + dq_node_info_left + dq_node_info_right) / d_node_neighbor_num.unsqueeze(-1)
            agg_q_node_info = (qq_node_info_left + qq_node_info_right + qd_node_info_left + qd_node_info_right) / q_node_neighbor_num.unsqueeze(-1)

            d_node = F.relu(self_d_node_info + agg_d_node_info)
            q_node = F.relu(self_q_node_info + agg_q_node_info)

        return d_node, q_node
