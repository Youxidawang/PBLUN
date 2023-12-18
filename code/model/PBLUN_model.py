import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from .table import TableEncoder
from .matching_layer import MatchingLayer
from .gcn import GCNModel
from .initial_dependency_graph import initial
import torch.nn.functional as F
class PBLUNModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.table_encoder = TableEncoder(config)
        self.inference = InferenceLayer(config)
        self.matching = MatchingLayer(config)
        self.gcn_layer_a = GCNModel(config)
        self.gcn_layer_o = GCNModel(config)
        self.gcn_layer_a2 = GCNModel(config)
        self.gcn_layer_o2 = GCNModel(config)
        self.cls_linear_aspect = torch.nn.Linear(768, 2)
        self.cls_linear_opinion = torch.nn.Linear(768, 2)
        self.ap_fc_S = torch.nn.Linear(768, 300)
        self.op_fc_S = torch.nn.Linear(768, 300)
        self.triplet_biaffine_S = Biaffine(config, 300, 300, 1, bias=(True, True))
        self.triplet_biaffine_E = Biaffine(config, 300, 300, 1, bias=(True, True))

    def forward(self, input_ids, attention_mask, ids, text, adj_pack,
                start_label_masks, end_label_masks,
                aspect_golde_tags, opinion_golde_tags, aspect_pred_tags, aspect_adjs, opinion_pred_tags, opinion_adjs,
                length, all_token_range,
                t_start_labels=None, t_end_labels=None,
                o_start_labels=None, o_end_labels=None,
                table_labels_S=None, table_labels_E=None,
                polarity_labels=None, pairs_true=None,
                ):

        seq = self.bert(input_ids, attention_mask)[0]

        lengths = []
        temp_lengths = []
        for i in range(input_ids.size(0)):
            token_len = torch.sum(input_ids[i] != 0, dim=-1)
            lengths.append(token_len)
            temp_lengths.append(token_len)

        aspect_tag_feature = self.gcn_layer_a(seq, input_ids, aspect_pred_tags, aspect_adjs, attention_mask)
        aspect_tag_feature = self.cls_linear_aspect(aspect_tag_feature)

        aspect_preds = torch.softmax(aspect_tag_feature, dim=-1)
        aspect_preds = torch.argmax(aspect_preds, dim=2)
        for i in range(aspect_pred_tags.shape[0]):
            temp_lengths[i] = temp_lengths[i] - 1
            for j in range(temp_lengths[i] + 1):
                if aspect_pred_tags[i][j] != 0 and 3 <= j <= temp_lengths[i] - 4 and temp_lengths[i] >= 7:
                    if aspect_preds[i][j] == 0:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                    else:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                        aspect_pred_tags[i][j - 1] = aspect_preds[i][j - 1]
                        aspect_pred_tags[i][j - 2] = aspect_preds[i][j - 2]
                        aspect_pred_tags[i][j + 1] = aspect_preds[i][j + 1]
                        aspect_pred_tags[i][j + 2] = aspect_preds[i][j + 2]

                elif aspect_pred_tags[i][j] != 0 and j == 2 and temp_lengths[i] >= 6:
                    if aspect_preds[i][j] == 0:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                    else:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                        aspect_pred_tags[i][j - 1] = aspect_preds[i][j - 1]
                        aspect_pred_tags[i][j + 1] = aspect_preds[i][j + 1]
                        aspect_pred_tags[i][j + 2] = aspect_preds[i][j + 2]

                elif aspect_pred_tags[i][j] != 0 and j == 2 and temp_lengths[i] == 5:
                    if aspect_preds[i][j] == 0:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                    else:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                        aspect_pred_tags[i][j - 1] = aspect_preds[i][j - 1]
                        aspect_pred_tags[i][j + 1] = aspect_preds[i][j + 1]

                elif aspect_pred_tags[i][j] != 0 and j == 2 and temp_lengths[i] == 4:
                    if aspect_preds[i][j] == 0:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                    else:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                        aspect_pred_tags[i][j - 1] = aspect_preds[i][j - 1]

                elif aspect_pred_tags[i][j] != 0 and j == 1 and temp_lengths[i] >= 5:
                    if aspect_preds[i][j] == 0:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                    else:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                        aspect_pred_tags[i][j + 1] = aspect_preds[i][j + 1]
                        aspect_pred_tags[i][j + 2] = aspect_preds[i][j + 2]

                elif aspect_pred_tags[i][j] != 0 and j == 1 and temp_lengths[i] == 4:
                    if aspect_preds[i][j] == 0:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                    else:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                        aspect_pred_tags[i][j + 1] = aspect_preds[i][j + 1]

                elif aspect_pred_tags[i][j] != 0 and j == temp_lengths[i] - 3 and temp_lengths[i] >= 6:
                    if aspect_preds[i][j] == 0:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                    else:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                        aspect_pred_tags[i][j - 1] = aspect_preds[i][j - 1]
                        aspect_pred_tags[i][j - 2] = aspect_preds[i][j - 2]
                        aspect_pred_tags[i][j + 1] = aspect_preds[i][j + 1]

                elif aspect_pred_tags[i][j] != 0 and j == temp_lengths[i] - 3 and temp_lengths[i] == 5:
                    if aspect_preds[i][j] == 0:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                    else:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                        aspect_pred_tags[i][j - 1] = aspect_preds[i][j - 1]
                        aspect_pred_tags[i][j + 1] = aspect_preds[i][j + 1]

                elif aspect_pred_tags[i][j] != 0 and j == temp_lengths[i] - 2 and temp_lengths[i] == 5:
                    if aspect_preds[i][j] == 0:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                    else:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                        aspect_pred_tags[i][j - 1] = aspect_preds[i][j - 1]
                        aspect_pred_tags[i][j - 2] = aspect_preds[i][j - 2]

                elif aspect_pred_tags[i][j] != 0 and j == temp_lengths[i] - 2 and temp_lengths[i] == 4:
                    if aspect_preds[i][j] == 0:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                    else:
                        aspect_pred_tags[i][j] = aspect_preds[i][j]
                        aspect_pred_tags[i][j - 1] = aspect_preds[i][j - 1]

        opinion_tag_feature = self.gcn_layer_o(seq, input_ids, aspect_pred_tags, aspect_adjs, attention_mask)
        opinion_tag_feature = self.cls_linear_opinion(opinion_tag_feature)

        opinion_preds = torch.softmax(opinion_tag_feature, dim=-1)
        opinion_preds = torch.argmax(opinion_preds, dim=2)
        for i in range(opinion_pred_tags.shape[0]):
            for j in range(temp_lengths[i] + 1):
                if opinion_pred_tags[i][j] != 0 and 3 <= j <= temp_lengths[i] - 4 and temp_lengths[i] >= 7:
                    if opinion_preds[i][j] == 0:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                    else:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                        opinion_pred_tags[i][j - 1] = opinion_preds[i][j - 1]
                        opinion_pred_tags[i][j - 2] = opinion_preds[i][j - 2]
                        opinion_pred_tags[i][j + 1] = opinion_preds[i][j + 1]
                        opinion_pred_tags[i][j + 2] = opinion_preds[i][j + 2]

                elif opinion_pred_tags[i][j] != 0 and j == 2 and temp_lengths[i] >= 6:
                    if opinion_preds[i][j] == 0:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                    else:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                        opinion_pred_tags[i][j - 1] = opinion_preds[i][j - 1]
                        opinion_pred_tags[i][j + 1] = opinion_preds[i][j + 1]
                        opinion_pred_tags[i][j + 2] = opinion_preds[i][j + 2]

                elif opinion_pred_tags[i][j] != 0 and j == 2 and temp_lengths[i] == 5:
                    if opinion_preds[i][j] == 0:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                    else:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                        opinion_pred_tags[i][j - 1] = opinion_preds[i][j - 1]
                        opinion_pred_tags[i][j + 1] = opinion_preds[i][j + 1]

                elif opinion_pred_tags[i][j] != 0 and j == 2 and temp_lengths[i] == 4:
                    if opinion_preds[i][j] == 0:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                    else:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                        opinion_pred_tags[i][j - 1] = opinion_preds[i][j - 1]

                elif opinion_pred_tags[i][j] != 0 and j == 1 and temp_lengths[i] >= 5:
                    if opinion_preds[i][j] == 0:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                    else:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                        opinion_pred_tags[i][j + 1] = opinion_preds[i][j + 1]
                        opinion_pred_tags[i][j + 2] = opinion_preds[i][j + 2]

                elif opinion_pred_tags[i][j] != 0 and j == 1 and temp_lengths[i] == 4:
                    if opinion_preds[i][j] == 0:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                    else:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                        opinion_pred_tags[i][j + 1] = opinion_preds[i][j + 1]

                elif opinion_pred_tags[i][j] != 0 and j == temp_lengths[i] - 3 and temp_lengths[i] >= 6:
                    if opinion_preds[i][j] == 0:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                    else:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                        opinion_pred_tags[i][j - 1] = opinion_preds[i][j - 1]
                        opinion_pred_tags[i][j - 2] = opinion_preds[i][j - 2]
                        opinion_pred_tags[i][j + 1] = opinion_preds[i][j + 1]

                elif opinion_pred_tags[i][j] != 0 and j == temp_lengths[i] - 3 and temp_lengths[i] == 5:
                    if opinion_preds[i][j] == 0:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                    else:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                        opinion_pred_tags[i][j - 1] = opinion_preds[i][j - 1]
                        opinion_pred_tags[i][j + 1] = opinion_preds[i][j + 1]

                elif opinion_pred_tags[i][j] != 0 and j == temp_lengths[i] - 2 and temp_lengths[i] == 5:
                    if opinion_preds[i][j] == 0:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                    else:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                        opinion_pred_tags[i][j - 1] = opinion_preds[i][j - 1]
                        opinion_pred_tags[i][j - 2] = opinion_preds[i][j - 2]

                elif opinion_pred_tags[i][j] != 0 and j == temp_lengths[i] - 2 and temp_lengths[i] == 4:
                    if opinion_preds[i][j] == 0:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                    else:
                        opinion_pred_tags[i][j] = opinion_preds[i][j]
                        opinion_pred_tags[i][j - 1] = opinion_preds[i][j - 1]

        final_aspect_adjs = []
        for i in range(len(aspect_adjs)):
            final_aspect_adjs.append(initial(text[i], all_token_range[i], aspect_pred_tags[i], length, adj_pack[i]))
        final_aspect_adjs = torch.tensor(final_aspect_adjs, dtype=torch.float).to('cuda')

        final_opinion_adjs = []
        for i in range(len(opinion_adjs)):
            final_opinion_adjs.append(initial(text[i], all_token_range[i], opinion_pred_tags[i], length, adj_pack[i]))
        final_opinion_adjs = torch.tensor(final_opinion_adjs, dtype=torch.float).to('cuda')

        aspect_gcn_feature = self.gcn_layer_a2(seq, input_ids, aspect_pred_tags, final_aspect_adjs, attention_mask)
        opinion_gcn_feature = self.gcn_layer_o2(seq, input_ids, opinion_pred_tags, final_opinion_adjs, attention_mask)

        bert_feature = seq + aspect_gcn_feature + opinion_gcn_feature

        ap_node_S = F.relu(self.ap_fc_S(bert_feature))
        op_node_S = F.relu(self.op_fc_S(bert_feature))

        biaffine_edge_S = self.triplet_biaffine_S(ap_node_S, op_node_S)
        biaffine_edge_E = self.triplet_biaffine_E(ap_node_S, op_node_S)

        biaffine_edge_S = torch.sigmoid(biaffine_edge_S)
        biaffine_edge_E = torch.sigmoid(biaffine_edge_E)

        table = self.table_encoder(bert_feature, attention_mask)

        output = self.inference(table, attention_mask, table_labels_S, table_labels_E, aspect_tag_feature,
                                opinion_tag_feature, aspect_golde_tags, opinion_golde_tags, biaffine_edge_S,
                                biaffine_edge_E)
        output['ids'] = ids

        output = self.matching(output, table, pairs_true)
        return output

class InferenceLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cls_linear_S = nn.Linear(768, 1)
        self.cls_linear_E = nn.Linear(768, 1)

    def span_pruning(self, pred, z, attention_mask):
        mask_length = attention_mask.sum(dim=1) - 2
        length = ((attention_mask.sum(dim=1) - 2) * z).long()
        length[length < 5] = 5
        max_length = mask_length ** 2
        for i in range(length.shape[0]):
            if length[i] > max_length[i]:
                length[i] = max_length[i]
        batch_size = attention_mask.shape[0]
        pred_sort, _ = pred.view(batch_size, -1).sort(descending=True)
        batchs = torch.arange(batch_size).to('cuda')
        topkth = pred_sort[batchs, length - 1].unsqueeze(1)
        return pred >= (topkth.view(batch_size, 1, 1))

    def forward(self, table, attention_mask, table_labels_S, table_labels_E, aspect_pred_tags, opinion_pred_tags,aspect_golde_tags, opinion_golde_tags, biaffine_edge_S, biaffine_edge_E):

        outputs = {}
        logits_S_temp = torch.squeeze(self.cls_linear_S(table), 3)
        logits_E_temp = torch.squeeze(self.cls_linear_E(table), 3)

        logits_S = logits_S_temp * (1 + torch.squeeze(biaffine_edge_S, 3))
        logits_E = logits_E_temp * (1 + torch.squeeze(biaffine_edge_E, 3))

        loss_func = nn.BCEWithLogitsLoss(weight=(table_labels_S >= 0))

        outputs['table_loss_S'] = loss_func(logits_S, table_labels_S.float())
        outputs['table_loss_E'] = loss_func(logits_E, table_labels_E.float())

        aspect_golde_tags = aspect_golde_tags[:, :aspect_golde_tags.shape[1]].reshape([-1])
        aspect_pred_tags = aspect_pred_tags.reshape([-1, aspect_pred_tags.shape[2]])
        aspect_loss = 0.1 * F.cross_entropy(aspect_pred_tags, aspect_golde_tags.long(), ignore_index=-1)

        opinion_golde_tags = opinion_golde_tags[:, :opinion_golde_tags.shape[1]].reshape([-1])
        opinion_pred_tags = opinion_pred_tags.reshape([-1, opinion_pred_tags.shape[2]])
        opinion_loss = 0.1 * F.cross_entropy(opinion_pred_tags, opinion_golde_tags.long(), ignore_index=-1)

        S_pred = torch.sigmoid(logits_S) * (table_labels_S >= 0)
        E_pred = torch.sigmoid(logits_E) * (table_labels_S >= 0)

        if self.config.span_pruning != 0:
            table_predict_S = self.span_pruning(S_pred, self.config.span_pruning, attention_mask)
            table_predict_E = self.span_pruning(E_pred, self.config.span_pruning, attention_mask)
        else:
            table_predict_S = S_pred > 0.5
            table_predict_E = E_pred > 0.5
        outputs['table_predict_S'] = table_predict_S
        outputs['table_predict_E'] = table_predict_E
        outputs['table_labels_S'] = table_labels_S
        outputs['table_labels_E'] = table_labels_E
        outputs['aspect_preds_loss'] = aspect_loss
        outputs['opinion_preds_loss'] = opinion_loss
        return outputs


class Biaffine(torch.nn.Module):
    def __init__(self, args, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.args = args
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.bia_linear = torch.nn.Linear(in_features=self.linear_input_size,
                                          out_features=self.linear_output_size,
                                          bias=False)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).cuda()
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).cuda()
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1
        affine = self.bia_linear(input1)
        affine = affine.view(batch_size, len1 * self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        biaffine = torch.bmm(affine, input2)
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        return biaffine
