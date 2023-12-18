import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import AutoTokenizer

import os
from . import load_json
from .initial_dependency_graph import initial
polarity_map = {
    'NEG': 0,
    'NEU': 1,
    'POS': 2
}

polarity_map_reversed = {
    0: 'NEG',
    1: 'NEU',
    2: 'POS'
}

class Example:
    def __init__(self, data, max_length=-1):

        self.data = data
        self.max_length = max_length
        self.data['tokens'] = eval(str(self.data['tokens']))


    def __getitem__(self, key):
        return self.data[key]


    def t_entities(self):
        return [tuple(entity[:3]) for entity in self['entities'] if entity[0]=='target']

    def o_entities(self):
        return [tuple(entity[:3]) for entity in self['entities'] if entity[0]=='opinion']

    def entity_label(self, target_oriented, length):
        entities = self.t_entities() if target_oriented else self.o_entities()
        return Example.make_start_end_labels(entities, length)

    def table_label(self, length, ty, id_len):
        label = [[-1 for _ in range(length)] for _ in range(length)]
        id_len = id_len.item()

        for i in range(1, id_len-1):
            for j in range(1, id_len-1):
                label[i][j] = 0

        for t_start, t_end, o_start, o_end, pol in self['pairs']:
            if ty == 'S':
                label[t_start+1][o_start+1] = 1
            elif ty == 'E':
                label[t_end][o_end] = 1
        return label

    @staticmethod
    def make_start_end_labels(entities, length, plus_one=True):
        start_label = [0] * length
        end_label   = [0] * length

        for (t, s, e) in entities:
            if plus_one:
                s, e = s+1, e+1

            if s < length:
                start_label[s] = 1

            if e-1 < length:
                end_label[e-1] = 1

        return start_label, end_label


class DataCollatorForASTE:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length


    def __call__(self, examples):

        batch = self.tokenizer_function(examples)

        length = batch['input_ids'].size(1)

        batch['t_start_labels'], batch['t_end_labels'] = self.start_end_labels(examples, True,  length)
        batch['o_start_labels'], batch['o_end_labels'] = self.start_end_labels(examples, False, length)
        batch['example_ids'] = [example['ID'] for example in examples]
        batch['table_labels_S'] = torch.tensor([examples[i].table_label(length, 'S', (batch['input_ids'][i]>0).sum()) for i in range(len(examples))], dtype=torch.long)
        batch['table_labels_E'] = torch.tensor([examples[i].table_label(length, 'E', (batch['input_ids'][i]>0).sum()) for i in range(len(examples))], dtype=torch.long)

        al = [example['pairs'] for example in examples]
        pairs_ret = []
        for pairs in al:
            pairs_chg = []
            for p in pairs:
                pairs_chg.append([p[0],p[1],p[2], p[3], polarity_map[p[4]]+1])
            pairs_ret.append(pairs_chg)
        batch['pairs_true'] = pairs_ret
        
        return {
            'ids': batch['example_ids'],
            'input_ids'     : batch['input_ids'],
            'attention_mask': batch['attention_mask'],

            't_start_labels': batch['t_start_labels'],
            't_end_labels'  : batch['t_end_labels'],
            'o_start_labels': batch['o_start_labels'],
            'o_end_labels'  : batch['o_end_labels'],

            'start_label_masks': batch['start_label_masks'],
            'end_label_masks'  : batch['end_label_masks'],

            'table_labels_S'  : batch['table_labels_S'],
            'table_labels_E'  : batch['table_labels_E'],
            'pairs_true'      : batch['pairs_true'],
            'aspect_golde_tags': batch['aspect_golde_tags'],
            'opinion_golde_tags': batch['opinion_golde_tags'],
            'aspect_pred_tags': batch['aspect_pred_tags'],
            'aspect_adjs': batch['aspect_adjs'],
            'opinion_pred_tags': batch['opinion_pred_tags'],
            'opinion_adjs': batch['opinion_adjs'],
            'text': batch['text'],
            'length': batch['length'],
            'all_token_range': batch['all_token_range'],
            'adj_pack': batch['adj_pack'],
        }

    def start_end_labels(self, examples, target_oriented, length):
        start_labels = []
        end_labels   = []

        for example in examples:
            start_label, end_label = example.entity_label(target_oriented, length)
            start_labels.append(start_label)
            end_labels.append(end_label)

        start_labels = torch.tensor(start_labels, dtype=torch.long)
        end_labels   = torch.tensor(end_labels, dtype=torch.long)

        return start_labels, end_labels

    def tokenizer_function(self, examples):
        text = [example['sentence'] for example in examples]
        gold = [example['pairs'] for example in examples]
        postag= [example['postag'] for example in examples]
        kwargs = {
            'text': text,
            'return_tensors': 'pt'
        }

        if self.max_seq_length in (-1, 'longest'):
            kwargs['padding'] = True
        else:
            kwargs['padding'] = 'max_length'
            kwargs['max_length'] = self.max_seq_length
            kwargs['truncation'] = True

        batch_encodings = self.tokenizer(**kwargs)
        length = batch_encodings['input_ids'].size(1)

        start_label_masks = []
        end_label_masks   = []

        for i in range(len(examples)):
            encoding = batch_encodings[i]
            word_ids = encoding.word_ids
            type_ids = encoding.type_ids

            start_label_mask = [(1 if type_ids[i]==0 else 0) for i in range(length)]
            end_label_mask   = [(1 if type_ids[i]==0 else 0) for i in range(length)]

            for token_idx in range(length):
                current_word_idx = word_ids[token_idx]
                prev_word_idx = word_ids[token_idx-1] if token_idx-1 > 0 else None
                next_word_idx = word_ids[token_idx+1] if token_idx+1 < length else None

                if prev_word_idx is not None and current_word_idx == prev_word_idx:
                    start_label_mask[token_idx] = 0
                if next_word_idx is not None and current_word_idx == next_word_idx:
                    end_label_mask[token_idx] = 0

            start_label_masks.append(start_label_mask)
            end_label_masks.append(end_label_mask)

        batch_encodings = dict(batch_encodings)
        batch_encodings['start_label_masks'] = torch.tensor(start_label_masks, dtype=torch.long)
        batch_encodings['end_label_masks']   = torch.tensor(end_label_masks, dtype=torch.long)

        adj_aspcet_golde_tags = []
        all_token_range = []
        for batch in range(len(examples)):
            temp_words = text[batch].split(" ")
            token_start = 1
            temp_range = []
            for i, w, in enumerate(temp_words):
                token_end = token_start + len(self.tokenizer.encode(w, add_special_tokens=False))
                temp_range.append([token_start, token_end - 1])
                token_start = token_end
            all_token_range.append(temp_range)

            aspect_gold_tag = torch.zeros(length)
            for i in range(len(gold[batch])):
                a_b = gold[batch][i][0]
                a_e = gold[batch][i][1]
                for j in range(a_e - a_b):
                    for k in range(len(temp_range)):
                        if temp_range[k][0] >= a_b + 1 + j >= temp_range[k][0]:
                            aspect_gold_tag[k + 1] = 1
            adj_aspcet_golde_tags.append(aspect_gold_tag)
        adj_aspect_golde_tags = torch.stack(adj_aspcet_golde_tags)

        aspect_golde_tags = []
        for i in range(len(all_token_range)):
            target_tag = torch.zeros(length)
            for j in range(len(all_token_range[i])):
                if all_token_range[i][j][1] != all_token_range[i][j][0]:
                    gap = all_token_range[i][j][1] - all_token_range[i][j][0] + 1
                    tag = adj_aspect_golde_tags[i][j + 1]
                    for k in range(gap):
                        target_tag[all_token_range[i][j][0] + k] = tag
                else:
                    tag = adj_aspect_golde_tags[i][j + 1]
                    target_tag[all_token_range[i][j][0]] = tag
            aspect_golde_tags.append(target_tag)
        aspect_golde_tags = torch.stack(aspect_golde_tags)

        adj_opinion_golde_tags = []
        all_token_range = []
        for batch in range(len(examples)):
            temp_words = text[batch].split(" ")
            token_start = 1
            temp_range = []
            for i, w, in enumerate(temp_words):
                token_end = token_start + len(self.tokenizer.encode(w, add_special_tokens=False))
                temp_range.append([token_start, token_end - 1])
                token_start = token_end
            all_token_range.append(temp_range)

            opinion_gold_tag = torch.zeros(length)
            for i in range(len(gold[batch])):
                o_b = gold[batch][i][2]
                o_e = gold[batch][i][3]
                for j in range(o_e - o_b):
                    for k in range(len(temp_range)):
                        if temp_range[k][0] >= o_b + 1 + j >= temp_range[k][0]:
                            opinion_gold_tag[k + 1] = 1
            adj_opinion_golde_tags.append(opinion_gold_tag)
        adj_opinion_golde_tags = torch.stack(adj_opinion_golde_tags)

        opinion_golde_tags = []
        for i in range(len(all_token_range)):
            target_tag = torch.zeros(length)
            for j in range(len(all_token_range[i])):
                if all_token_range[i][j][1] != all_token_range[i][j][0]:
                    gap = all_token_range[i][j][1] - all_token_range[i][j][0] + 1
                    tag = adj_opinion_golde_tags[i][j + 1]
                    for k in range(gap):
                        target_tag[all_token_range[i][j][0] + k] = tag
                else:
                    tag = adj_opinion_golde_tags[i][j + 1]
                    target_tag[all_token_range[i][j][0]] = tag
            opinion_golde_tags.append(target_tag)
        opinion_golde_tags = torch.stack(opinion_golde_tags)

        adj_opinion_pred_tags = []
        for i in range(len(examples)):
            temp_tag = torch.zeros(length)
            for j in range(len(postag[i])):
                if postag[i][j] == 'JJ' or postag[i][j] == 'JJR' or postag[i][j] == 'JJS':
                    temp_tag[j+1] = 1
            adj_opinion_pred_tags.append(temp_tag)
        adj_opinion_pred_tags = torch.stack(adj_opinion_pred_tags)

        adj_aspect_pred_tags = []
        for i in range(len(examples)):
            temp_tag = torch.zeros(length)
            for j in range(len(postag[i])):
                if postag[i][j] == 'NN' or postag[i][j] == 'NNS':
                    temp_tag[j+1] = 1
            adj_aspect_pred_tags.append(temp_tag)
        adj_aspect_pred_tags = torch.stack(adj_aspect_pred_tags)

        aspect_pred_tags = []
        for i in range(len(all_token_range)):
            target_tag = torch.zeros(length)
            for j in range(len(all_token_range[i])):
                if all_token_range[i][j][1] != all_token_range[i][j][0]:
                    gap = all_token_range[i][j][1] - all_token_range[i][j][0] + 1
                    tag = adj_aspect_pred_tags[i][j + 1]
                    for k in range(gap):
                        target_tag[all_token_range[i][j][0] + k] = tag
                else:
                    tag = adj_aspect_pred_tags[i][j + 1]
                    target_tag[all_token_range[i][j][0]] = tag
            aspect_pred_tags.append(target_tag)
        aspect_pred_tags = torch.stack(aspect_pred_tags)

        opinion_pred_tags = []
        for i in range(len(all_token_range)):
            target_tag = torch.zeros(length)
            for j in range(len(all_token_range[i])):
                if all_token_range[i][j][1] != all_token_range[i][j][0]:
                    gap = all_token_range[i][j][1] - all_token_range[i][j][0] + 1
                    tag = adj_opinion_pred_tags[i][j + 1]
                    for k in range(gap):
                        target_tag[all_token_range[i][j][0] + k] = tag
                else:
                    tag = adj_opinion_pred_tags[i][j + 1]
                    target_tag[all_token_range[i][j][0]] = tag
            opinion_pred_tags.append(target_tag)
        opinion_pred_tags = torch.stack(opinion_pred_tags)

        adj = [example['adj'] for example in examples]
        adj_pack = [F.pad(torch.tensor(ls), [0, length - len(ls), 0, length - len(ls)]).tolist() for ls in adj]
        ori_adj = [F.pad(torch.tensor(ls), [0, length - len(ls), 0, length - len(ls)]).tolist() for ls in adj]

        aspect_adjs = []
        opinion_adjs = []
        for i in range(len(adj)):
            aspect_adjs.append(initial(text[i], all_token_range[i], aspect_pred_tags[i], length, adj_pack[i]))
            opinion_adjs.append(initial(text[i], all_token_range[i], opinion_pred_tags[i], length, adj_pack[i]))
        aspect_adjs = torch.tensor(aspect_adjs, dtype=torch.float)
        opinion_adjs = torch.tensor(opinion_adjs, dtype=torch.float)

        batch_encodings['text'] = text
        batch_encodings['aspect_golde_tags'] = aspect_golde_tags
        batch_encodings['opinion_golde_tags'] = opinion_golde_tags
        batch_encodings['aspect_pred_tags'] = aspect_pred_tags
        batch_encodings['aspect_adjs'] = aspect_adjs
        batch_encodings['opinion_pred_tags'] = opinion_pred_tags
        batch_encodings['opinion_adjs'] = opinion_adjs
        batch_encodings['length'] = length
        batch_encodings['all_token_range'] = all_token_range
        batch_encodings['adj_pack'] = ori_adj
        return batch_encodings


class ASTEDataModule(pl.LightningDataModule):
    def __init__(self, args):

        super(ASTEDataModule, self).__init__()
        self.args = args


        self.model_name_or_path = args.model_name_or_path
        self.max_seq_length     = args.max_seq_length if args.max_seq_length > 0 else 'longest'
        self.train_batch_size   = args.train_batch_size
        self.eval_batch_size    = args.eval_batch_size

        self.data_dir    = args.prefix + args.dataset
        self.num_workers = args.num_workers
        self.cuda_ids    = args.cuda_ids

        self.table_num_labels = 6  # 4

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

    def load_dataset(self):
        train_file_name = os.path.join(self.data_dir, 'train.json')
        dev_file_name   = os.path.join(self.data_dir, 'dev.json')
        test_file_name  = os.path.join(self.data_dir, 'test.json')

        if not os.path.exists(dev_file_name):
            dev_file_name = test_file_name
        
        train_examples = [Example(data, self.max_seq_length) for data in load_json(train_file_name)]
        dev_examples   = [Example(data, self.max_seq_length) for data in load_json(dev_file_name)]
        test_examples  = [Example(data, self.max_seq_length) for data in load_json(test_file_name)]


        self.raw_datasets = {
            'train': train_examples, 
            'dev'  : dev_examples,
            'test' : test_examples
        }

    def get_dataloader(self, mode, batch_size, shuffle):
        dataloader = DataLoader(
            dataset=self.raw_datasets[mode],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=DataCollatorForASTE(tokenizer=self.tokenizer, 
                                           max_seq_length=self.max_seq_length),
            pin_memory=True,
            prefetch_factor=16
        )

        print(mode, len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.eval_batch_size, shuffle=False)
