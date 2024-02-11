import json
import os
from logging import Logger
from typing import Any, cast

import numpy as np
import prettytable as pt
import requests
import torch
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

import utils
from config import Config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype="int64")
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = "<pad>"
    UNK = "<unk>"
    SUC = "<suc>"

    def __init__(self) -> None:
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label: str) -> None:
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def label_to_id(self, label: str) -> int:
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i: int) -> str:
        return self.id2label[i]


def collate_fn(
    data: list[Any],
) -> Any:
    (
        bert_inputs,
        grid_labels,
        grid_mask2d,
        pieces2word,
        dist_inputs,
        sent_length,
        entity_text,
    ) = map(list, zip(*data))

    max_tok = np.max(sent_length)
    sent_length_tensor = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs_tensor = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs_tensor.size(0)

    def fill(data: list[torch.Tensor], new_data: torch.Tensor) -> torch.Tensor:
        for j, x in enumerate(data):
            new_data[j, : x.shape[0], : x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs_tensor = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels_tensor = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d_tensor = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word_tensor = fill(pieces2word, sub_mat)

    return (
        bert_inputs_tensor,
        grid_labels_tensor,
        grid_mask2d_tensor,
        pieces2word_tensor,
        dist_inputs_tensor,
        sent_length_tensor,
        entity_text,
    )


class RelationDataset(Dataset):
    def __init__(
        self,
        bert_inputs: list[np.ndarray],
        grid_labels: list[np.ndarray],
        grid_mask2d: list[np.ndarray],
        pieces2word: list[np.ndarray],
        dist_inputs: list[np.ndarray],
        sent_length: list[int],
        entity_text: list[set[Any]],
    ) -> None:

        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text

    def __getitem__(self, item: int) -> tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        int,
        set[Any],
    ]:
        return (
            torch.LongTensor(self.bert_inputs[item]),
            torch.LongTensor(self.grid_labels[item]),
            torch.LongTensor(self.grid_mask2d[item]),
            torch.LongTensor(self.pieces2word[item]),
            torch.LongTensor(self.dist_inputs[item]),
            self.sent_length[item],
            self.entity_text[item],
        )

    def __len__(self) -> int:
        return len(self.bert_inputs)


def process_bert(
    data: list[dict[str, Any]], tokenizer: BertTokenizerFast, vocab: Vocabulary
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[int],
    list[set[Any]],
]:

    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []

    for index, instance in enumerate(data):
        if len(instance["sentence"]) == 0:
            continue

        tokens = [tokenizer.tokenize(word) for word in instance["sentence"]]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs_list = cast(list[int], tokenizer.convert_tokens_to_ids(pieces))
        _bert_inputs = np.array(
            [tokenizer.cls_token_id] + _bert_inputs_list + [tokenizer.sep_token_id]
        )

        length = len(instance["sentence"])
        _grid_labels = np.zeros((length, length), dtype=int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=bool)
        _dist_inputs = np.zeros((length, length), dtype=int)
        _grid_mask2d = np.ones((length, length), dtype=bool)

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces_range = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces_range[0] + 1 : pieces_range[-1] + 2] = 1
                start += len(pieces_range)

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        for entity in instance["ner"]:
            entoty_index = entity["index"]
            for i in range(len(entoty_index)):
                if i + 1 >= len(entoty_index):
                    break
                _grid_labels[entoty_index[i], entoty_index[i + 1]] = 1
            _grid_labels[entoty_index[-1], entoty_index[0]] = vocab.label_to_id(
                entity["type"]
            )

        _entity_text = set(
            [
                utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"]))
                for e in instance["ner"]
            ]
        )

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)

    return (
        bert_inputs,
        grid_labels,
        grid_mask2d,
        pieces2word,
        dist_inputs,
        sent_length,
        entity_text,
    )


def fill_vocab(vocab: Vocabulary, dataset: list[dict[str, Any]]) -> int:
    entity_num = 0
    for instance in dataset:
        for entity in instance["ner"]:
            vocab.add_label(entity["type"])
        entity_num += len(instance["ner"])
    return entity_num


def load_data_bert(
    config: Config,
) -> tuple[
    tuple[RelationDataset, RelationDataset, RelationDataset],
    tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
]:
    with open(
        "./data/{}/train.json".format(config.dataset), "r", encoding="utf-8"
    ) as f:
        train_data = json.load(f)
    with open("./data/{}/dev.json".format(config.dataset), "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    with open("./data/{}/test.json".format(config.dataset), "r", encoding="utf-8") as f:
        test_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")

    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data)
    dev_ent_num = fill_vocab(vocab, dev_data)
    test_ent_num = fill_vocab(vocab, test_data)

    table = pt.PrettyTable([config.dataset, "sentences", "entities"])
    table.add_row(["train", len(train_data), train_ent_num])
    table.add_row(["dev", len(dev_data), dev_ent_num])
    table.add_row(["test", len(test_data), test_ent_num])
    # TODO fix this
    cast(Logger, config.logger).info("\n{}".format(table))

    config.label_num = len(vocab.label2id)
    config.vocab = vocab

    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    return (train_dataset, dev_dataset, test_dataset), (train_data, dev_data, test_data)
