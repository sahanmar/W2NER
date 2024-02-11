import argparse
import json
from typing import Any, Iterable, cast

import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

import data_loader
import utils
from config import Config
from model import Model


class Trainer:
    def __init__(self, model: Model):
        self.model: Model = model
        self.criterion = nn.CrossEntropyLoss()

        # TODO check why one is set and another is list
        bert_params: set[Parameter] = set(self.model.bert.parameters())
        other_params: list[Parameter] = list(set(self.model.parameters()) - bert_params)
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [
                    p
                    for n, p in model.bert.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "lr": config.bert_learning_rate,
                "weight_decay": config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.bert.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "lr": config.bert_learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": other_params,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
        ]

        self.optimizer = transformers.AdamW(
            cast(Iterable[Parameter], params),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warm_factor * updates_total,
            num_training_steps=updates_total,
        )

    def train(self, epoch: int, data_loader: DataLoader, device: torch.device) -> float:
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []

        for i, data_batch in enumerate(data_loader):
            data_batch = [data.to(device) for data in data_batch[:-1]]

            (
                bert_inputs,
                grid_labels,
                grid_mask2d,
                pieces2word,
                dist_inputs,
                sent_length,
            ) = data_batch

            outputs = model(
                bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length
            )

            grid_mask2d = grid_mask2d.clone()
            loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config.clip_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())

            self.scheduler.step()

        label_tensor_res = torch.cat(label_result)
        pred_tensor_res = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(
            label_tensor_res.numpy(), pred_tensor_res.numpy(), average="macro"
        )

        table = pt.PrettyTable(
            ["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"]
        )
        table.add_row(
            ["Label", "{:.4f}".format(np.mean(loss_list))]
            + ["{:3.4f}".format(x) for x in [f1, p, r]]
        )
        logger.info("\n{}".format(table))
        return cast(float, f1)

    def eval(self, epoch: int, data_loader: DataLoader, is_test: bool = False) -> float:
        self.model.eval()

        pred_result = []
        label_result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                entity_text = data_batch[-1]
                data_batch = [data.to(device) for data in data_batch[:-1]]
                (
                    bert_inputs,
                    grid_labels,
                    grid_mask2d,
                    pieces2word,
                    dist_inputs,
                    sent_length,
                ) = data_batch

                outputs = model(
                    bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length
                )
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, _ = utils.decode(
                    outputs.cpu().numpy(), entity_text, length.cpu().numpy()
                )

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())

        label_torch_res = torch.cat(label_result)
        pred_torch_res = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(
            label_torch_res.numpy(), pred_torch_res.numpy(), average="macro"
        )
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "EVAL" if not is_test else "TEST"
        logger.info(
            "{} Label F1 {}".format(
                title,
                f1_score(label_torch_res.numpy(), pred_torch_res.numpy(), average=None),
            )
        )

        table = pt.PrettyTable(
            ["{} {}".format(title, epoch), "F1", "Precision", "Recall"]
        )
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        logger.info("\n{}".format(table))
        return e_f1

    def predict(
        self, epoch: str, data_loader: DataLoader, data: list[dict[str, Any]]
    ) -> float:
        self.model.eval()

        pred_result = []
        label_result = []

        result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        i = 0
        with torch.no_grad():
            for data_batch in data_loader:
                sentence_batch = data[i : i + config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [data.to(device) for data in data_batch[:-1]]
                (
                    bert_inputs,
                    grid_labels,
                    grid_mask2d,
                    pieces2word,
                    dist_inputs,
                    sent_length,
                ) = data_batch

                outputs = model(
                    bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length
                )
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(
                    outputs.cpu().numpy(), entity_text, length.cpu().numpy()
                )

                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence["sentence"]
                    instance: dict[str, Any] = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        instance["entity"].append(
                            {
                                "text": [sentence[x] for x in ent[0]],
                                "type": config.vocab.id_to_label(ent[1]),
                            }
                        )
                    result.append(instance)

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
                i += config.batch_size

        label_tensor_res = torch.cat(label_result)
        pred_tensor_res = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(
            label_tensor_res.numpy(), pred_tensor_res.numpy(), average="macro"
        )
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "TEST"
        logger.info(
            "{} Label F1 {}".format(
                "TEST",
                f1_score(
                    label_tensor_res.numpy(), pred_tensor_res.numpy(), average=None
                ),
            )
        )

        table = pt.PrettyTable(
            ["{} {}".format(title, epoch), "F1", "Precision", "Recall"]
        )
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        logger.info("\n{}".format(table))

        with open(config.predict_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        return e_f1

    # TODO Make Path
    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    # TODO Make Path
    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/conll03.json")
    parser.add_argument("--save_path", type=str, default="./model.pt")
    parser.add_argument("--predict_path", type=str, default="./output.json")
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--dist_emb_size", type=int)
    parser.add_argument("--type_emb_size", type=int)
    parser.add_argument("--lstm_hid_size", type=int)
    parser.add_argument("--conv_hid_size", type=int)
    parser.add_argument("--bert_hid_size", type=int)
    parser.add_argument("--ffnn_hid_size", type=int)
    parser.add_argument("--biaffine_size", type=int)

    parser.add_argument("--dilation", type=str, help="e.g. 1,2,3")

    parser.add_argument("--emb_dropout", type=float)
    parser.add_argument("--conv_dropout", type=float)
    parser.add_argument("--out_dropout", type=float)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)

    parser.add_argument("--clip_grad_norm", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)

    parser.add_argument("--bert_name", type=str)
    parser.add_argument("--bert_learning_rate", type=float)
    parser.add_argument("--warm_factor", type=float)

    parser.add_argument("--use_bert_last_4_layers", type=int, help="1: true, 0: false")

    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    config = Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # random.seed(config.seed)
    # np.random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # torch.cuda.manual_seed(config.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    logger.info("Loading Data")
    datasets, ori_data = data_loader.load_data_bert(config)

    train_loader, dev_loader, test_loader = (
        DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            collate_fn=data_loader.collate_fn,
            shuffle=i == 0,
            num_workers=4,
            drop_last=i == 0,
        )
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(datasets[0]) // config.batch_size * config.epochs

    logger.info("Building Model")
    model = Model(config)

    model = model.to(device)

    trainer = Trainer(model)

    best_f1 = 0.0
    best_test_f1 = 0.0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader, device)
        f1 = trainer.eval(i, dev_loader)
        test_f1 = trainer.eval(i, test_loader, is_test=True)
        if f1 > best_f1:
            best_f1 = f1
            best_test_f1 = test_f1
            trainer.save(config.save_path)
    logger.info("Best DEV F1: {:3.4f}".format(best_f1))
    logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
    trainer.load(config.save_path)
    trainer.predict("Final", test_loader, ori_data[-1])
