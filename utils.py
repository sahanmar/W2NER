import logging
import time
from collections import defaultdict, deque
from logging import Logger
from typing import Any

import numpy as np


def get_logger(dataset: str) -> Logger:
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def convert_index_to_text(index: list[int], type: int) -> str:
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text: str) -> tuple[list[int], int]:
    index, type = text.split("-#-")
    indices = [int(x) for x in index.split("-")]
    return indices, int(type)


def decode(
    outputs: np.ndarray, entities: list[set[str]], length: np.ndarray
) -> tuple[int, int, int, list[Any]]:
    class Node:
        def __init__(self) -> None:
            self.THW: list[Any] = []  # [(tail, type)]
            self.NNW: dict[Any, Any] = defaultdict(set)  # {(head,tail): {next_index}}

    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = []
    q: deque[list[int]] = deque()
    for instance, ent_set, l in zip(outputs, entities, length):
        predicts = []
        nodes = [Node() for _ in range(l)]
        for cur in reversed(range(l)):
            heads = []
            for pre in range(cur + 1):
                # THW
                if instance[cur, pre] > 1:
                    nodes[pre].THW.append((cur, instance[cur, pre]))
                    heads.append(pre)
                # NNW
                if pre < cur and instance[pre, cur] == 1:
                    # cur node
                    for head in heads:
                        nodes[pre].NNW[(head, cur)].add(cur)
                    # post nodes
                    for head, tail in nodes[cur].NNW.keys():
                        if tail >= cur and head <= pre:
                            nodes[pre].NNW[(head, tail)].add(cur)
            # entity
            for tail, type_id in nodes[cur].THW:
                if cur == tail:
                    predicts.append(([cur], type_id))
                    continue
                q.clear()
                q.append([cur])
                while len(q) > 0:
                    chains = q.pop()
                    for idx in nodes[chains[-1]].NNW[(cur, tail)]:
                        if idx == tail:
                            predicts.append((chains + [idx], type_id))
                        else:
                            q.append(chains + [idx])

        text_predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in text_predicts])
        ent_r += len(ent_set)
        ent_p += len(text_predicts)
        ent_c += len(text_predicts.intersection(ent_set))
    return ent_c, ent_p, ent_r, decode_entities


# TODO do not return arguments
def cal_f1(c: float, p: float, r: float) -> tuple[float, float, float]:
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r
