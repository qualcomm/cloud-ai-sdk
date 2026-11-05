####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

import math

import numpy as np


def parse_duration_buckets(value):
    items = value if isinstance(value, (list, tuple)) else value.split(",")
    buckets = sorted({int(str(item).strip()) for item in items if str(item).strip()})
    if not buckets:
        raise ValueError("duration bucket list must contain at least one bucket")
    return buckets


def select_duration_bucket(required_duration, buckets):
    required_duration = int(math.ceil(float(required_duration)))
    return next((bucket for bucket in buckets if required_duration <= bucket), buckets[-1])


def parse_device_ids(value):
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(device_id.strip()) for device_id in value.split(",") if device_id.strip()]


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


def build_transformer_batch_inputs(noise, cat_mel_text, cat_mel_text_drop, input_length):
    return {
        "noise": np.ascontiguousarray(np.repeat(noise, 2, axis=0)),
        "cat_mel_text": np.ascontiguousarray(
            np.concatenate((cat_mel_text, cat_mel_text_drop), axis=0)
        ),
        "input_lengths": np.ascontiguousarray(np.repeat(input_length, 2)),
    }


def convert_char_to_pinyin(text_list, polyphone=True):
    import jieba
    from pypinyin import Style, lazy_pinyin

    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )

    def is_chinese(c):
        return "㄀" <= c <= "鿿"

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


def list_str_to_idx(text, vocab_char_map, padding_value=-1):
    import torch

    get_idx = vocab_char_map.get
    list_idx_tensors = [torch.tensor([get_idx(c, 0) for c in t], dtype=torch.int32) for t in text]
    return torch.nn.utils.rnn.pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
