import numpy as np
from shap.utils import safe_isinstance
from shap.utils.transformers import (
    SENTENCEPIECE_TOKENIZERS,
    getattr_silent,
)
from dataclasses import dataclass
from typing import List
import yaml
from warnings import warn
import yaml

def row_to_string(row, cols):
    row["text"] = " | ".join(f"{col}: {row[col]}" for col in cols)
    return row


def multiple_row_to_string(row, cols, multiplier=1, nodesc=False):
    row["text"] = " | ".join(
        f'{col}: {(str(row[col]) + " ")*multiplier}' for col in cols
    )
    if not nodesc:
        row["text"] = row["text"] + " | Description: " + row["Description"]
    return row


def prepare_text(dataset, version, di, reverse=False, model_name=None):
    """This is all for preparing the text part of the dataset
    Could be made more robust by referring to dataset_info.py instead"""

    # Used for reorder versions
    if model_name == "bert-base-uncased":
        model_code = "bert"
    elif model_name == "distilbert-base-uncased":
        model_code = "disbert"
    elif model_name == "distilroberta-base":
        model_code = "drob"
    elif model_name == "microsoft/deberta-v3-small":
        model_code = "deberta"
    else:
        model_code = None

    if version == "all_as_text":
        cols = di.tab_cols + di.text_cols
        dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
        return dataset
    elif version == "text_col_only":
        if len(di.text_cols) == 1:
            # dataset rename column
            dataset = dataset.rename_column(di.text_cols[0], "text")
        else:
            dataset = dataset.map(row_to_string, fn_kwargs={"cols": di.text_cols})
    elif version == "all_as_text_base_reorder":
        cols = di.base_reorder_cols[model_code]
        cols = cols[::-1] if reverse else cols
        dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
        return dataset
    elif version == "all_as_text_tnt_reorder":
        cols = di.tnt_reorder_cols[model_code]
        cols = cols[::-1] if reverse else cols
        dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
        return dataset

    else:
        raise ValueError(
            f"Unknown dataset type ({di.config}) and version ({version}) combination"
        )


def format_text_pred(scores, order):
    # scores = [p["score"] for p in pred]
    # order = [int(p["label"][6:]) for p in pred]
    return np.array(
        [scores[i] for i in sorted(range(len(scores)), key=lambda x: order[x])]
    )


def format_fts_for_plotting(fts, tab_data):
    for i in range(len(tab_data)):
        fts[i] = fts[i] + f" = {tab_data[i]}   "
    # for j in range(len(tab_data), len(fts)):
    #     fts[j] = fts[j] + ""
    return fts


def text_ft_index_ends(text_fts, tokenizer):
    lens = []
    sent_indices = []
    for idx, col in enumerate(text_fts):
        # First text col
        if lens == []:
            tokens, token_ids = token_segments(str(col), tokenizer)
            # -1 as we don't use SEP tokens (unless it's the only text col)
            also_last = 1 if len(text_fts) == 1 else 0
            token_len = len(tokens) - 1 + also_last
            lens.append(token_len - 1)
            sent_indices.extend([idx] * token_len)
        # Last text col
        elif idx == len(text_fts) - 1:
            tokens, token_ids = token_segments(str(col), tokenizer)
            # -1 for CLS tokens
            token_len = len(tokens) - 1
            lens.append(lens[-1] + token_len)
            sent_indices.extend([idx] * token_len)
        # Middle text cols
        else:
            tokens, token_ids = token_segments(str(col), tokenizer)
            # -2 for CLS and SEP tokens
            token_len = len(tokens) - 2
            lens.append(lens[-1] + token_len)
            sent_indices.extend([idx] * token_len)

    return lens[:-1]


def token_segments(s, tokenizer):
    """Same as Text masker"""
    """ Returns the substrings associated with each token in the given string.
    """

    try:
        token_data = tokenizer(s, return_offsets_mapping=True)
        offsets = token_data["offset_mapping"]
        offsets = [(0, 0) if o is None else o for o in offsets]
        parts = [
            s[offsets[i][0] : max(offsets[i][1], offsets[i + 1][0])]
            for i in range(len(offsets) - 1)
        ]
        parts.append(s[offsets[len(offsets) - 1][0] : offsets[len(offsets) - 1][1]])
        return parts, token_data["input_ids"]
    except (
        NotImplementedError,
        TypeError,
    ):  # catch lack of support for return_offsets_mapping
        token_ids = tokenizer(s)["input_ids"]
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
        else:
            tokens = [tokenizer.decode([id]) for id in token_ids]
        if hasattr(tokenizer, "get_special_tokens_mask"):
            special_tokens_mask = tokenizer.get_special_tokens_mask(
                token_ids, already_has_special_tokens=True
            )
            # avoid masking separator tokens, but still mask beginning of sentence and end of sentence tokens
            special_keep = [
                getattr_silent(tokenizer, "sep_token"),
                getattr_silent(tokenizer, "mask_token"),
            ]
            for i, v in enumerate(special_tokens_mask):
                if v == 1 and (
                    tokens[i] not in special_keep or i + 1 == len(special_tokens_mask)
                ):
                    tokens[i] = ""

        # add spaces to separate the tokens (since we want segments not tokens)
        if safe_isinstance(tokenizer, SENTENCEPIECE_TOKENIZERS):
            for i, v in enumerate(tokens):
                if v.startswith("_"):
                    tokens[i] = " " + tokens[i][1:]
        else:
            for i, v in enumerate(tokens):
                if v.startswith("##"):
                    tokens[i] = tokens[i][2:]
                elif v != "" and i != 0:
                    tokens[i] = " " + tokens[i]

        return tokens, token_ids
