# File: read_files.py
# -------------------
from __future__ import annotations

import json
import pathlib
from typing import Any, TYPE_CHECKING
from datasets import load_dataset
import aiohttp
from transformers import AutoImageProcessor

if TYPE_CHECKING:
    from argparse import Namespace
    from transformers.image_processing_utils import BaseImageProcessor
    from datasets import Dataset


def read_files(args: Namespace) -> list[dict[str, str]]:
    """Takes the path to a data directory and a task, reads the
    JSONL datafiles in the directory and returns a list of
    dictionaries containing all the information used by the
    evaluation.

    Args:
        args(Namespace): A class containing all the information
            necessary to retrive the data. For example: data
            path, task name, etc.

    Returns:
        list[dict[str, str]]: A list of dictionaries containing
            the information to evaluate the given task.
    """
    data = []
    images = None
    image_processor = None
    if args.images_path is not None:
        images = load_dataset(args.images_path, split=args.image_split, storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
        image_processor = AutoImageProcessor.from_pretrained(args.model_path_or_name)
    for filename in args.data_path.iterdir():
        if filename.suffix != ".jsonl":
            continue

        with filename.open("r") as f:
            for line in f:
                data.append(decode(line, filename, args.task, args.full_sentence_scores, images, image_processor))

    del images

    return data


def decode(line: str, file_name: pathlib.Path, task: str, full_sentence_scores: bool, images: Dataset | None, image_processor: BaseImageProcessor | None) -> dict[str, str]:
    """This function takes a line of a JSONL file and returns a
    dictionary of terms to be used by the evaluation.

    Args:
        line(str): A JSONL line from a datafile.
        file_name(pathlib.Path): The file name the line comes
            from.
        task(str): The task we are evaluating, this tells us
            what needs to be imported.
        images(Dataset | None): The collection of images
            associated with the dataset.
        image_processor(BaseImageProcessor): The image
            processor of the model being tested.

    Returns:
        dict[str, str]: A dictionary with values used for
            evaluation.
    """

    raw_dict = json.loads(line.strip())

    if task == "blimp":
        data_dict = decode_blimp(raw_dict, file_name)
    elif task == "ewok":
        data_dict = decode_ewok(raw_dict, full_sentence_scores)
    elif task == "wug":
        data_dict = decode_wug_adj_nominalization(raw_dict)
    elif task == "entity_tracking":
        data_dict = decode_entity_tracking(raw_dict, file_name)
    elif task == "vqa":
        data_dict = decode_vqa(raw_dict, images, image_processor)
    else:
        raise NotImplementedError(f"The task {task} is not implemented! Please implement it or choose one of the implemented tasks.")

    return data_dict


def decode_blimp(raw_dict: dict[str, Any], file_name: pathlib.Path) -> dict[str, str]:
    """This function takes a dictionary of a single datapoint
    of a BLiMP datafile and returns a dictionary of terms to be
    used by the evaluation.

    Args:
        raw_dict(dict[str, Any]): A dictionary from a single
            datapoint of a BLiMP datafile.
        file_name(pathlib.Path): When no UID is mentioned, we
            take the file name.

    Returns:
        dict[str, str]: A dictionary with values used for
            evaluation.
    """
    if "field" in raw_dict:
        pair = {
            "sentences": [raw_dict["sentence_good"], raw_dict["sentence_bad"]],
            "prefixes": [None, None],
            "completions": [raw_dict["sentence_good"], raw_dict["sentence_bad"]],
            "label": 0,
            "field": raw_dict["field"],
            "UID": raw_dict["UID"],
            "linguistics_term": raw_dict["linguistics_term"],
        }
        if pair["field"] == "syntax_semantics":  # Standardizing the style of this field
            pair["field"] = "syntax/semantics"
    else:  # For the supplemetal tasks, there is no field or UID
        pair = {
            "sentences": [raw_dict["sentence_good"], raw_dict["sentence_bad"]],
            "prefixes": [None, None],
            "completions": [raw_dict["sentence_good"], raw_dict["sentence_bad"]],
            "label": 0,
            "field": "supplement",
            "UID": file_name.stem,
            "linguistics_term": "supplement",
        }

    return pair


def decode_ewok(raw_dict: dict[str, Any], full_sentence_scores: bool) -> dict[str, str]:
    """This function takes a dictionary of a single datapoint
    of a EWoK datafile and returns a dictionary of terms to be
    used by the evaluation.

    Args:
        raw_dict(dict[str, Any]): A dictionary from a single
            datapoint of a EWoK datafile.

    Returns:
        dict[str, str]: A dictionary with values used for
            evaluation.
    """
    if full_sentence_scores:
        completions = [" ".join([raw_dict["Context1"], raw_dict["Target1"]]), " ".join([raw_dict["Context1"], raw_dict["Target2"]])]
    else:
        completions = [" " + raw_dict["Target1"], " " + raw_dict["Target2"]]
    pair = {
        "sentences": [" ".join([raw_dict["Context1"], raw_dict["Target1"]]), " ".join([raw_dict["Context1"], raw_dict["Target2"]])],
        "prefixes": [raw_dict["Context1"], raw_dict["Context2"]],
        "completions": completions,
        "label": 0,
        "UID": raw_dict["Domain"],
        "context_type": raw_dict["ContextType"],
        "context_contrast": raw_dict["ContextDiff"],
        "target_contrast": raw_dict["TargetDiff"],
    }

    return pair


def decode_wug_adj_nominalization(raw_dict: dict[str, Any]) -> dict[str, str]:
    """This function takes a dictionary of a single datapoint
    of the wug test datafile and returns a dictionary of terms
    to be used by the evaluation.

    Args:
        raw_dict(dict[str, Any]): A dictionary from a single
            datapoint of a BLiMP datafile.
        file_name(pathlib.Path): When no UID is mentioned, we
            take the file name.

    Returns:
        dict[str, str]: A dictionary with values used for
            evaluation.
    """
    pair = {
        "sentences": raw_dict["sentences"].split('\t'),
        "prefixes": [None],
        "completions": raw_dict["sentences"].split('\t'),
        "ratio": float(raw_dict["ratio"]),
        "label": 0,
        "UID": "wug_adj_nominalization",
    }

    return pair


def decode_entity_tracking(raw_dict: dict[str, Any], file_name: pathlib.Path) -> dict[str, str]:
    """This function takes a dictionary of a single datapoint
    of an Entity Tracking datafile and returns a dictionary of
    terms to be used by the evaluation.

    Args:
        raw_dict(dict[str, Any]): A dictionary from a single
            datapoint of an Entity Tracking datafile.

    Returns:
        dict[str, str]: A dictionary with values used for
            evaluation.
    """
    subset = f'{file_name.stem}_{raw_dict["numops"]}_ops'
    pair = {
        "sentences" : [raw_dict["input_prefix"] + option for option in raw_dict["options"]],
        "prefixes": [raw_dict["input_prefix"] for _ in raw_dict["options"]],
        "completions" : [option for option in raw_dict["options"]],
        "label" : 0,
        "UID" : subset,
    }

    return pair


def decode_vqa(raw_dict: dict[str, Any], images: Dataset, image_processor: BaseImageProcessor) -> dict[str, str]:
    """This function takes a dictionary of a single datapoint
    of the VQA dataset and the associated image and returns a
    dictionary of terms to be used by the evaluation.

    Args:
        raw_dict(dict[str, Any]): A dictionary from a single
            datapoint of the VQA datafile.
        images(Dataset): The collection of images associated of
            the VQA dataset.
        image_processor(BaseImageProcessor): The image
            processor of the model being tested.

    Returns:
        dict[str, str]: A dictionary with values used for
            evaluation.
    """
    pair = {
        "sentences": [" ".join([raw_dict["question"], raw_dict["target_ans"]])] + [" ".join([raw_dict["question"], answer]) for answer in raw_dict["distractors"]],
        "prefixes": [raw_dict["question"] for _ in range(len(raw_dict["distractors"]) + 1)],
        "completions": [" " + raw_dict["target_ans"]] + [" " + answer for answer in raw_dict["distractors"]],
        "label": 0,
        "UID": "VQA",
        "image": image_processor(images[raw_dict["idx_in_hf_dataset"]]["image"].convert("RGB"), return_tensors="pt")["pixel_values"],
    }

    return pair
