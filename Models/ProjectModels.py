import json
import os.path
from pathlib import Path
from transformers import AutoModelForSequenceClassification


def lab_2_id():
    j_path = Path(__file__).resolve().parent / "idx_label.json"
    j_dict = {}

    if not os.path.isfile(j_path):
        raise FileNotFoundError("idx_label.json is not present")

    with open(j_path, "r") as f:
        j_dict.update(json.load(f))

    return j_dict


def id_2_lab():
    key_tup, idx_tup = list(zip(*list(lab_2_id().items())))
    return dict(list(zip(idx_tup, key_tup)))


def get_model():
    lab_dict = lab_2_id()

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                               problem_type="multi_label_classification",
                                                               num_labels=len(lab_dict),
                                                               id2label=id_2_lab(),
                                                               label2id=lab_dict)

    return model
