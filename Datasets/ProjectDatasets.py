import json
import os
from pathlib import Path

import torch
from BenchKit.Data.Datasets import ProcessorDataset, IterableChunk
from BenchKit.Data.FileSaver import JsonFile, TextFile
from datasets import load_dataset
from transformers import AutoTokenizer


# Write your datasets or datapipes here

class TweetProcessor(ProcessorDataset):

    def __init__(self,
                 set_type: int):

        super().__init__()
        dataset_selector = {idx: i for idx, i in enumerate(["train", "test", "validation"])}

        if set_type not in list(dataset_selector.keys()):
            raise RuntimeError(f"{set_type} is not a valid option, "
                               f"the valid options are {list(dataset_selector.keys())}")

        dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")
        self.dataset = dataset[dataset_selector[set_type]]

        labels = [i for i in self.dataset.features if i not in ["ID", "Tweet"]]
        labels = {i: idx for idx, i in enumerate(labels)}

        j_path = Path(__file__).resolve().parent.parent / "Models" / "idx_label.json"
        if not os.path.isfile(j_path):
            with open(j_path, "w") as f:
                json.dump(labels, f)

        self.label_dict = labels
        self.jf = JsonFile()
        self.txt = TextFile()

    def _get_savers(self):
        return self.jf, self.txt

    def _get_data(self,
                  idx: int):

        data_dict = self.dataset[idx]
        data_dict.pop("ID")

        tweet = data_dict.pop("Tweet")
        z_list = [0.0] * len(self.label_dict)

        for k, v in data_dict.items():
            if v:
                z_list[self.label_dict[k]] = 1.0

        self.txt.append(tweet)
        self.jf.append(z_list)

    def __len__(self):
        return self.dataset.num_rows


class DatasetTest(IterableChunk):

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def unpack_data(self,
                    idx):

        label, txt = super().unpack_data(idx)

        ret_dict = self.tokenizer(txt,
                                  padding="max_length",
                                  truncation=True,
                                  max_length=128,
                                  return_tensors="pt")

        for i in ret_dict:
            ret_dict[i] = torch.squeeze(ret_dict[i], dim=0)

        ret_dict["labels"] = torch.Tensor(label)

        return ret_dict


def main():
    """
    This method returns all the necessary components to build your dataset
    You will return a list of tuples, each tuple represents a different dataset
    The elements of the tuple represent the components to construct your dataset
    Element one will be your ProcessorDataset object
    Element two will be your IterableChunk class
    Element three will be the name of your Dataset
    Element four will be all the args needed for your Iterable Chunk as a list
    Element five will be all the kwargs needed for your Iterable Chunk as a Dict
    """

    du_train = TweetProcessor(0)
    du_test = TweetProcessor(1)
    du_val = TweetProcessor(2)

    return [
        (du_train, DatasetTest, "TWT_Train_V1", [], {}),
        (du_test, DatasetTest, "TWT_Test_V1", [], {}),
        (du_val, DatasetTest, "TWT_val_V1", [], {})
    ]
