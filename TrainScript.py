import numpy as np
import torch
from BenchKit.Tracking.Helpers import upload_model_checkpoint
from BenchKit.Tracking.Tracker import get_tensorboard_tracker
from accelerate import Accelerator
from BenchKit.Data.Helpers import get_dataloader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_scheduler
from Datasets.ProjectDatasets import DatasetTest
from Models.ProjectModels import get_model, id_2_lab
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as f


def get_f1_score(output_arr, target_arr) -> dict[str: float]:

    f_one = f1_score(target_arr, output_arr, average=None)

    id_dict = id_2_lab()

    return {id_dict[idx]: i for idx, i in enumerate(f_one)}


def get_accuracy(output_arr, target_arr) -> dict[str: float]:
    output_arr = np.transpose(output_arr)
    target_arr = np.transpose(target_arr)

    f_one = f1_score(target_arr, output_arr, average=None)

    id_dict = id_2_lab()

    return {id_dict[idx]: i for idx, i in enumerate(f_one)}


def train_one_epoch(accelerate: Accelerator,
                    train_dl: DataLoader,
                    model,
                    optim,
                    length,
                    scheduler: LambdaLR):
    model.train()

    total_loss = 0
    count = 0

    for batch in tqdm(train_dl, colour="blue", total=length, disable=not accelerate.is_local_main_process):
        optim.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss
        accelerate.backward(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()
        scheduler.step()

        full_loss: torch.Tensor = accelerate.gather_for_metrics(loss)
        total_loss += full_loss.item()
        count += 1

    accelerate.print("train", total_loss / count)


def validate_one_epoch(accelerate: Accelerator,
                       test_dl: DataLoader,
                       model,
                       length,
                       threshold=0.5):
    model.eval()
    target_arr = np.array([np.NaN])
    output_arr = np.array([np.NaN])

    with torch.no_grad():
        for batch in tqdm(test_dl, colour="blue", total=length, disable=not accelerate.is_local_main_process):
            output_dict = model(**batch)

            output = f.sigmoid(output_dict.logits)
            output = output.cpu().detach().numpy()
            zero_arr = np.zeros(output.shape)
            zero_arr[np.where(output >= threshold)] = 1
            target = batch.labels.cpu().detach().numpy()

            target_arr = target if np.isnan(target_arr).any() else np.concatenate((target_arr, target), axis=0)
            output_arr = zero_arr if np.isnan(output_arr).any() else np.concatenate((output_arr, zero_arr), axis=0)

        f1_dict = get_f1_score(output_arr, target_arr)
        accuracy_dict = {"accuracy": accuracy_score(target_arr, output_arr)}
        accelerate.print(f"The f1 score per class is {f1_dict}")
        accelerate.print(f"The accuracy is {accuracy_dict}")


def main():
    config = {
        "lr": 2e-5,
        "batch_size": 8,
        "epochs": 5,
        "weight_decay": 0.01,
    }

    model = get_model()

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    train_chunker = DatasetTest()
    val_chunker = DatasetTest()

    train_dataset: DataLoader = get_dataloader(train_chunker,
                                               "TWT_Train_V1",
                                               batch_size=config["batch_size"])

    val_dataset: DataLoader = get_dataloader(val_chunker,
                                             "TWT_val_V1",
                                             batch_size=config["batch_size"])

    length = int(np.ceil(train_dataset.dataset.length / config["batch_size"]))
    val_length = int(np.ceil(val_dataset.dataset.length / config["batch_size"]))

    scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=length * config["epochs"]
    )

    acc, train_dataset, val_dataset, model, optim, scheduler = get_tensorboard_tracker(config,
                                                                                       train_dataset,
                                                                                       val_dataset,
                                                                                       model,
                                                                                       optimizer,
                                                                                       scheduler)

    for e in range(config["epochs"]):
        train_one_epoch(acc, train_dataset, model, optim, length, scheduler)
        validate_one_epoch(acc, val_dataset, model, val_length)
        upload_model_checkpoint(acc, "test-checkpoint")


if __name__ == '__main__':
    main()
