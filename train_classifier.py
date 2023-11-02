# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import torch
import os
import time
import random
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats
import pickle


import datasets
from collections import Counter
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, accuracy
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm
from torch.utils.data import WeightedRandomSampler
import numpy as np
from datasets import Dataset, DatasetDict
from metrics import *

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from collections import Counter
from transformers.utils.versions import require_version

try:
    import wandb

    USE_WANDB = True
except:
    USE_WANDB = False

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    
    parser.add_argument(
        "--ind_test_file", type=str, default=None, help="A csv or a json file containing the in-domain test data."
    )

    parser.add_argument(
        "--ood_test_file", type=str, default=None, help="A csv or a json file containing the  out-of-domain test data."
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=229,
        help=(
            "Patience for early stopping"
        ),
    )
    parser.add_argument(
    "--idil_loss",
    action='store_true',
    default=False,
    help="Flag indicating whether to use idil loss. Must be set to True or False."
)

    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="If passed, will predict on test_file",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=0,
        help=(
            "Early stop or not"
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--secondary_model",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        required=False,
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--with_wandb",
        required=False,
        default=True,
        help="Whether to use wandb for logging",
    )


    parser.add_argument(
        "--with_custom_loss",
        type = bool,
        default= True,
        required=False,
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "pkl"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "pkl"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args



def calculate_ranking_loss(logits: torch.Tensor, true_labels: torch.Tensor, num_labels: int, normalize: bool = True,
                           binary_ce_loss: bool = False):
    """
    Function to calculate a pair wise ranking loss where loss is defined as sum( max(0, P(Y' = c| X1)- P(Y' = c| X2))) 
    where X1, X2 are input pairs of the batch, c is class label and Y(X1)!=c, Y(X2)=c.
    """
    if normalize:
        prediction_tensor = torch.nn.functional.softmax(logits, dim=1)
    else:
        prediction_tensor = logits

    # To count the number of pairs where true labels differ
    # positive_negative_pairs = 0
    ranking_loss_func = torch.nn.MarginRankingLoss()
    # b=0.00001

    positive_probs = torch.tensor([]).cuda()
    negative_probs = torch.tensor([]).cuda()


    for i in range(num_labels):
        positive_indices = torch.where(true_labels == i)[0]
        negative_indices = torch.where(true_labels != i)[0]

        negative_probabilities = torch.index_select(prediction_tensor, 0, negative_indices)[:, i]
        for row in positive_indices:
            positive_probs = torch.cat(
                [positive_probs, prediction_tensor[row, i].repeat(negative_probabilities.shape[0]).cuda()], dim=0)
            negative_probs = torch.cat([negative_probs, negative_probabilities.reshape(-1)], dim=0)


    positive_probs=positive_probs.detach()
    loss = ranking_loss_func(positive_probs, negative_probs, torch.ones(negative_probs.shape, dtype=torch.int32).cuda())
    # loss = (loss-b).abs() + b
    loss = loss* torch.sigmoid(loss)   # SiLU loss
    return loss




def main():
    args = parse_args()
    global USE_WANDB

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment

    if args.with_wandb==False:
        USE_WANDB=False

    if USE_WANDB:
        accelerator = (
            Accelerator(log_with="wandb", logging_dir=args.output_dir)
        )
        accelerator.init_trackers("SelPred", init_kwargs={"wandb": {"dir": args.output_dir}})
    else:
        accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.ind_test_file is not None and args.ood_test_file is not None:
            data_files["ind_test"] = args.ind_test_file
            data_files["ood_test"] = args.ood_test_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        if extension == "pkl":
            raw_datasets = load_dataset('pandas', data_files=data_files)
        else:
            raw_datasets = load_dataset(extension, data_files=data_files)
        if args.do_predict:
            ind_true_labels = list(raw_datasets["ind_test"]["label"])
            ood_true_labels = list(raw_datasets["ood_test"]["label"])


    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=not args.use_slow_tokenizer)

    


    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        from_tf=bool(".ckpt" in args.model),
        config=config,
    )




    if args.model == "gpt2":
        tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        print(f"tokenizer pad token is: {tokenizer.pad_token}")
        model.config.pad_token_id = model.config.eos_token_id

    # Preprocessing the datasets
    sentence1_key = "text"
    sentence2_key = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    if args.do_predict:
        ind_test_dataset = processed_datasets["ind_test"]
        ood_test_dataset = processed_datasets["ood_test"]


    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))


    # class_counts = Counter([example["labels"] for example in train_dataset])
    # sample_weights = [1/class_counts[i] for i in [example["labels"] for example in train_dataset] ]
    # sampler = WeightedRandomSampler(weights = sample_weights, num_samples = len(train_dataset), replacement=True)
    train_dataloader = DataLoader(
        train_dataset,  shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    if args.do_predict:
        ind_test_dataloader = DataLoader(ind_test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        ood_test_dataloader = DataLoader(ood_test_dataset, shuffle=False, collate_fn=data_collator,batch_size=args.per_device_eval_batch_size)


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        accelerator.init_trackers("glue_no_trainer", args)

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            resume_step = None
            path = args.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        if "epoch" in path:
            print("\n\n\n\n")
            print(args.num_train_epochs)
            args.num_train_epochs -=  int(path.split("epoch_")[1].strip()) #int(path.replace("epoch_", ""))
            print(args.num_train_epochs)
        else:
            resume_step = int(path.replace("step_", ""))
            args.num_train_epochs -= resume_step // len(train_dataloader)
            resume_step = (args.num_train_epochs * len(train_dataloader)) - resume_step

    counter = 0
    patience = args.patience
    best_score = None
    early_stop = False

    if USE_WANDB:
        wandb.watch(model, log='all', log_freq=100)

    
    if not args.do_predict:
        # Running the training loop of the code
        print(args.num_train_epochs)
        fpr_list=[]
        ce_loss_list=[]
        idil_loss_list=[]        
        step_num = 0

        for epoch in range(args.num_train_epochs):
            

            if early_stop:
                logger.info("Early stopping!!")
                break
            model.train()
            entropy_list = []
            if args.with_tracking:
                total_loss = 0
                custom_loss = 0


            for step, batch in enumerate(train_dataloader):
                
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == 0 and  resume_step is not None and step < resume_step:
                    continue
                    
                step_num+=1
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                if args.idil_loss:
                    custom_ranking_loss = calculate_ranking_loss(logits, batch.labels, num_labels, True, False)
                    # custom_ranking_loss = calculate_ranking_loss(logits, batch.labels, num_labels, True, True)
                    loss = custom_ranking_loss

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()

                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if USE_WANDB:
                    counter_dict = Counter( batch["labels"].tolist())
                    value_counts = np.array(list(counter_dict.values()))
                    value_probs = value_counts / len( batch["labels"].tolist())
                    entropy = scipy.stats.entropy(value_probs, base=2)
                    entropy_list.append(entropy)
                    if args.with_custom_loss:
                        wandb.log({'loss': loss.item(),"ce_loss": outputs.loss.item(), "custom_ranking_loss": custom_ranking_loss.item(), "entropy": entropy})
                    else:
                        wandb.log({'loss': loss.item(), "ce_loss": outputs.loss.item()})
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break

            if USE_WANDB:
                wandb.log({"avg_entropy": np.mean(entropy_list)})

            
            model.eval()
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )

            eval_metric = metric.compute()
            logger.info(f"epoch {epoch}: {eval_metric}")

            if args.early_stop:
                if best_score is None:
                    best_score = eval_metric["accuracy"]
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(args.output_dir)
                elif eval_metric["accuracy"] <= best_score:
                    counter += 1
                    logger.info(f'EarlyStopping counter: {counter} out of {patience}')
                    if counter >= patience:
                        early_stop = True
                else:
                    counter = 0
                    best_score = eval_metric["accuracy"]
                    logger.info("Best dev result: {}".format(best_score))
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(args.output_dir)

            if args.with_tracking:
                accelerator.log(
                    {
                        "accuracy" if args.task_name is not None else "glue": eval_metric,
                        "train_loss": total_loss,
                        "epoch": epoch,
                    },
                    step=completed_steps,
                )

            if args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)

            if args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

        torch.save(torch.tensor(fpr_list), f'{args.output_dir}fpr_list.pt')
        print(f"len ce list: {len(ce_loss_list)} , len idil loss list: {len(idil_loss_list)}")
        torch.save(torch.tensor(ce_loss_list), f'{args.output_dir}ce_list.pt')
        torch.save(torch.tensor(idil_loss_list), f'{args.output_dir}idil_list.pt')
        if not args.early_stop:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
            del unwrapped_model

        del model
        del tokenizer
        torch.cuda.empty_cache()

    if args.do_predict:
        logger.info("***** Running testing *****")
        logger.info(f"  Indomain Num examples = {len(ind_test_dataset)}")
        logger.info(f"  Out-of-domain Num examples = {len(ood_test_dataset)}")

        config = AutoConfig.from_pretrained(args.model, num_labels=num_labels,
                                            finetuning_task=args.task_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=not args.use_slow_tokenizer)
        if args.resume_from_checkpoint is None:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model,
                from_tf=bool(".ckpt" in args.model),
                config=config,
            )

        model, ind_test_dataloader, = accelerator.prepare(model, ind_test_dataloader)
        model, ood_test_dataloader, = accelerator.prepare(model, ood_test_dataloader)

        model.eval()
        pred_ind = []
        pred_ood = []

        ind_pred_logits = torch.Tensor().cuda()
        ood_pred_logits = torch.Tensor().cuda()
        ind_ground_truth  = []
        ood_ground_truth  = []

        for step, batch in enumerate(ind_test_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            ind_pred_logits = torch.cat((ind_pred_logits, outputs.logits ), axis=0)
            pred_ind += list(predictions)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )
            
            ind_ground_truth.extend(batch["labels"])

        for step, batch in enumerate(ood_test_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            ood_pred_logits = torch.cat((ood_pred_logits, outputs.logits ), axis=0)
            pred_ood += list(predictions)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )
            
            ood_ground_truth.extend(batch["labels"])
        
        # saving the prediction tensors
        print(f"Pred_logits_shape : {ind_pred_logits.shape}")
        torch.save(ind_pred_logits, f'{args.output_dir}ind_pred_logits.pt')
        print(f"Ground Truth_shape : {len(ind_ground_truth)}")
        # saving the ground truth
        torch.save(torch.tensor(ind_ground_truth,dtype=torch.float64), f'{args.output_dir}ind_labels.pt')


        # saving the prediction tensors
        print(f"Pred_logits_shape : {ood_pred_logits.shape}")
        torch.save(ood_pred_logits, f'{args.output_dir}ood_pred_logits.pt')
        print(f"Ground Truth_shape : {len(ood_ground_truth)}")
        # saving the ground truth
        torch.save(torch.tensor(ood_ground_truth,dtype=torch.float64), f'{args.output_dir}ood_labels.pt')

            
        logits_ind = ind_pred_logits.cpu()
        ind = torch.Tensor([0]*logits_ind.shape[0])

        logits_ood = ood_pred_logits.cpu()
        ood = torch.Tensor([1]*logits_ood.shape[0])


        
        logits  = torch.cat((logits_ind, logits_ood ), axis=0) 
        is_ood = torch.cat((ind, ood), axis=0)

        # Print metrics
        fpr, detection_err, auroc, aupr = calc_metrics(logits, is_ood)


if __name__ == "__main__":
    main()
