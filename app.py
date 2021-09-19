#streamlit run dashboard.py --server.port 4000 --server.address 0.0.0.0 --server.baseUrlPath test
import streamlit as st
import plotly.graph_objects as go
from annotated_text import annotated_text
import json
# import SessionState
import numpy as np
import tensorflow as tf
import torch as pt
import logging
import json
from datetime import datetime
import faulthandler
import glob
import json
import logging
import math
import os
import random
import sys
from json import JSONEncoder

import dataclasses
from pathlib import Path

import wandb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    multilabel_confusion_matrix,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import compute_class_weight

import torch
from sklearn.utils.extmath import softmax
from torch.cuda.amp import autocast
from torch import nn
from torch.nn import CrossEntropyLoss

from datasets import load_dataset, concatenate_datasets
import transformers
import transformers.adapters.composition as ac
from transformers import (
    AdapterConfig,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    EarlyStoppingCallback,
    HfArgumentParser,
    LongformerForSequenceClassification,
    MultiLingAdapterArguments,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

import adapter_script.LongBert
import adapter_script.Longformer
from adapter_script.HierarchicalBert import HierarchicalBert
from adapter_script.data_arguments import DataArguments
from adapter_script.model_arguments import ModelArguments, long_input_bert_types

class JsonFormatter(logging.Formatter):

    def format(self, record):
        return json.dumps({"time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "name": record.name,
                    "level": record.levelname, "message": record.msg})

@st.cache(allow_output_mutation=True)
def get_logger():
    logger = logging.getLogger('allennlp')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('allennlp.log')
    #formatter = logging.Formatter('time="%(asctime)s" name=%(name)s level=%(levelname)s option="%(option)s" message="%(message)s"')
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)
    return logger


query_params = st.experimental_get_query_params()
app_state = st.experimental_get_query_params()

# session_state = SessionState.get(first_query_params=query_params, conversation=Conversation())
# first_query_params = session_state.first_query_params

options = ["Welcome", "Criticality Detection"]

# default = options.index(first_query_params["option"][0]) if "option" in app_state else 0

option = st.sidebar.radio(
        'Select area of interest',
        options)#, index=default)

app_state["option"] = option
st.experimental_set_query_params(**app_state)

import torch
from transformers import AutoTokenizer, AutoModel

if option == "Welcome":
    '''
    # Welcome to Justice Served Team
    ## Our Team Members are Joel niklaus, Aswin Subramanian, Connor Boyle and Ade Romadhony
    - [Criticality Detection]

    [source code of this booth](path to be given)
    `  `  
    `  `  
    `  `  
    '''
    html_string = '''
                    <p style="font-size:80%;">
                        <br>
                        <a href="mailto:"></a><br>
                        <a href="mailto:"></a><br>
                        <a href="mailto:"></a><br>
                        <a href="mailto:"></a>
                    </p>
                    '''

    st.markdown(html_string, unsafe_allow_html=True)


if option == "Criticality Detection":
    '''
    ## Criticality Detection for legal cases
    All judicial cases need to be treated with fairness. But some cases require a more serious judiciary at the apex level due to the
    magnitude of the seriouseness involed in the case. This model is used to predict which legal cases are liked to be tried in the apex court of Swiss legal courts. 
    Please make sure the input file is in txt format.
    '''
    @st.cache
    def get_text(file):
        import os
        import sys
        # print("Printing file:", file)
        data = file.read()
        data = data.decode("utf-8") 
        # print("Printing info:", data)
        return data

    def get_prediction(file, text):
        # keep the model. tokenizer and all config in the model-dir folder in the same level as this script
        from pathlib import Path
        model_dir = "D:/GDrive/Projects/Hackathons/justice-served/model_dir/"# str(Path(os.getcwd()).joinpath("model_dir"))
        # tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # umberto = AutoModel.from_pretrained(model_dir)
        # config = AutoConfig.from_pretrained(model_dir)
        long_input_bert_types = ['hierarchical']
        

        finetuning_task = "text-classification"  
        TOKENIZERS_PARALLELISM = "True"
        # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # use this when debugging

        model_types = ['distilbert', 'bert', 'roberta', 'camembert']
        problem_type="single_label_classification"
        evaluation_language ='it'
        long_input_bert_type = "hierearchical"
        languages = ['de', 'fr', 'it']
        max_seq_length = 2048
        use_adapters = True
        device = "cuda"
        pad_to_max_length= True
        do_lower_case = True
        do_predict = True
        max_predict_samples =10
        prediction_threshold = False
        tune_hyperparams=False
        max_train_samples = 0
        from transformers import Trainer, TrainingArguments
        training_args = TrainingArguments(
        adafactor=False,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=None,
        debug=[],
        deepspeed=None,
        disable_tqdm=False,
        do_eval=False,
        do_predict=True,
        do_train=False,
        eval_accumulation_steps=32,
        eval_steps=500,
        evaluation_strategy="steps",
        fp16=False,
        fp16_backend="auto",
        fp16_full_eval=False,
        fp16_opt_level="O1",
        gradient_accumulation_steps=32,
        greater_is_better=False,
        group_by_length=True,
        ignore_data_skip=False,
        label_names=None,
        label_smoothing_factor=0.0,
        learning_rate=3e-05,
        length_column_name="length",
        load_best_model_at_end=True,
        local_rank=-1,
        logging_first_step=False,
        logging_steps=500,
        logging_strategy="steps",
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        max_steps=-1,
        metric_for_best_model="eval_loss",
        no_cuda=False,
        num_train_epochs=5.0,
        output_dir="model_dir",
        overwrite_output_dir=True,
        past_index=-1,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        prediction_loss_only=False,
        push_to_hub=False,
        push_to_hub_model_id=123,
        push_to_hub_organization=None,
        push_to_hub_token=None,
        remove_unused_columns=True,
        report_to=[],
        resume_from_checkpoint=None,
        run_name="train",
        save_on_each_node=False,
        save_steps=500,
        save_strategy="epoch",
        save_total_limit=3,
        seed=123,
        sharded_ddp=[],
        skip_memory_metrics=True,
        tpu_metrics_debug=False,
        tpu_num_cores=None,
        use_legacy_prediction_loop=False,
        warmup_ratio=0.0,
        warmup_steps=0,
        weight_decay=0.0,
        )
        print("This script only supports PyTorch models!")

        seed = 123
        evaluation_language="de"
        # Set seed before initializing model.
        set_seed(seed)

        langs = [evaluation_language]
        # if evaluation_language == 'all':
        #     langs = languages
        #     train_datasets = []
        #     eval_datasets = []
        #     predict_datasets = []

        assert len(langs) > 0
        print("file in hand:", file)
        for lang in langs:
            # In distributed training, the load_dataset function guarantees that only one local process can concurrently
            # download the dataset.
            # Labels: they will get overwritten if there are multiple languages
            with open(f'labels.json', 'r') as f:
                label_dict = json.load(f)
                label_dict['id2label'] = {int(k): v for k, v in label_dict['id2label'].items()}
                label_dict['label2id'] = {k: int(v) for k, v in label_dict['label2id'].items()}
                label_list = list(label_dict["label2id"].keys())
            num_labels = len(label_list)

        #     if evaluation_language == 'all':
        #         predict_datasets.append(predict_dataset)

        # if evaluation_language == 'all':
        #     predict_dataset = concatenate_datasets(predict_datasets)

        # Load pretrained model and tokenizer
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        finetuning_task = "text-classification"
        config = AutoConfig.from_pretrained(
            model_dir,
            num_labels=num_labels,
            id2label=label_dict["id2label"],
            label2id=label_dict["label2id"],
            finetuning_task=finetuning_task,
            problem_type="single_label_classification",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            do_lower_case=do_lower_case,
            )

        max_length = max_seq_length
        if long_input_bert_type == 'hierarchical':
            max_segment_length = 512  # because RoBERTa has 514 max_seq_length and not 512
            max_segments = math.ceil(max_length / max_segment_length)
            # we need to make space for adding the CLS and SEP token for each segment
            max_length -= max_segments * 2

        def get_encoder_and_classifier(model):
            if config.model_type not in model_types:
                raise ValueError(f"{config.model_type} is not supported. "
                                f"Please use one of the supported model types {model_types}")

            if config.model_type == 'distilbert':
                encoder = model.distilbert
            if config.model_type == 'bert':
                encoder = model.bert
            if config.model_type in ['camembert', 'xlm-roberta']:
                encoder = model.roberta

            classifier = model.classifier
            # classifier = model.heads
            return encoder, classifier

        def model_init():
            model_dir = "D:/GDrive/Projects/Hackathons/justice-served/model_dir/"
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                config=config,
                )
            print("model loaded inside model init")

            if use_adapters:
                # Setup adapters
             train_adapter = True
            if train_adapter:
                task_name = "glue"
                adapter_config_p = "pfeiffer"
                # check if adapter already exists, otherwise add it
                if task_name not in model.config.adapters:
                    # resolve the adapter config
                    adapter_config = AdapterConfig.load(
                        adapter_config_p,
                        non_linearity=None,
                        reduction_factor=None,
                    )
                    # load a pre-trained from Hub if specified
                    load_adapter = ""
                    if load_adapter:
                        model.load_adapter(
                            load_adapter,
                            config=adapter_config_p,
                            load_as=task_name,
                        )
                    # otherwise, add a fresh adapter
                    else:
                        model.add_adapter(task_name, config=adapter_config_p)
                # optionally load a pre-trained language adapter
                load_lang_adapter = None
                if load_lang_adapter:
                    # resolve the language adapter config
                    lang_adapter_config = AdapterConfig.load(
                        lang_adapter_config,
                        non_linearity=lang_adapter_non_linearity,
                        reduction_factor=lang_adapter_reduction_factor,
                    )
                    # load the language adapter from Hub
                    lang_adapter_name = model.load_adapter(
                        load_lang_adapter,
                        config=lang_adapter_config,
                        load_as=language,
                    )
                else:
                    lang_adapter_name = None
                # Freeze all model weights except of those of this adapter
                model.train_adapter([task_name])
                # Set the adapters to be used in every forward pass
                if lang_adapter_name:
                    model.set_active_adapters(ac.Stack(lang_adapter_name, task_name))
                else:
                    model.set_active_adapters(task_name)
            else:
                if load_adapter or load_lang_adapter:
                    raise ValueError(
                        "Adapters can only be loaded in adapters training mode."
                        "Use --train_adapter to enable adapter training"
                    )

            if long_input_bert_type in long_input_bert_types:
                if long_input_bert_type not in ['bigbird']: # nothing to do for bigbird
                    encoder, classifier = get_encoder_and_classifier(model)

                    if long_input_bert_type == 'hierarchical':
                        long_input_bert = HierarchicalBert(encoder,
                                                        max_segments=max_segments,
                                                        max_segment_length=max_segment_length,
                                                        cls_token_id=tokenizer.cls_token_id,
                                                        sep_token_id=tokenizer.sep_token_id,
                                                        device=device,
                                                        seg_encoder_type='lstm')

                    if long_input_bert_type == 'long':
                        long_input_bert = LongBert.resize_position_embeddings(encoder,
                                                                            max_length=max_length,
                                                                            device=device)

                    if long_input_bert_type in ['hierarchical', 'long']:
                        if config.model_type == 'distilbert':
                            model.distilbert = long_input_bert

                        if config.model_type == 'bert':
                            model.bert = long_input_bert

                        if config.model_type in ['camembert', 'xlm-roberta']:
                            model.roberta = long_input_bert
                            if long_input_bert_type == 'hierarchical':
                                dense = nn.Linear(config.hidden_size, config.hidden_size)
                                dense.load_state_dict(classifier.dense.state_dict())  # load weights
                                dropout = nn.Dropout(config.hidden_dropout_prob).to(device)
                                out_proj = nn.Linear(config.hidden_size, config.num_labels).to(device)
                                out_proj.load_state_dict(classifier.out_proj.state_dict())  # load weights
                                model.classifier = nn.Sequential(dense, dropout, out_proj).to(device)

                    if last_checkpoint or not do_train:
                        # Make sure we really load all the weights after we modified the models
                        model_dir = f'{model_name_or_path}/model.bin'
                        print(f"loading file {model_dir}")
                        model.load_state_dict(torch.load(model_dir))

            return model

        # Preprocessing the datasets
        # Padding strategy
        if pad_to_max_length:
            padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            # IMPORTANT: Can lead to problem with HierarchicalBert
            padding = "longest"

        add_special_tokens = True
        if long_input_bert_type == 'hierarchical':
            add_special_tokens = False  # because we split it internally and then add the special tokens ourselves

        def labels_to_bools(labels):
            return [tl == 1 for tl in labels]

        def preds_to_bools(preds):
            return [pl > prediction_threshold for pl in preds]

        def process_results(preds, labels):
            preds = preds[0] if isinstance(preds, tuple) else preds
            probs = softmax(preds)
            if problem_type == 'single_label_classification':
                preds = np.argmax(preds, axis=1)
            return preds, labels, probs

        # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
        # predictions and label_ids field) and has to return a dictionary string to float.
        def compute_metrics(p: EvalPrediction):
            labels = p.label_ids
            preds = p.predictions
            preds, labels, probs = process_results(preds, labels)

            accuracy = accuracy_score(labels, preds)
            # weighted averaging is a better evaluation metric for imbalanced label distributions
            precision, recall, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            f1_micro = f1_score(labels, preds, average='micro')
            f1_macro = f1_score(labels, preds, average='macro')
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_weighted': f1_weighted,
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
            }

        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        if pad_to_max_length:
            data_collator = default_data_collator
        elif fp16:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None
        
        # Initialize our Trainer
        do_eval=False
        do_train = False
        trainer = Trainer(
            model=model_init() if not tune_hyperparams else None,
            model_init=model_init if tune_hyperparams else None,
            args=training_args,
            train_dataset=train_dataset if do_train else None,
            eval_dataset=eval_dataset if do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )
        print("Model and tokenizer loaded")

        def predict(text):
            # Tokenize the texts
            tokenized = tokenizer(text, padding=padding, truncation=True,
                                max_length=max_length, add_special_tokens=add_special_tokens,
                                return_token_type_ids=True)

            # # Map labels to IDs
            # if problem_type == 'single_label_classification':
            #     if label_dict["label2id"] is not None and "label" in batch:
            #         tokenized["label"] = [label_dict["label2id"][l] for l in batch["label"]]

            print("printing tokwenizedL:", tokenized)
            print("input_ids:", tokenized["input_ids"])
            print("token_type_ids:", tokenized["token_type_ids"])
            print("attention_mask:", tokenized["input_ids"])
            print("input_ids:", len(tokenized["input_ids"]))
            print("token_type_ids:", len(tokenized["token_type_ids"]))
            print("attention_mask:", len(tokenized["attention_mask"]))
            prediction_inputs = torch.tensor(tokenized["input_ids"]),
            # print(prediction_inputs.shape)
            prediction_token_type_ids = torch.tensor(tokenized["token_type_ids"]),
            # print(prediction_token_type_ids.shape)
            prediction_masks = torch.tensor(tokenized["attention_mask"]),
            # prediction_labels = torch.tensor(torch.zeros(2048)),
            # Create the DataLoader.\n",
            from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
            prediction_data = TensorDataset(prediction_inputs, prediction_token_type_ids, prediction_masks),
            prediction_sampler = SequentialSampler(prediction_data),
            prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        
            preds, labels, metrics = trainer.predict(prediction_dataloader)
            print("preds:", preds)
            print("labels:", labels)
            print("metrics:", metrics)
            
            preds, labels, probs = process_results(preds, labels)
            return preds, labels, probs, metrics

        def pred2label(pred):
            if problem_type == 'single_label_classification':
                return label_dict["id2label"][pred]

        # Prediction
        if do_predict and not tune_hyperparams:
            print("*** Predict ***")
            preds, labels, probs, metrics = predict(text)

            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        text = get_text(uploaded_file)
        output = get_prediction(uploaded_file, text)
        area_txt = st.text_area("Text", value=text, height=200)

        '''
        `  `  
        **Did not get what you were expecting?**

        The possible reasons could be:
        - Out of Domain Inference - The text which you have typed is out of domain with training data. The model may not give accurate outcome when domain changes.
        - Annotated Differently - Similar text in the training data could have been annotated differently.
        - Gray Zone - The model predicts with slightely lower confidence on the correct class.
        '''

