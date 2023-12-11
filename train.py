import sys
import gc
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time
import json

import torch
from torch import nn
from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
import transformers
from transformers import BitsAndBytesConfig
from datasets import Dataset, DatasetDict, load_dataset
from transformers import TrainingArguments
from peft import LoraConfig, AutoPeftModelForCausalLM
print(transformers.__version__)

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from langchain.prompts import PromptTemplate
import wandb

# 导入一些自定义的工具函数
from utils import get_now_time_fullstring, format_instruction, plot_sequence_lengths, seed_everything, init_logger

# 设置随机种子，确保实验可复现
seed_everything(42)
# 登录wandb，用于实验管理和记录
wandb.login()
# 设置wandb项目名称
os.environ["WANDB_PROJECT"] = "Kaggle-llm-science-exam"

# 使用argparse进行命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--suff", type=str, default="099999")
parser.add_argument("--train", type=bool, default=True)
parser.add_argument("--no-train", action='store_false', dest='train')
parser.add_argument("--model_arch", type=str, default="llama2-13b")
parser.add_argument("--select_maxlen", type=int, default=40960)
parser.add_argument("--train_maxlen", type=int, default=2048)
parser.add_argument("--test_maxlen", type=int, default=2048)
parser.add_argument("--train_bs", type=int, default=2)
parser.add_argument("--test_bs", type=int, default=2)
parser.add_argument("--grad_accum", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--steps", type=int, default=440)
parser.add_argument("--ep", type=float, default=1.0)
parser.add_argument("--lora_r", type=int, default=16) 
parser.add_argument("--lora_a", type=int, default=32)
parser.add_argument("--lora_d", type=float, default=0.05)
parser.add_argument("--comment", type=str, default="")

args = parser.parse_args()

# 根据参数生成一个后缀，用于区分不同的实验设置
SUFF = f"{args.suff}_lr{args.lr}_{args.ep}ep_lora(r{args.lora_r},a{args.lora_a},d{args.lora_d},default)"
print(f"SUFF: {SUFF}")
# 定义输入输出路径
input_dir  = "/home/br/workspace/LLM/input"
output_dir = f"/home/br/workspace/LLM/output/{args.model_arch}_{SUFF}"
model_name = f"{input_dir}/{args.model_arch}"
# 数据集路径
df_path = f"{input_dir}/chris-60k/60k.parquet"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
# 初始化日志
LOGGER = init_logger(f'{output_dir}/{args.suff}_train.log')
LOGGER.info(f"############## {get_now_time_fullstring()} ##############")
start_time = time.time()

# 判断是否进行训练
if args.train:
    # 读取数据集
    df = pd.read_parquet(df_path)
    LOGGER.info(args.comment)

    # 对数据集进行预处理
    df['instruction'] = df.apply(lambda row: format_instruction(row, "train"), axis=1)

    LOGGER.info("================================================== example ==================================================")
    LOGGER.info(df.iloc[0].instruction)
    LOGGER.info("=============================================================================================================")

    # 将数据集转为huggingface的Dataset格式
    dataset = Dataset.from_pandas(df)
    data = DatasetDict({"train": dataset})

    # 筛选长度小于特定值的数据
    keep_indices_train = plot_sequence_lengths(
        data, 
        split='train', 
        max_length=args.select_maxlen, 
        is_plot=False
    )
    data['train'] = data['train'].select(keep_indices_train)
    LOGGER.info(data)
        
    # 加载预训练模型的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
        truncation_side="left",

    )
    tokenizer.pad_token = tokenizer.eos_token

    # 设置模型的配置，这里使用了8位量化
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # 加载预训练模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir, 
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.test_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=20,
        logging_strategy="steps",
        num_train_epochs=args.ep, 
        lr_scheduler_type='cosine', 
        warmup_ratio = 0.1, 
        warmup_steps = 20, 
        optim="adamw_hf", 
        fp16=True,
        logging_dir = f"{output_dir}", 
        dataloader_num_workers = 4, 
        run_name=f"{args.model_arch}_{SUFF}",
    )

    # 设置LOra的配置
    qlora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_a,
        lora_dropout=args.lora_d,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 设置数据的处理方式
    response_template_ids = tokenizer.encode("\nAnswer:", add_special_tokens=False)[2:]  
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    # 初始化训练器
    trainer = SFTTrainer(
        base_model,
        
        train_dataset=data["train"],
        dataset_text_field="instruction",
        max_seq_length=args.train_maxlen,
        tokenizer=tokenizer,
        data_collator=data_collator,

        args=training_args,
        peft_config=qlora_config,
    )

    # 开始训练
    LOGGER.info("start training...")
    trainer.train()
    # 保存模型
    trainer.save_model(output_dir)
    LOGGER.info("finish training...")
