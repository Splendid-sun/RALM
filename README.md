# RALM on Knowledge-Intensive Discriminative Task

## 背景/目标

随着大型语言模型（Large Language Models，LLMs）的能力范围不断扩大，使用LLMs来描述自身在自然语言处理（Natural Language Processing，NLP）领域中的应用越来越多。本次研究任务是回答由gpt3.5生成的多项选择题。我们面临的挑战是预测给定提示的最可能的前三个答案。

## 解决方案流程

因为本研究的数据均来自于维基百科的内容，并且由GPT3.5生成选择题，所以我们为每个题目检索最相似的维基百科段落，然后将 维基百科数据+题目 作为训练数据，训练一个基于预训练大语言模型的模型，最后使用这个模型来预测答案。

1. **数据预处理**
    - 读取数据集，并对数据进行预处理，包括填充空值，合并文本等。
    - 使用预训练的SentenceTransformer模型将文本转换为嵌入向量。
    - 使用FAISS库建立索引以便后续的快速搜索。
    - 为每个问题匹配最相似的n个wiki段落。
    
2. **模型训练**
    - 加载预训练模型的tokenizer。
    - 设置模型的配置，这里使用了8位量化。
    - 加载预训练模型。
    - 设置训练参数。
    - 设置LORA的配置。
    - 设置数据的处理方式。
    - 初始化训练器并开始训练。
    - 保存训练好的模型。

## 解决方案重要参数

1. **数据预处理相关**：
    - SentenceTransformer模型：'BAAI/bge-large-en-v1.5'

2. **模型训练相关**：
    - 预训练模型：'llama2-13b' 和 'llama2-7b'
    - 训练和测试的最大序列长度：2048
    - 训练和测试的批大小：2或4
    - 梯度累积步数：1或2
    - 学习率：1e-4 至 5e-4
    - 训练epoch：1.0
    - LORA配置：r=16, a=32, d=0.05

## 总结

本次研究的目标是根据用检索增强语言模型RALM根据给定的提示预测最可能的前三个答案。我们采用了基于预训练大语言模型的解决方案，整个流程包括数据预处理和模型训练两个部分。

在数据预处理阶段，我们使用预训练的SentenceTransformer模型将文本转换为嵌入向量，并使用FAISS库建立索引以便后续的快速搜索。最后，我们为每个问题匹配最相似的n个wiki段落。

在模型训练阶段，我们使用了两个预训练模型，分别是llama2-13b和llama2-7b。微调方式使用的是QLORA。我们使用了8位量化，训练和测试的最大序列长度为2048，训练1个epoch，LORA配置为r=16, a=32, d=0.05。

最终，我们融合了llama2-13b和llama2-7b两个模型的预测结果，得到了0.92x的分数。
