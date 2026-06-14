# Actor-Critic based Online Data Mixing For Language Model Pre-Training

This repository is the official implementation of **Actor-Critic based Online Data Mixing For Language
 Model Pre-Training**. 

![AC-ODM - Overview](./imgs/alg.jpeg)

## Requirements

### Recommended Hardware

We recommend using a single server equipped with 8x NVIDIA H800 80GB or 8x NVIDIA H100 80GB GPUs. Additionally, at least **512GB of RAM** and **2TB of storage** are recommended.

### Codebase and Compatibility

Our codebase is based on the Pythia version of [GPT-NEOX](https://github.com/EleutherAI/gpt-neox), which is an older codebase. We have modified several functions to make it compatible with **PyTorch 2.x** and NVIDIA's **Hopper architecture**. If you are already familiar with [GPT-NEOX](https://github.com/EleutherAI/gpt-neox), you should be able to quickly get started with this repository.

### Environment Setup (Recommended: Anaconda)

We suggest using Anaconda to configure your environment.

First, create and activate a new conda environment:

```bash
conda create -n neox python=3.9
conda activate neox
```

Install PyTorch and CUDA dependencies:

```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then, install FlashAttention:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install flash_attn-2.5.6+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```

Install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

Finally, compile and install the custom CUDA kernels:

```bash
python ./megatron/fused_kernels/setup.py install
```


## Preprocessing Data
Our project and code rely on [The Pile](https://github.com/EleutherAI/the-pile) dataset, which you can download from the official GitHub repository.

Let's start preprocessing the data using the 30 training set `.jsonl` files (`00.jsonl` to `29.jsonl`) and one test set `.jsonl` file you have downloaded.

### Step 1: Divide Domains of the Training Set

Ensure all `00.jsonl` to `29.jsonl` files are located in the `INPUT_DIR` directory. Then run the following command:

```bash
# Divide domains of the training set
python divide_pile_single.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```

### Step 2: Divide Domains of the Test Set

To process the test set, run the following command:

```bash
# Divide domains of the test set
python divide_pile_test.py --input_file TEST_JSONL_FILE --output_dir OUTPUT_DIR
```

### Step 3: Preprocess All Domains

Finally, preprocess all the domain-separated files using:

```bash
# Preprocess all domain-separated files
./preprocess_pile.sh OUTPUT_DIR PROCESSED_DIR
```

By default, `preprocess_pile.sh` uses `tokenizer_config/20B_tokenizer.json`. To use a different tokenizer file, set `TOKENIZER_FILE` before running the script.

## Training

Before starting training, configure Weights & Biases without writing secrets into the source tree. Either run `wandb login` or export `WANDB_API_KEY` in your shell:

```bash
export WANDB_API_KEY="YOUR_API_KEY_HERE"
```

### Experiment Configuration

Our experimental settings are primarily defined in the following files:

- The data configuration file: `./acodm_config/data/pile.yml`
- Model hyperparameter files: located in `./acodm_config/models/`
- DDPG configuration files: such as `./acodm_config/ddpg_X.yml`, where `X` can be `1B`, `410m`, etc.

Before training, make sure to update the `PROCESSED_DIR` field in `./acodm_config/data/pile.yml` to match the `PROCESSED_DIR` path obtained in Step 3 of the Preprocessing Data section.

### Training AC-ODM and AC-ODM-410M

As these are the main experiments presented in our paper, you need to train both AC-ODM and AC-ODM-410M models separately.

#### Train AC-ODM (1B)

Run the following command:

```bash
./acodm_scripts/train_1B_acodm.sh
```

#### Train AC-ODM-410M (Proxy Model & 1B)

First, train the proxy model by running:

```bash
./acodm_scripts/train_410M_acodm.sh
```

Once the proxy model has finished training, proceed to train the 1B Pythia model with:

```bash
./acodm_scripts/train_410M_1B_acodm.sh
```


## Evaluation

Since the main contribution of our work lies in the training process, this section provides guidance for evaluating downstream tasks after training is completed.

### MMLU Evaluation

We provide four scripts to reproduce our experimental results on the MMLU benchmark:

- **Evaluate 0-shot performance of AC-ODM on MMLU:**

```bash
./acodm_scripts/eval_1B_acodm_0shot.sh
```

- **Evaluate 5-shot performance of AC-ODM on MMLU:**

```bash
./acodm_scripts/eval_1B_acodm_5shot.sh
```

- **Evaluate 0-shot performance of AC-ODM-410M on MMLU:**

```bash
./acodm_scripts/eval_410M_1B_acodm_0shot.sh
```

- **Evaluate 5-shot performance of AC-ODM-410M on MMLU:**

```bash
./acodm_scripts/eval_410M_1B_acodm_5shot.sh
```

### HumanEval Evaluation

You need to add the `humaneval` task from [HumanEval](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks) to your `lm_eval` site-packages.

- **Evaluate AC-ODM on HumanEval:**

```bash
./acodm_scripts/eval_1B_acodm_he.sh
```

- **Evaluate AC-ODM-410M on HumanEval:**

```bash
./acodm_scripts/eval_410M_1B_acodm_he.sh
```



## Results

After training is complete, you can view our main experimental results under the `acodm_1B` project on your Weights & Biases (wandb) account.

**The main results are shown below:**

![Main Results - Validation Perplexity During Training](./imgs/main_result_1.jpeg)

**We also provide a comparison of the final perplexity (PPL) across different domains:**

![Perplexity by Domain](./imgs/main_result_2.jpeg)

### Evaluation of Downstream Tasks on [MMLU](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu) and [HumanEval](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/humaneval)

| Algorithm     | MMLU - 0-shot Accuracy | MMLU - 5-shot Accuracy | HumanEval (pass@1) |
|---------------|------------------------|-------------------------|---------------------|
| **AC-ODM**      | 0.25146                | 0.29868                 | 0.60256             |
| **AC-ODM-410M** | 0.29980                | 0.35215                 | 0.72644             |
