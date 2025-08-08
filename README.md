## Less is More: Selective Reflection for Compatible and Efficient Knowledge Distillation in Large Language Models


## 1 Requirements

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

pip uninstall numpy -y
pip install numpy==1.24.0  

pip install transformers==4.51.1

pip install vllm==0.5.0
pip install deepspeed==0.16.5
pip install nltk==3.9.1
pip install numerize==0.12
pip install rouge-score==0.1.2
pip install torchtyping==0.1.5
pip install rich==14.0.0
pip install accelerate==1.2.1
pip install datasets==3.2.0
pip install sentencepiece
pip install protobuf==4.23.4
pip install peft==0.14.0
```
or
```bash
bash install.sh
```

Please download the pretrained model checkpoints (GPT2, openllama, qwen2.5-math-instruct, qwen2.5-coder-instruct, T5, and mT5) and put them in the checkpoints/ folder before running the training or evaluation scripts.  

## 2 Data Processing

The raw datasets can download by following scripts. 
```bash
huggingface-cli download MiniLLM/dolly --repo-type dataset /PATH_TO/LMOps/srd/data/dolly/
huggingface-cli download MiniLLM/self-inst --repo-type dataset /PATH_TO/LMOps/srd/data/self-inst/
huggingface-cli download MiniLLM/Vicuna --repo-type dataset /PATH_TO/LMOps/srd/data/vicuna/
huggingface-cli download MiniLLM/sinst --repo-type dataset /PATH_TO/LMOps/srd/data/sinst/
huggingface-cli download MiniLLM/uinst --repo-type dataset /PATH_TO/LMOps/srd/data/uinst/
huggingface-cli download openai/gsm8k --repo-type dataset /PATH_TO/LMOps/srd/data/gsm8k/
huggingface-cli download Muennighoff/mbpp --repo-type dataset /PATH_TO/LMOps/srd/data/mbpp/
huggingface-cli download IWSLT/iwslt2017 --repo-type dataset /PATH_TO/LMOps/srd/data/iwslt2017/
huggingface-cli download EdinburghNLP/xsum --repo-type dataset /PATH_TO/LMOps/srd/data/xsum/
```


Before running the training or evaluation scripts, please preprocess the datasets. The template of processing data are as follows: 

```bash
bash scripts/gpt2/tools/process_data_dolly_srd.sh 
bash scripts/openllama2/tools/process_data_dolly_srd.sh
```
These scripts will generate processed data in the appropriate directories under processed_data/.
## 3 Training

We provide example commands for GPT-2 models. Similar scripts for model families can be found in `scripts/openllama2`. All our experiments are conducted on 4 \* A800-80G, which can be reduced for small models.


### 3.1 Baselines
The final checkpoints are selected by the Rouge-L scores.
#### Fine-tune the teacher models
```bash
bash scripts/gpt2/sft/sft_xlarge.sh /PATH_TO/srd
```
#### SFT Baseline
```bash
bash scripts/gpt2/sft/sft_base.sh /PATH_TO/srd
```

#### SeqKD Baseline
```bash
bash scripts/gpt2/seqkd/seqkd_base.sh /PATH_TO/srd
```

#### GKD Baseline
```bash
bash scripts/gpt2/gkd/gkd_base.sh /PATH_TO/srd
```

#### KD series Baselines
```bash
bash scripts/gpt2/kd/kd_base.sh --type kd
bash scripts/gpt2/kd/kd_base.sh --type rkl
bash scripts/gpt2/kd/kd_base.sh --type jsd
bash scripts/gpt2/kd/kd_base.sh --type tvd
bash scripts/gpt2/kd/kd_base.sh --type sfkl
bash scripts/gpt2/kd/kd_base.sh --type srkl
```


### 3.2 SRD

The final checkpoints are selected by the Rouge-L scores.
```bash
bash scripts/gpt2/srd/train_0.1B_1.5B/train_0.1B_1.5B.sh --type tfkl
bash scripts/gpt2/srd/train_0.1B_1.5B/train_0.1B_1.5B.sh --type trkl
bash scripts/gpt2/srd/train_0.1B_1.5B/train_0.1B_1.5B.sh --type tjsd
bash scripts/gpt2/srd/train_0.1B_1.5B/train_0.1B_1.5B.sh --type ttvdf
bash scripts/gpt2/srd/train_0.1B_1.5B/train_0.1B_1.5B.sh --type tsfkl
bash scripts/gpt2/srd/train_0.1B_1.5B/train_0.1B_1.5B.sh --type tsrkl
```
## 4 Run Evaluation
```bash
bash scripts/gpt2/eval/run_eval.sh /PATH_TO/srd
```

