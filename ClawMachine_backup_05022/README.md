# ClawMachine

## Install

Install Package

```Shell
conda create -n clawmachine python=3.10 -y
conda activate clawmachine
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

Install additional packages for training cases

```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

Download LaVIT-7B-v2 [GitHub - jy0205/LaVIT: LaVIT: Empower the Large Language Model to Understand and Generate Visual Content](https://github.com/jy0205/LaVIT) for the basic language model and its visual tokenizer.

Then prepare the datasets following LLaVA-1.5's instructions.

## Train

ClawMachine is trained on 8 A800 GPUs with 80GB memory. 

```
sh scripts/clawmachine/stage_one.sh

sh scripts/clawmachine/stage_two.sh

sh scripts/clawmachine/stage_three.sh
```

## Evaluation

For grounding evaluation:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash evaluatiin/GND_batch.sh
```

Then use evaluation/corr_check_v3.ipynb to process the results, and you can get the final scores using GND_test.ipynb.

For referring evaluation:

```
CUDA_VISIBLE_DEVICES=0 python evaluation/model_REF_caption_mirage.py
```

Then use evaluation/REF_test.ipynb  get the final scores.

