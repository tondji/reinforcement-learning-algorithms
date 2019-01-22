# Synchronous Advantage Actor-Critic (A2C)
This is the pytorch version of the [A2C](https://arxiv.org/pdf/1602.01783.pdf) - **which is basiclly the same as the openai baselines** 
## Requirements
- python-3.5.2
- openai baselines
- pytorch-0.4.0
## Installation
Install OpenAI Baselines (**the openai-baselines update so quickly, please use the older version as blow, will solve in the future.**)
```bash
# clone the openai baselines
git clone https://github.com/openai/baselines.git
cd baselines
git checkout 366f486
pip install -e .

```
## Instruction to run the code
### Train the Network:
```bash
python train_network.py --env-name 'BreakoutNoFrameskip-v4' --cuda (if you have a GPU, you can add this flag)

```
### Test the Network:
```bash
python demo.py --env-name 'BreakoutNoFrameskip-v4'

```
### Download the Pre-trained Model
Please download them from the [Google Driver](https://drive.google.com/open?id=1ZXqRKwGI7purOm0CJtIVFXOZnmxqvA0p), then put the `saved_models` under the current folder.

## Results
It can be found some learning curves was not perfect, because it's the problem of the implementation of `RMSProp` in pytorch. Please check this [blog](https://blog.millionintegrals.com/vel-pytorch-meets-baselines/)
### Training Performance
![Scheme](figures/result.png)
## Demo: BreakoutNoFrameskip-v4
![Scheme](figures/demo.gif)


