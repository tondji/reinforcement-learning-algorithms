# Trust Region Policy Optimization (TRPO)
This is a pytorch-version implementation of [Trust Region Policy Optimisation(TRPO)](https://arxiv.org/abs/1502.05477). 

## Requirements
- python 3.5.2
- openai-gym
- mujoco-1.50.1.56
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
python train_network.py

```
### Test the Network
```bash
python demo.py 

```
### Download the Pre-trained Model
Please download them from the [Google Driver](https://drive.google.com/open?id=1ZXqRKwGI7purOm0CJtIVFXOZnmxqvA0p), then put the `saved_models` under the current folder.

## Results
### Training Performance
![Training_Curve](figures/result.png)
### Demo: Walker2d-v1
**Note:** the new-version openai-gym has problem in rendering, so I use the demo of `Walker2d-v1`  
**Tips**: when you watch the demo, you can press **TAB** to switch the camera in the mujoco.  
![Demo](figures/demo.gif)







