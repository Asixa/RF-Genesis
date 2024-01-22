# RF Genesis
### [Project Page](https://rfgen.xingyuchen.me/) | [Paper](https://xingyuchen.me/files/Xingyu.Chen_SenSys23_RFGen.pdf) 

The offical implementation of [  *RF Genesis: Zero-Shot Generalization of mmWave Sensing
through Simulation-Based Data Synthesis and Generative
Diffusion Models*](https://rfgen.xingyuchen.me/).

[Xingyu Chen](https://people.eecs.berkeley.edu/~bmild/),
[Xinyu Zhang](https://people.eecs.berkeley.edu/~bmild/),
UC San Diego.

In SenSys 2023

## News
📢 **22/Jan/24** - Inital Release of RF Genesis!

## To-Do List
- [ ] Release the RFLoRA pretrained model.
- [ ] Release the RFLoRA training dataset and procedures.


## Quick Start
```
git clone git@github.com:Asixa/RF-Genesis.git
```
Download the dependencies [*RFGen_Dependencies.zip*](https://rfgen.xingyuchen.me/) and drop it to the root folder 
**without unzipping it**.

Create a conda environment.
```
conda conda create -n rfgen python=3.10 -y 
conda activate rfgen
```
Install python packages
```
pip install -r requirements.txt
sh setup.sh
```
Run a simple example.
```
python run.py -o "a person walking back and forth" -e "a living room" -n "hello_rfgen"
```

## Citation
```
@inproceedings{chen2023rfgenesis,
      author = {Chen, Xingyu and Zhang, Xinyu},
      title = {RF Genesis: Zero-Shot Generalization of mmWave Sensing through Simulation-Based Data Synthesis and Generative Diffusion Models},
      booktitle = {ACM Conference on Embedded Networked Sensor Systems (SenSys ’23)},
      year = {2023},
      pages = {1-14},
      address = {Istanbul, Turkiye},
      publisher = {ACM, New York, NY, USA},
      url = {https://doi.org/10.1145/3625687.3625798},
      doi = {10.1145/3625687.3625798}
  }
```