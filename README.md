# RF Genesis
### [Project Page](https://rfgen.xingyuchen.me/) | [Paper](https://xingyuchen.me/files/Xingyu.Chen_SenSys23_RFGen.pdf) 

The offical implementation of [  *RF Genesis: Zero-Shot Generalization of mmWave Sensing
through Simulation-Based Data Synthesis and Generative
Diffusion Models*](https://rfgen.xingyuchen.me/).

[Xingyu Chen](https://people.eecs.berkeley.edu/~bmild/),
[Xinyu Zhang](https://people.eecs.berkeley.edu/~bmild/),
UC San Diego.

In SenSys 2023
![teaser](https://rfgen.xingyuchen.me/RFGen/pull.png)
## News
📢 **22/Jan/24** - Initial Release of RF Genesis!

## To-Do List
- [ ] Release the RFLoRA pretrained model.
- [ ] Release the RFLoRA training dataset and procedures.


## Quick Start
This code was tested on `Ubuntu 20.04.5 LTS` and requires:

* Python 3.10
* conda3 or miniconda3
* CUDA capable GPU (one is enough)


Clone the repository
```
git clone git@github.com:Asixa/RF-Genesis.git
```
Download the dependencies [*RFGen_Dependencies.zip*](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/xic063_ucsd_edu/EWUa1yi8V-RKrs2mYwWgom8B5ezkctME6_W_nkSc10iDLg?e=tbVhfX) and drop it to the root folder 
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


## License
This code is distributed under an [MIT LICENSE](LICENSE).
Note that our code depends on other libraries, including CLIP, SMPL and uses datasets that each have their own respective licenses that must also be followed.