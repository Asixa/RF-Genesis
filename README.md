# RF Genesis
### [Project Page](https://rfgen.xingyuchen.me/) | [Paper](https://xingyuchen.me/files/Xingyu.Chen_SenSys23_RFGen.pdf) 

The offical implementation of [  *RF Genesis: Zero-Shot Generalization of mmWave Sensing
through Simulation-Based Data Synthesis and Generative
Diffusion Models*](https://rfgen.xingyuchen.me/).

[Xingyu Chen](https://xingyuchen.me/),
[Xinyu Zhang](http://xyzhang.ucsd.edu/index.html),
UC San Diego.

In SenSys 2023
![teaser](https://rfgen.xingyuchen.me/RFGen/pull.png)
## News
📢 **22/Jan/24** - Initial Release of RF Genesis!

📢 **29/March/24** - Added the code for point-cloud processing and visualization.

## To-Do List
- [ ] Release the RFLoRA pretrained model.
- [ ] Release the RFLoRA training dataset and procedures.
- [ ] More documentations.


## Quick Start
This code was tested on `Ubuntu 20.04.5 LTS` and requires:

* Python 3.10
* conda3 or miniconda3
* CUDA capable GPU (one is enough)


Clone the repository
```
git clone https://github.com/Asixa/RF-Genesis.git
cd RF-Genesis
```

Create a conda environment.
```
conda create -n rfgen python=3.10 -y 
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
## Visualization
![ezgif-7-eec8a9c9af](https://github.com/Asixa/RF-Genesis/assets/22312333/a53ef6d7-18b3-4f02-a82a-5bca3aaf08f8)

Rendered SMPL animation and radar point clouds. 

**Credits**: The point cloud algorithm is borrowed  from [mmMesh](https://github.com/HavocFiXer/mmMesh).

## Radar Hardware
The current simulation is based on the model of [**Texas Instruments AWR 1843**](https://www.ti.com/product/AWR1843#all) radar, with 3TX 4RX MIMO setup. 
![TI1843](https://github.com/Asixa/RF-Genesis/assets/22312333/bf68a6df-a3d2-4889-a7eb-509caf52a2fb)

The radar configuration is shown in [TI1843.json](https://github.com/Asixa/RF-Genesis/blob/main/models/TI1843_config.json) and it can be freely adjusted.

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
Note that our code depends on other libraries, including [CLIP](https://github.com/openai/CLIP), [SMPL](https://smpl.is.tue.mpg.de/), [MDM](https://guytevet.github.io/mdm-page/), [mmMesh](https://github.com/HavocFiXer/mmMesh) and uses datasets that each have their own respective licenses that must also be followed.
