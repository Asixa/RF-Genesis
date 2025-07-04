import argparse
from termcolor import colored
import time
from genesis.raytracing import pathtracer
from genesis.raytracing import signal_generator

from genesis.environment_diffusion import environemnt_diff
from genesis.object_diffusion import object_diff
from genesis.visualization import visualize


import torch
import numpy as np
import os
torch.set_default_device('cuda')

def get_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='List the content of a folder')
    parser.add_argument('-o', '--obj-prompt', type=str, help='Specify the object prompt')
    parser.add_argument('-e','--env-prompt', type=str, help='Specify the environment prompt')
    parser.add_argument('-n', '--name', type=str, help='Specify the name (optional)')
    parser.add_argument('--no-visualize',  dest='skip_visualize', default= False,
                        help='Disable visualization step (default: enabled)')
    parser.add_argument('--no-environment',  dest='skip_environment', default= False,
                        help='Disable environment PIR generation (default: enabled)')


    args = parser.parse_args()

    return args.obj_prompt, args.env_prompt, args.name, args.skip_visualize, args.skip_environment


def main():
    obj_prompt, env_prompt, name, skip_visualize, skip_environment = get_args()
    # obj_prompt, env_prompt, name = "a person walking back and forth", "", "test"

    

    if name is None:
        name = f"output_{int(time.time())}"

    output_dir = os.path.join("output", name)
    os.makedirs(output_dir, exist_ok=True)


    if not os.path.exists(os.path.join(output_dir, 'obj_diff.npz')):
        print(colored('[RFGen] Step 1/4: Generating the human body motion: ', 'green'))
        object_diff.generate(obj_prompt, output_dir)
    else:
        print(colored('[RFGen] Step 1/4: Already done, existing body motion file, skiping this step.', 'green'))

    
    os.chdir("genesis/")
    print(colored('[RFGen] Step 2/4: Rendering the human body PIRs: ', 'green'))
    body_pir, body_aux = pathtracer.trace(os.path.join("../",output_dir, 'obj_diff.npz'))
    os.chdir("..")
    
    
    # print(colored('[RFGen] Step 3/4: Generating the environmental PIRs: ', 'green'))
    # print(colored('[RFGen] Step 3/4: [Jan 2024] RFLoRA and Environment Diffusion is Temporarily Disabled.', 'red'))ç
    # print(colored('                  We will update tuned RFLoRA soon.', 'red'))
    # print(colored('                  RFGen will continue without RFLoRA.', 'green'))


    if not skip_environment:
        print(colored('[RFGen] Step 3/4: Generating the environmental PIRs: ', 'green'))
        envir_diff = environemnt_diff.EnvironmentDiffusion(lora_path="Asixa/RFLoRA")
        env_pir = envir_diff.generate(env_prompt)
    else:
        print(colored('[RFGen] Step 3/4: Skipping environment generation as requested.', 'yellow'))
        env_pir = None


    print(colored('[RFGen] Step 4/4: Generating the radar signal.', 'green'))
    radar_frames = signal_generator.generate_signal_frames(body_pir, body_aux, env_pir, radar_config="models/TI1843_config.json")

    print(colored('[RFGen] Saving the radar bin file. Shape {}'.format(radar_frames.shape), 'green'))
    np.save(os.path.join(output_dir, 'radar_frames.npy'), radar_frames)

    if not skip_visualize:
        print(colored('[RFGen] Rendering the visualization.', 'green'))
        torch.set_default_device('cpu')  # To avoid OOM
        visualize.save_video(
            "models/TI1843_config.json", 
            os.path.join(output_dir, 'radar_frames.npy'), 
            os.path.join(output_dir, 'obj_diff.npz'), 
            os.path.join(output_dir, 'output.mp4'))
    else:
        print(colored('[RFGen] Skipping visualization step.', 'yellow'))


    print(colored('----------------------------------------', 'green')) 
    print(colored('[RFGen] Hooray! you are all set! ', 'green')) 
    print(colored('----------------------------------------', 'green')) 
    print(colored('        Please ignore the segmentation faults if there are any.', 'green'))

    exit(0)
if __name__ == '__main__':
    main()


    
