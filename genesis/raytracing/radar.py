import torch
import matplotlib.pyplot as plt
import numpy as np
import json

def FSPL(distance, frequency=77e9):
    wavelength = 3e8 / frequency
    return (wavelength / (4 * 3.14159265358979323846 * distance)) ** 2

def dechirp(x,xref):
    return xref * torch.conj(x)

class Radar:

    def __init__(self, config="config.json"):
        self.c0 = 299792458

        if isinstance(config, str):
            with open(config, 'r') as f:
                config = json.load(f)
        elif not isinstance(config, dict):
            raise ValueError("config must be a filename or dict")


        self.num_tx = config["num_tx"]
        self.num_rx = config["num_rx"]
        self.fc = config["fc"]
        self.slope = config["slope"]
        self.adc_samples = config["adc_samples"]
        self.adc_start_time = config["adc_start_time"]
        self.sample_rate = config["sample_rate"]
        self.idle_time = config["idle_time"]
        self.ramp_end_time = config["ramp_end_time"]
        self.chirp_per_frame = config["chirp_per_frame"]
        self.frame_per_second = config["frame_per_second"]
        self.num_doppler_bins = config["num_doppler_bins"]
        self.num_range_bins = config["num_range_bins"]
        self.num_angle_bins = config["num_angle_bins"]
        self.power = config["power"]

        antenna_spacing = self.c0/self.fc/2
        self.tx_loc = np.array(config["tx_loc"],dtype = np.float32)*antenna_spacing
        self.rx_loc = np.array(config["rx_loc"],dtype = np.float32)*antenna_spacing

        self.tx_pos = torch.tensor(self.tx_loc)
        self.rx_pos = torch.tensor(self.rx_loc)


        # Here is an example of how to generate the tx and rx locations, you can usually find them on the user manual of the radar.
        # self.tx_loc = np.array([[0,0,0],[4*spacing,0,0],[2*spacing,spacing,0]])
        # self.rx_loc = np.array([[-6*spacing,0,0],[-5*spacing,0,0],[-4*spacing,0,0],[-3*spacing,0,0]])

        self.range_resolution = (3e8 * self.sample_rate * 1e3) / (2 * self.slope * 1e12 * self.adc_samples)
        self.max_range = (300 * self.sample_rate) / (2 * self.slope * 1e3)
        self.doppler_resolution = 3e8 / (2 * self.fc * 1e9 * (self.idle_time + self.ramp_end_time) * 1e-6 * self.num_doppler_bins * self.num_tx)
        self.max_doppler = 3e8 / (4 * self.fc * 1e9 * (self.idle_time + self.ramp_end_time) * 1e-6 * self.num_tx)
       
        self._lambda = self.c0/self.fc


    
    def waveform(self,t,phi=0): 
        
        fc = (self.fc * t + 0.5 * (self.slope * 1e6* 1e6) *  t * t )
        yI = torch.cos( 2 * torch.pi * fc +phi)
        yQ = torch.sin( 2 * torch.pi  * fc +phi)
        y =  yI+yQ*1j
        return y

    def chirp(self,distance,intensity):
        t_sample = torch.arange(0,self.adc_samples,dtype=torch.float64)/(self.sample_rate*1e3) +self.adc_start_time*1e-6
        # loss = RF_path_loss_torch(tof).view(-1,1)
        loss = FSPL(distance).view(-1,1)
        intensity = intensity.view(-1,1)
        tof = distance / self.c0
        
        tx = self.waveform(t_sample)
        rx = self.waveform(t_sample-tof)*  loss * intensity

        rx_combined = torch.sum(rx,axis=0)
        sig = dechirp(rx_combined,tx)
        return sig
    
    def frame(self,interpolator,t0, n_ray = 1):
        frame = torch.zeros((self.chirp_per_frame, self.adc_samples),dtype = torch.complex128)
        for chirp_id in range(self.chirp_per_frame):
            time_in_frame = chirp_id / self.chirp_per_frame / self.frame_per_second
            intensity,loc =  interpolator(t0+time_in_frame)
            tx_pos = self.tx_pos[0].unsqueeze(0)  # Convert shape from (3,) to (1, 3)
            rx_pos = self.rx_pos[0].unsqueeze(0)  # Convert shape from (3,) to (1, 3)
            tof = torch.cdist(loc,tx_pos)+ torch.cdist(loc,rx_pos) 

            chirp =  self.chirp(tof,intensity)
            frame[chirp_id,:] =chirp
        return frame
    

    def frameMIMO(self,interpolator,t0=0, n_ray = 1):
        frame = torch.zeros((self.num_tx, self.num_rx, self.chirp_per_frame, self.adc_samples),dtype = torch.complex128)
        for chirp_id in range(self.chirp_per_frame): 
            
            # it is inefficient to calculate the location for every sample point, so we do that for every chirp
            # this is a tradeoff between accuracy and computation
            
            time_in_frame = chirp_id / self.chirp_per_frame  / self.frame_per_second                 
            intensity,loc = interpolator(t0+time_in_frame) # loc is a set of reflection points (N,3)
            for tx_id in range(self.num_tx):
                for rx_id in range(self.num_rx):
                    tx_pos = self.tx_pos[tx_id].unsqueeze(0)  # Convert shape from (3,) to (1, 3)
                    rx_pos = self.rx_pos[rx_id].unsqueeze(0)  # Convert shape from (3,) to (1, 3)
                    tof = torch.cdist(loc,tx_pos)+ torch.cdist(loc,rx_pos)    # Tx - Surface - Rx
                    frame[tx_id,rx_id,chirp_id,:] = self.chirp(tof,intensity)
        return frame
