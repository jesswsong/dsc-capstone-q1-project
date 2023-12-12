# A Theoretical Exploration of Diffusion Model Capacities on 2D Data
## DSC 180 Quarter 1 Project

The goal of this project is to implement the denoising diffusion probablistic model (DDPM) and replicate results using a simple 2D point dataset. This repo contains the source code for creating the 2D point dataset (containing 10000 points) and implementing the forward and reverse processes with visualizations.

## Running Code
To run the code, run `python run.py [target]` to run the corresponding target. The available targets and their description are listed below:

- `visualize`: Run the diffusion model with visualization of the forward diffusion process.

- `random`: Run the diffusion model with random noise added during the reverse diffusion process.

Running the code as above would first create the 2D point dataset and then build a DDPM to perform the forward process and reverse process, producing several file outputs. The first file is the image of the original dataset named "dataset.png." The second file created is "forward_process_visualization.png," demonstrating how points diffuse in the forward process through a series of images in different chosen timesteps (if targets contain visualize). The third file is created is  "backward_visualization.png," illustrating the reverse process at different epochs during the training process. The fourth and last file produced by the code is "final output.png," containing the final output after training and a comparison between original data and data created by diffusion model.
