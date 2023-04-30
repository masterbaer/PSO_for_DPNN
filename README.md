# PSO for DPNN
Particle Swarm Optimization for Dataparallel Neural Nets.

This is the repository for my Master's thesis. 

To run the script on the server:

- Clone the repository via: "git clone https://github.com/masterbaer/PSO_for_DPNN.git"
- Navigate to the folder with the .sh files.
- Create a virtual environment with "python -m venv venv".
- Upgrade pip twice "pip install --upgrade pip" "pip install --upgrade pip"
- Install some python packages: "pip install numpy torch torchvision matplotlib"
- Close the virtual environment again "deactivate".

Now you can run the scripts with e.g. "sbatch submit_alexnet_sequetial.sh".
