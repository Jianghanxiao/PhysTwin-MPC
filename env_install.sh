conda create -y -n qqtt python=3.10
conda activate qqtt

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

conda install -y numpy==1.26.4
pip install warp-lang
pip install usd-core matplotlib
pip install "pyglet<2"
pip install open3d
pip install trimesh
pip install rtree 
pip install pyrender

pip install termcolor
pip install fvcore
pip install wandb
pip install moviepy imageio
conda install -y opencv
pip install cma

# Install the env for realsense camera
pip install Cython
pip install pyrealsense2
pip install atomics
pip install pynput

# Install the env for grounded-sam-2
pip install --no-build-isolation git+https://github.com/IDEA-Research/Grounded-SAM-2.git
pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git

pip install kornia


# For calibration env 
# pip uninstall opencv-python-headless 
# pip uninstall opencv-python
# pip install opencv-contrib-python