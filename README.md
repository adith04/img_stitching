# Computer Vision Image Stitching Full Simulation Setup

Hello! The following is a full setup for an image stitching program for a drone in Gazebo.
If you are not currently operating on Linux or have not setup Linux on a virtual machine, download the following link to set this up: [Virtual Box Ubuntu Setup](https://github.com/adith04/stitching/raw/master/Installing_virtual_box.pdf)


## Setting Up ArduPilot:

Open a terminal and make sure you are in the home directory. Then run the commands as follows:
```bash
cd ~
```
Install git
```bash
sudo apt install git
```
Clone the ardupilot repository
```bash
git clone https://github.com/ArduPilot/ardupilot.git
```
Install prereqs
```bash
cd ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile
```
Checkout copter
```bash
git config --global url.https://.insteadOf git://
git checkout Copter-4.0.4
git submodule update --init --recursive
```
Check if simulation works
```bash
cd ArduCopter
sim_vehicle.py -w
```
To exit the program, enter Ctrl C in the command line

## Setting Up ROS Environment:

Open a terminal and enter commands as follows:
```bash
cd ~
```
Ensure ros packages recognized
```bash
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
```
Setup keys
```bash
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
```
Keep packages up to date
```bash
sudo apt update
```
Install gazebo11
```bash
sudo apt-get install gazebo11 libgazebo11-dev
```
Clone ardupilot_gazebo repository
```bash
git clone https://github.com/khancyr/ardupilot_gazebo.git
```
Build package
```bash
cd ardupilot_gazebo
mkdir build
cd build
cmake ..
make -j4
sudo make install
```
```bash
echo 'source /usr/share/gazebo/setup.sh' >> ~/.bashrc
```
```bash
. ~/.bashrc
```
```bash
cd
```
```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```
```bash
sudo apt install curl
```
```bash
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
```
```bash
sudo apt update
```
```bash
sudo apt install ros-noetic-desktop-full
source /opt/ros/noetic/setup.bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
Install python dependencies
```bash
sudo apt-get install python3-wstool python3-rosinstall-generator python3-catkin-lint python3-pip python3-catkin-tools
```
```bash
pip3 install osrf-pycommon
```
Create workspace
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
wstool init ~/catkin_ws/src
```
Setup mavros and mavlink
```bash
rosinstall_generator --upstream mavros | tee /tmp/mavros.rosinstall
rosinstall_generator mavlink | tee -a /tmp/mavros.rosinstall
wstool merge -t src /tmp/mavros.rosinstall
wstool update -t src
sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro `echo $ROS_DISTRO` -y
```
Build workspace
```bash
catkin build
```
Add to .bashrc and source
```bash
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
Install geographiclib
```bash
sudo ~/catkin_ws/src/mavros/mavros/scripts/install_geographiclib_datasets.sh
```
Clone simulation repository and build workspace
```bash
git clone -b main https://github.com/adith04/simulation.git
catkin build
```
Source .bashrc
```bash
source ~/.bashrc
```
Add to end of .bashrc and source
```bash
echo "export GAZEBO_MODEL_PATH=~/ardupilot_gazebo/models" >> ~/.bashrc
echo "export PATH=$PATH:$HOME/.local/bin" >> ~/.bashrc
echo "export GAZEBO_MODEL_PATH=/home/$USER/ardupilot_gazebo/models:/home/$USER/catkin_ws/src/simulation/models" >> ~/.bashrc
echo "export GAZEBO_MODEL_PATH=~/gazebo_ws/gazebo_models:${GAZEBO_MODEL_PATH}" >> ~/.bashrc
echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```
Copy startsitl.sh from the simulation directory
![Alt text](https://github.com/adith04/stitching/raw/master/first.png)

Paste it to your home directory 
![Alt text](https://github.com/adith04/stitching/raw/master/second.png)


## Install QGroundControl (Optional):
```bash
sudo usermod -a -G dialout $USER
sudo apt-get remove modemmanager
```
Installation
```bash
wget https://s3-us-west-2.amazonaws.com/qgroundcontrol/latest/QGroundControl.AppImage
```
Make executable
```bash
chmod +x ./QGroundControl.AppImage 
```
To run enter 
```bash
./QGroundControl.AppImage
```

## Clone stitching repository:
```bash
cd catkin_ws/src
```
Clone repository
```bash
git clone -b main https://github.com/adith04/stitching.git
```
Build repository
```bash
cd ..
catkin build
```

## Running panorama.py and object_detections.py
```bash
pip install ultralytics
```

## Contact:
For any questions/concerns about this project please contact me at the following email address: radithya@umich.edu
