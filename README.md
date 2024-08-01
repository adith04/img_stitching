# Computer Vision Image Stitching Full Simulation Setup

Hello! The following is a full setup for an image stitching program for a drone in Gazebo.
If you are not currently operating on Linux or have not setup Linux on a virtual machine, visit the following link to set this up:

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
Checkout Copter
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
Copy startsitl.sh to home directory

sudo usermod -a -G dialout $UdSER

sudo apt-get remove modemmanager

wget https://s3-us-west-2.amazonaws.com/qgroundcontrol/latest/QGroundControl.AppImage

chmod +x ./QGroundControl.AppImage 

./QGroundControl.AppImage

cd catkin_ws/src

Git clone -b main https://github.com/adith04/stitching.git

Cd ..

Catkin build



### training/crawl.py

This script is used to crawl the URLs of football related articles found by the team for model training purposes. Al of these URLs are contained in training/training_articles_urls.txt. This file is a list of links labeled with their corresponding topic, which then gets crawled and processed into the format needed to train our model. 

Navigate to the training/ folder and run this script to crawl the training URLs:
```bash
cd training
python3 crawl.py training_article_urls.txt
```

The raw training data for this run should be shown in training/training_data_test/ 

### clean_training_data.py

This script simply takes the raw training data contained in training/training_data_raw/ and outputs the cleaned version into training/training_data_processed/

```bash
python3 clean_training_data.py
```

### training/naivebayes.py

This script runs the Naive Bayes algorithm on the processed training data. We wanted to use this as an evaluation metric, to compare our custom trained model with a more naive approach to the same problem. 

Navigate to the naivebayes.py script and run it on the cleaned training data folder as an input as shown below:
```bash
python3 naivebayes.py training_data_processed/
```
In the terminal you can see the number of training files analyzed as well as the accuracy of the system using Naive Bayes.

### classify_sentences.py

Using the training data contained in training/training_data_processed/ the team was able to train a model.

Google Colab Link: https://colab.research.google.com/drive/1q6gOAOhHm3mVFlxmRt_nCp3P5aD8Te-3?usp=sharing

Above is the Google Colab that contains scripts used to train the language model that classify_sentences.py uses. The Colab can be run by
running every cell in sequence and uploading the appropriate training files (articles) when prompted. The downloaded model is placed in the model_directory/ folder for local inference by classify_sentences.py.

Navigate to the root of this repository and run classify_sentences.py on a sample test file created by the team:
```bash
cd ..
python3 classify_sentences.py example_sentences.txt
```

As you can see, each of the sentences in example_sentences.txt is classified by the custom trained language model. Take a look at the output file:
```bash
cat classified_sentences.txt
```

### transcribe_google.py

This script takes in a .mp3 file name as input and produces a audio transcription files containing sentences. The Google Cloud credential file 'credentials.json' needs to be at the root of the repository for the audio transcription files to be properly generated. This script takes in 1 command line argument, the .mp3 file to be transcribed. The actual .mp3 file is located in a Google Cloud bucket associated with the 'credentials.json'. All the .mp3 files that this script can transcribe are contained in 'transcribe_podcast_options.txt'.

Verify that the credentials.json is contained in the repository.

```bash
cat credentials.json
```

The audio transcription usually takes ~20 minutes for a podcast episode. Thus, the team created a small snippet of an episode called 'podcast_snippet.mp3' that is around 4 minutes long so that the transcription process can be faster for you. It should take ~5 minutes.

```bash
python3 transcribe_google.py podcast_snippet.mp3
```

Now take a look at the podcast_snippet/ folder that was generated. Inside are all the sentence transcription files and sentence timestamp information generated by Google Cloud.

```bash
cd podcast_snippet
ls 
cat sentences_timestamps.txt 
cat sentences_transcribed_raw.txt
```

### merge_classification.py

This script takes in the audio transcription files generated by Google Cloud for a specific podcast episode and splits up the podcast into segmented categories. Simply run the script with the podcast episode name as the argument. This name must match that of a podcast contained in 'transcribe_podcast_options.txt'. Let's do the podcast episode snippet 'podcast_snippet.mp3' and 'in_the_trenches_episode_428.mp3' as examples. 

```bash
cd ..
python3 merge_classification.py podcast_snippet.mp3
python3 merge_classification.py in_the_trenches_episode_428.mp3
```

Take a look at the output. It showcases how a podcast changes topic as time goes on. A timeline plot generated by matplotlib should also be shown on your machine that visualize these topic changes. 

```bash
cat in_the_trenches_episode_428_segments.txt
```

### evaluate.py

This script was used for efficient calculation of the IoU metrics for our manually reviewed segment evaluations contained in the evaluation/ folder. 

Navigate to the evaluation/ folder and run it on a labeled podcast segments file manually verified by a team member. 
```bash
cd evaluation
python3 evaluate.py in_the_trenches_episode_428_segments.txt
```

The results of this evaluation should be shown in the terminal. These metrics are included in a table in our project report. 

## Contact
For any questions/concerns about this project please contact the team at the following email addresses:
