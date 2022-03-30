# Body pose API

Our Body pose model is based on OpenPose:
https://github.com/CMU-Perceptual-Computing-Lab/openpose

The openpose model is implemented in C++, we have used python api for our use case to integrate with rest of the project.

## Installation:

Follow the steps as mentioned here:

https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#compiling-and-running-openpose-from-source

In Brief:

1. Get git project:

git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose

cd openpose/

git submodule update --init --recursive --remote

2. Install cmake:

https://cmake.org/

3. Open cmake

cd {OpenPose_folder}
mkdir build/
cd build/
cmake-gui ..

4. Configure as given in the above installation link, and build the project

5. Run the flask app

python app.py

6. now use the POST api in your script

127.0.0.1:5000

Body:
{

	"data" : [#img_numpy_array_to_list]

}

Response:

{

	"keypoints" : #keypoints_in_json,

	"image": [#pose_img_numpy_array_to_list]

}

The keypoints represents json data of the pose detected and image is the numpy array of segmented image which can be stored/viewed or used further in different models as required.

# Test Script

Run the following command to run a test script that opens a camera, and segments each frame.

python script.py



