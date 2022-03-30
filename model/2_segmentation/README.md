# Body Segmentation API

Our Body segmentation model is based on Self-Correction-Human-Parsing:
https://github.com/PeikeLi/Self-Correction-Human-Parsing

There are three pre-trained models which may be used depending upon the number of classes each have: LIP has 20 classes, ATR has 18 and Pascal-Person-Part has 7 classes.

We are using LIP model. The weight file can be downloaded from: https://drive.google.com/file/d/1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH/view

NOTE: Rename the file to final.pth and place it under checkpoints/ directory

## Steps to run segmentation API on local

1. enable Hyper-V on your machine and restart

2. Install docker on your machine

3. Build custom docker image from "nvidia/cuda:11.6.0-devel-ubuntu20.04" base image

docker build -t segmentation:latest .

4. Run a docker container which runs the flask app for segmentation API

docker run -it --rm -p 127.0.0.1:8180:8888 --gpus all segmentation

5. now use the POST api

127.0.0.1:8180

Body:
{

 "data" : [#img_numpy_array_to_list]

}

Response:

[#segmented_img_numpy_array_to_list]

This is a numpy array of segmented image which can be stored/viewed or used further in different models as required.

# Test Script

Run the following command to run a test script that opens a camera, and segments each frame.

python script.py

### Extras:
#### Command to run an example script from the container
python simple_extractor.py --dataset 'lip' --model-restore 'checkpoints/final.pth' --input-dir 'inputs' --output-dir 'outputs'



