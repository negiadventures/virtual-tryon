import argparse
import sys
from sys import platform

import numpy as np
from flask import Flask, jsonify, request
import os, glob
import json

app = Flask(__name__)

def cleanup():
    dir = 'json'
    for file in os.scandir(dir):
        os.remove(file.path)

@app.route("/pose", methods=["POST"])
def get_pose():
    cleanup()
    if request.method == 'POST':
        posted_data = request.json
        data = posted_data['data']
        nparr = np.array(data, np.uint8)
        datum = op.Datum()
        datum.cvInputData = nparr
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        data_encode = np.array(datum.cvOutputData)
        str_encode = data_encode.tolist()
        path='json'
        for file in os.scandir(path):
            f = open(file.path)
            break
        json_data = json.load(f)
        f.close()
        return jsonify({"keypoints": json_data, "image": str_encode})


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/build/python/openpose/Release');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/build/x64/Release;' + dir_path + '/build/bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="examples/media/",
                        help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    parser.add_argument("--disable_blending",default = True)
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "models/"
    params["net_resolution"] = "-1x368"
    params["output_resolution"] = "768x1024"
    params["write_json"] = "json"
    params["net_resolution_dynamic"] = 1
    params["num_gpu"] = 1
    params["num_gpu_start"] = 0

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    app.run(debug=True)
