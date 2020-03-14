## PoseNet Python

This repository contains a pure Python implementation (multi-pose only) of the Google TensorFlow.js Posenet model.

Further optimization is possible
* The base MobileNet models have a throughput of 200-300 fps on a GTX 1080 Ti (or better)
* The multi-pose post processing code brings this rate down significantly. With a fast CPU and a GTX 1080+:
  * A literal translation of the JS post processing code dropped performance to approx 30fps
  * My 'fast' post processing results in 90-110fps
* A Cython or pure C++ port would be even better...  


### Usage

There are multiple demo apps in the root that utilize the PoseNet model. They are very basic and could definitely be improved.

The first time these apps are run (or the library is used) model weights will be downloaded from the TensorFlow.js version and converted on the fly.

For all demos, the model can be specified with the '--model` argument by using its ordinal id (0-3) or integer depth multiplier (50, 75, 100, 101). The default is the 101 model.

#### image_demo.py 

Image demo runs inference on an input folder of images and outputs those images with the keypoints and skeleton overlayed.

`python image_demo.py --model 101 --image_dir ./images --output_dir ./output`

A folder of suitable test images can be downloaded by first running the `get_test_images.py` script.

#### benchmark.py

A minimal performance benchmark based on image_demo. Images in `--image_dir` are pre-loaded and inference is run `--num_images` times with no drawing and no text output.

#### webcam_demo.py

Shows a stick figure pose in real-time using a web-cam. To overlay the stick figure on background turn on the "bgimage" parameter

#### webcam_demo_react.py

Fun game that uses hand detection and calculates reaction time taken to touch targets with hand.

#### webcam_demo_circles2.py

Matches real-time pose to a pre-set pose based on lower body keypoints.



### Credits

The original model, weights, code, etc. was created by Google and can be found at https://github.com/tensorflow/tfjs-models/tree/master/posenet

This port and my work is in no way related to Google.

The Python conversion code that started me on my way was adapted from the CoreML port at https://github.com/infocom-tpo/PoseNet-CoreML

