# PointCloudCPP

## Overview

This repository contains the code that creates a complete coverage path for the robot to execute with the linear ultrasonic probe. The code should be applied to segments deemed to be planar through the hierarchical face clustering segmentation algorithm. Code contained here has either been integrated into the Scan_to_plan_to_scan repository, where the coverage path generated here can be verified with RoboDK and sent as commands to the robot, or is kept here as an archive. 

## Files Description

There are two different methods of planning paths in this repository. The primary method fits a line to the two primary edges and interpolates between them. This guarantees that the edges of the segment will be well followed, ensuring data collection and protection from collisions with the part (but often leads to redundant coverage). Both the boundary_detection.py and the plain_Best_fit.py scripts contain this edge-fitting method. boundary_detection.py is written as an object-oriented script, relying completely on the classes and functions found in the PCAClass.py script. plain_Best_fit.py script is a procedural script that uses the same functions and accomplishes the same as the boundary_decection.py but without the additional class structures used to simplify the code.

The second path-planning method is the naive method. This method differs from the edge-fitting method by only relying on a bounding box to get the scanning direction. A plane parallel to the scanning direction is intersected with the segment at even intervals. While this method in theory also yields complete coverage, additional verifications must occur to ensure the path is collision-free and can hold a water column. The redundant coverage issue is still present, and the more mismatch between the bounding box and the segment, the worse this method will perform. It is contained in the procedural script boundingbox.py and is stored in this repository as an archive of previous path-planning methods.

The other files in the repository are auxiliary. Num_pass_calc.py explores the necessary number of passes needed to completely inspect a part, including the spacing of each pass for various edge-offsetting processes. The plane_segments folder contains a variety of .stl files, each a planar segment from a real scan taken of our spar. Finally, the segmentscan.rdk is a RoboDK file, containing the RoboDK station and is needed to interface with the RoboDK API.

## Installation
The scripts in this repository are written in Python 3.11, and do require additional external libraries not native to Python. The libaries, and the version confirmed to integrate with all other libraries are listed here:

- **`open3d`**: version 0.18.0
- **`numpy`**: version 1.26.4
- **`scipy`**: version 1.14.1
- **`matplotlib`**: version 3.9.2
- **`robodk`**: version 5.7.5
- **`math`**: native
- **`itertools`**: native
- **`collections`**: native

These python packages can be installed with the included 'requirements.txt' with the following command in the appropriate directory
```
pip install -r requirements.txt
```

Most of these libraries do not affect each other, but numpy versions 2.xx.x will cause the visualization window of open3d version 0.18.0 to crash. 

## Usage
To use the code in this repository in a stand-alone manner, simply upload the .stl file of the segment you desire to inspect to the plane_segments folder. Then change the name in the mesh-import portion of the code in any of the scripts to the name of 

## Authors and acknowledgment
Jonas Beachy with help from Richard Lu, Professor Xu Chen, Andrew Na, and Bill Moon

## License
IP generated under the UW-Boeing master agreement