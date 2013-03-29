facex
=====

Geometrical Face Features Extraction


This software is the result of the first approachan effort to develop a geometrical 
facial features extraction algorithm. It was developed in Sao Paulo University (Brazil), 
and in cooperation with Universidad Politecnica de Madrid (Spain) and Carnegie Mellon 
University (USA). It covers mouth, eyes, eyebrows, nosetrils, and face tilt estimation.

It provides a very simple interface based on a single C/C++ class named "Face" that 
will compute the location of the mentioned points of geomatrical interest of the face 
the provided image. See "face.h".

Performance was totally prioriticed over precission. The extraction algorithm takes 
from 300usec to 1.5 ms on a desktop PC. It still has to be cleaned up from some 
unnecessary operations but shows promising results. 

Additionaly, the lack of accuracy can be easely palliated with temporal filtering 
techniques. This is shown in the template example "faceImitator.cpp".

We are working on a inteligent imitation algorithm for applications in social robotics
as well.

The algorithm is robust to 2D rotations but not to 3D changes in pose. Optical flow 
could be used for 3D tracking after face detection but it is not developed yet. This 
would strongly improve the robustness of the algorithm.


Please, note that this code was a research effort without commercial purposes so:

  
  - You are highly welcome to improve it and send it back to us to make it available!! ;-)

  - You may note that the code is a bit dirty, buggy and badly commented. Sorry for that.
  
  - There are many things that could be improved such as using color information to
    improve lips boundaries detection, and or add kalman/time filtering for accuracy.
  
  - It works fine in Ubuntu but can show some problems with integer arithmetics in Windows.

  - Feel free to contact to share ideas or contribute!.
  
This software uses the wel known and brilliant OpenCV library for image processing 
algorithms and you will need it to be able to build it: http://opencv.org/

You can find a description of the algortihm in the paper cited below. Please quote the
authors in any relevant work you make out of this software.


de Carvalho Santos, V.; Romero, R.A.F.; Coca, S.R.D.-M., "Imitation of Facial Expressions for a Virtual Robotic Head," Robotics Symposium and Latin American Robotics Symposium (SBR-LARS), 2012 Brazilian , vol., no., pp.251,254, 16-19 Oct. 2012
doi: 10.1109/SBR-LARS.2012.48
keywords: {computer vision;feature extraction;human-robot interaction;computer vision;eye extraction;eyebrows;face detection;face tilt angle;facial expressions;mouth characteristic points;natural interaction;nostrils;real-time geometric facial features extraction;social robots;virtual robotic head;Computer architecture;Eyebrows;Face;Feature extraction;Humans;Mouth;Robots;facial expression;facial features extraction;imitation},
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6363351&isnumber=6363255
