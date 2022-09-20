# Introduction
This repo implements a video-based Person Re-Identification algorithm on a single camera. This project is a technical assessment of [Lauretta.io](https://lauretta.io/).

# Implementation - StrongSORT

I use the implementation of StrongSORT from [Mikel Brostrom](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet). To make it easier to use, I created an ipynb file to run inference quickly. To run the demo, do the followings:
1. Clone the repo by typing `git clone --recurse-submodules https://github.com/austenjs/Yolov5_StrongSORT_OSNet.git` in the terminal.
2. Install the dependencies by typing `pip install -r requirements.txt` in the terminal. (Make sure you have pip installed on your computer).
3. Open *main.ipynb* in Jupyter Notebook or Google Colab.
4. Click *Run All* at the top of the notebook file.
5. At the last cell, you could see the real-time demo of the video on a sample video.

**Note**: The real-time demo of the video can't be seen at Google Colab as `cv2.imshow()` is incompatible with Google Colab. Refer to this [link](https://github.com/jupyter/notebook/issues/3935)

# Reasons of choosing implementation
When I was deciding which model to use, I considered several factors:
1. Credibility of the paper
2. Credibility of the code
3. Implementation time

For factor 1, we can verify it by checking the number of citations. For factor 2, we can verify it by checking how many people have forked/starred the code repository in GitHub. For factor 3, we need to check whether the author has created an API that we can use conveniently. After factoring in all three factors, I decided to use the implementation of [Mikel Brostrom](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet). I did consider several other implementations:

- [Mhhtx](https://github.com/Mhttx2016/Multi-Camera-Object-Tracking-via-Transferring-Representation-to-Top-View):  Low credibility of the code (only 56 forks)
- [Kaiyang Zhou](https://github.com/KaiyangZhou/deep-person-reid): Need to implement inference script from scratch
- [Layumi](https://github.com/layumi/Person_reID_baseline_pytorch): Need to implement inference script from scratch
- [Ppriyank](https://github.com/ppriyank/Video-Person-Re-ID-Fantastic-Techniques-and-Where-to-Find-Them): Low number of citations

# Traditional Trackers

## SORT - Simple Online and Realtime Tracking

SORT is the first object-tracking algorithm that utilizes Convolutional Neural Networks. It consists of 4 parts:
1.	Detection: Detect objects on video frame using Faster R-CNN
2.	Estimation: Predict new states via Kalman Filter
3.	Target Association: Match detected objects to identities (IDs)
4.	Track Identity Life Cycle
a.	Creation of new object moving into the frame
b.	Deletion of objects moving out of frame

## DeepSORT: Simple Online and Realtime Tracking with a Deep Association Metric

DeepSORT is another object tracking algorithm that is better in SORT. Previously, SORT had the following issues: 
- High identity switches
- Issues in tracking through occlusions and different viewpoints 

DeepSORT then introduces the appearance feature to handle issues of SORT. A CNN model creates the appearance feature. The CNN model is a finetuned classification model with the classification head removed.	

With the introduction of the new feature, the loss function during target association is created using the combination of the:
- Mahalanobis distance between the predicted Kalman states and detected objects
- Cosine distance between the appearance feature vectors of the detected objects in the current frame and the appearance features of detected objects in the past 100 frames. 

The cosine distance considers appearance information that is particularly useful to recover identities after long-term occlusions when motion is less discriminative. The appearance model reduces the number of identity switches by 45% compared to SORT.

## FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking

In SORT and DeepSORT, the pipeline is as follows:
1. Object Detection
2. Real-time tracking
3. Re-Identification

Both of them are slow, and the result of re-ID and real-time tracking depends on the object detection ability. FairMOT is an object tracking algorithm that performs object detection and re-ID simultaneously. The model has an encoder-decoder backbone and two heads. The first head predicts:
- Heatmap
- Box size
- Center offset: Offset the center more

The second head extracts embeddings from each bounding box to do Re-ID. By doing the computation simultaneously, FairMOT has a faster inference speed than SORT and DeepSORT.
FairMOT is influenced by CenterTrack, an object detection algorithm that tracks objects as points instead of a box and is anchorless.

# StrongSORT vs Traditional Trackers

StrongSORT is very similar to DeepSORT. However, there are some changes, which are:
1. The appearance feature is now produced by a bigger model (e.g., ResneSt50) instead of a simple CNN
2. Usage of NSA Kalman Filter instead of Vanilla Kalman Filter
3. Usage of motion information when computing loss of target association
4. Usage of an efficient Appearance-free model (AFLink) to predict the connectivity between two tracklets using Multi-Layer Perceptron
5. Usage of Gaussian smoothed interpolation (GSI) over Linear Interpolation

# References
- https://arxiv.org/pdf/2202.13514v1.pdf
- https://arxiv.org/pdf/1703.07402.pdf
- https://arxiv.org/pdf/2004.01888.pdf
- https://wandb.ai/vbagal/Multi-Object%20Tracking/reports/Yolov5_DeepSort-vs-FairMOT--Vmlldzo4Nzk0MjQ
- https://nanonets.com/blog/object-tracking-deepsort/
- https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet
- https://paperswithcode.com/task/video-based-person-re-identification
