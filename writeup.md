## Vehicle Detection Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_nocar.jpg
[image2]: ./output_images/hog.jpg
[image3]: ./output_images/pipeline.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./test_videos_output/project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `search_classify.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then selected V channel of HSV color spaces and parameters (`orientations=11`, `pixels_per_cell=16`, and `cells_per_block=2`).  I grabbed a random images from car classe and displayed it to get a feel for what the `skimage.hog()` output looks like. I didn't explore a lots of parameters because it is mainly tested for SVM accuracy.

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I trained SVM with various combinations of parameters and compared the accuracy to choose the best parameters in  `search_classify.ipynb`.  
Notice that the false negatives and false positives are numbers of samples on all dataset. So consider the accuracy and time the 7th set is the best.  
  
| color_space | orient | pix_per_cell | cell_per_block | spatial_size | hist_bins | Accuracy | Time(s) | false negatives | false positives |  
|:-----------:|:-------:|:-----------:|:--------------:|:------------:|:---------:|:--------:|:---:|:-----------:| :-----------:|  
| YCrCb | 12 | 16 | 2 | 32 | 32 | 0.9944 | 0.1947 | 6 | 4 |  
| YCrCb | 9 | 8 | 2 | 64 | 64 | 0.9938 | 0.3596 | 9 | 2 |  
| RGB | 12 | 16 | 2 | 32 | 32 | 0.9865 | 0.1938 | 12 | 12 |  
| RGB | 9 | 8 | 2 | 64 | 64 | 0.987 | 0.3616 | 15 | 8 |  
| HSV | 12 | 16 | 2 | 32 | 32 | 0.9932 | 0.193 | 7 | 5 |  
| HSV | 9 | 8 | 2 | 64 | 64 |  0.9949 | 0.3684 | 7 | 2 |  
| YUV | 12 | 16 | 2 | 32 | 32 | 0.9955 | 0.1899 | 7 | 1 |  
| YUV | 9 | 8 | 2 | 64 | 64 | 0.9944 | 0.3692 | 7 | 3 |  


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
  
I trained a linear SVM using the best set of parammters from above. It is described by `train_model` function in `search_classify.ipynb`. And it's mainly consisted of `extract_features`, `StandardScaler` and `LinearSVC`. The whole pipeline is easy thanks to `skleran`!  
  
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search all pixels along x axis and the following ranges along y axis:  
  
|  |  |  |  |  |  |  |  |  |  |  
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|  
|scales|1| 1| 1.5| 1.5| 1.5| 2| 2| 3| 3|  
|ystart |400|416| 400| 432| 464| 400| 432| 400| 464|  
|ystop |464| 480| 496| 528| 560| 528| 560| 596| 660|  
  
It is demonstrated by `find_cars` in `pipeline.ipynb` how to implement a  sliding window search. Because the long the distance is, the small the car is. So the above segments including cars in pictures are smaller than below segments, which means the scale is increasing. The first scale is 1, which means 1 by 64 pixels. The top left part of the following picture shows all the sliding windows in read boxs.  
![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

