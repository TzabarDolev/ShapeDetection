# ShapeDetection

**Itroduction:** <br>
A world that is full of triangles and circles has been found. <br>
The mission is to find as many objects as possible, get their correct classification and define the metrics to measure my success.

**Installation:**<br>
`git clone https://github.com/TzabarDolev/ShapeDetection.git`

**Project requirements appear at requirements.txt file**

**Usage:**<br>
To display an image with ground truth, prediction and my results:<br>
`python3 demo.py`

To apply the detection algo:<br>
`python3 main_detection.py`

To extract quality measures:<br>
`python3 quality_measures.py`

To show results:
`python3 show_results.py`

Quality measures metrics chosen:<br>
<h4>Precision, Recall, IOU</h4><br>
<h5>Precision: </h5>Precision is the ratio of the number of true positives to the total number of positive predictions. For example, if the model detected 100 trees, and 90 were correct, the precision is 90 percent.<br>
<h5>Recall: </h5>Recall is the ratio of the number of true positives to the total number of actual (relevant) objects. For example, if the model correctly detects 75 trees in an image, and there are actually 100 trees in the image, the recall is 75 percent.<br>
<h5>IOU: </h5>The Intersection over Union (IoU) ratio is used as a threshold for determining whether a predicted outcome is a true positive or a false positive. IoU is the amount of overlap between the bounding box around a predicted object and the bounding box around the ground reference data.<br>

<h5>prediction algo: </h5>
First of all, breaking the images to different images for circles and for triangles (most of the images are obvious - spheres are red and triangles are green. for those who are not - well, that's a sacrifice i'm willing to make. sort of.

<h5>Things I would have done i i had some spare time:</h5><br>
1. Add tqdm to long loops
2. Add legends to graphs
3. Try again using cv2.findcountours
