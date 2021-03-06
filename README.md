# ShapeDetection

**Itroduction:** <br>
A world that is full of triangles and circles has been found. <br>
The mission is to find as many objects as possible, get their correct classification and define the metrics to measure my success.

**Installation:**<br>
`git clone https://github.com/TzabarDolev/ShapeDetection.git`

**Project requirements appear at requirements.txt file**

**Usage:**<br>
First of all: after cloning the repo - add the images, ground truth and predictions to the appropriate folders<br>

To display an image with ground truth, prediction and my results:<br>
`python3 demo.py`

To apply the detection algo:<br>
`python3 main_detection.py`

To extract quality measures:<br>
`python3 quality_measures.py`

To show results:
`python3 show_results.py`

To preperly operate the files you need to have a directory named: results, img, ground_truth and prediction.<br>
`img` folder will hold the images.<br>
`ground_truth` folder will hold the ground truth json files
`prediction` folder will hold the prdictions given
`results` will gain the results once `python3 main_detection.py` is being operated.

Quality measures metrics chosen:<br>
<h4>Precision, Recall, IOU</h4>
<h5>Precision: </h5>Precision is the ratio of the number of true positives to the total number of positive predictions. For example, if the model detected 100 trees, and 90 were correct, the precision is 90 percent.<br>
<h5>Recall: </h5>Recall is the ratio of the number of true positives to the total number of actual (relevant) objects. For example, if the model correctly detects 75 trees in an image, and there are actually 100 trees in the image, the recall is 75 percent.<br>
<h5>IOU: </h5>The Intersection over Union (IoU) ratio is used as a threshold for determining whether a predicted outcome is a true positive or a false positive. IoU is the amount of overlap between the bounding box around a predicted object and the bounding box around the ground reference data.<br>

<h5>Prediction algo: </h5>
First of all, breaking the images to different images for circles and for triangles (most of the images are obvious - spheres are red and triangles are green. for those who are not - well, that's a sacrifice i'm willing to make. sort of. After all, I didn't want to be SOTA, I wanted to do something **good** at somewhat close to 8 hours.<br>**After some experiments it seemed that the original images without the seperation actually performed better, so I commented this section out.**<br>
I used cv2.HoughLinesP and cv2.Canny edge detection to locate **lines** of the triangles ,and cv2.HoughCircles locate circles.<br>
Finding lines is similar to finding triangles, given that setting a rectangle around a triangle and setting 3 rectangles surrounding each side of it, gives somewhat same area of IOU, and performed better on triangles that weren't exactly clear. Non maximum suppression was added to reduce overlaps. Finding lines performed better than cv2.findContours for the triangles<br>
.
I wanted to use simple tools like image gradients and color changes so opencv is a great catch for these ones. Also, we wanted to stay with *classic computer vision* so we didn't need here some big guns like attention, keypoints and CNNs.<br>
Detection algo ends with a json dump to the results folder. It's important to create the folder because I didn't create it automatically (8 hours and so).
**python3 quality_measures.py** will give you dataframes of all performances in **pred_performance.txt** and **results_performacen.txt**.<br>
Use the **python3 show_results.py** to see the top and worst k predictions, and also see comparioson graphs. <br>
The comparison graphs were meant to give the trend of the images. So if some local minima appear at some prediction case, it may appear also (only lower minima) at my results case. Basically it can help us understand wether we were correct on the easy spots and wrong at the tough ones, and I think it shows something like this. Also, it can be seen clearrly at the 5 samples example.<br>

<h5>Remark:</h5>
I had to modify your predictions a little bit, because you had bounding boxes that exceeded image pixels, which don't really make sense. So for each case of negetive pixel values or beyond image shape i gave my quality measures corrected values of prediction bounding boxes.<br>

<h5>Performance:</h5>
IOU: 0.74<br>
IOU_baseline: 0.95<br>

Precision: 0.86<br>
Precision_baseline: 0.98<br>

Recall: 0.83<br>
Recall_baseline: 0.96<br>


<h5>Images: (expand to see something useful)</h5>
Top 5 IOU:
<img src="assets/IOU_top_k_GT.jpg"> 
<img src="assets/IOU_top_k.jpg"> 
Worst 5 IOU:
<img src="assets/IOU_worst_k_GT.jpg"> 
<img src="assets/IOU_worst_k.jpg"> 
IOU comparison:
<img src="assets/IOU_comparison_example.jpg">
<img src="assets/IOU_comparison.jpg">
Top 5 precision:
<img src="assets/precision_top_k_GT.jpg"> 
<img src="assets/precision_top_k.jpg"> 
Worst 5 precision:
<img src="assets/precision_worst_k_GT.jpg"> 
<img src="assets/precision_worst_k.jpg"> 
Precision comparison:
<img src="assets/precision_comparison_example.jpg">
<img src="assets/precision_comparison.jpg">
Top 5 recall:
<img src="assets/recall_top_k_GT.jpg">
<img src="assets/recall_top_k.jpg">
Worst 5 recall:
<img src="assets/recall_worst_k_GT.jpg"> 
<img src="assets/recall_worst_k.jpg"> 
Recall comparison:
<img src="assets/recall_comparison_example.jpg">
<img src="assets/recall_comparison.jpg">
Recall doesn't look good, but recall calculates TP/(TP+FN), and once almost everything is being marked, there aren't any FN.



<h5>Things I would have done i i had some spare time:</h5><br>
1. Add legends to graphs<br>
2. Try again using cv2.findcountours<br>
3. Add a code to automatically create folders which don't exist<br>
4. Automate show_results.py file<br>
5. Create a fast-track for a single specific image<br>
6. Add ROC curve measure<br>
7. Seperate 20% of ground truth to "train" on it (in terms of how many triangles/circles, sizes, distances and so on<br>
8. debug non max suppression.<br>
