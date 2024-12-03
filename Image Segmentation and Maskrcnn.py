# Image Segmentation and Maskrcnn

# Theoretical Questions:

1.	What is image segmentation, and why is it important?

=> Image segmentation is the process of partitioning an image into meaningful regions or segments. It is essential for many computer vision applications, such as object recognition, medical image analysis, autonomous driving, and more.


2.	Explain the difference between image classification, object detection, and image segmentation.


=> 	Image Classification: Assigns a single label to an entire image.
=> Object Detection: Identifies and locates objects within an image, providing bounding boxes around them.
=> 	Image Segmentation: Divides the image into regions corresponding to different objects or semantic categories.


3.	What is the difference between traditional object detection models and Mask R-CNN?

=> 	Mask R-CNN extends traditional object detection models by adding a mask branch to predict pixel-level segmentation masks for each detected object.


4.	What role does the "RoIAlign" layer play in Mask R-CNN?

=> RoIAlign ensures that the features extracted from the backbone network are aligned with the corresponding pixels in the input image, preventing misalignment issues.


5.	What are semantic, instance, and panoptic segmentation?

=> Semantic Segmentation: Assigns a class label to each pixel in the image.
o Instance Segmentation: Identifies and segments individual instances of objects within the image.
o	Panoptic Segmentation: Combines semantic and instance segmentation, providing a comprehensive understanding of the scene.


6.	Describe the role of bounding boxes and masks in image segmentation models.

=> 	Bounding Boxes: Used to roughly locate objects within an image.
o	Masks: Provide pixel-level information about the exact shape and extent of objects.


7.	What is the purpose of data annotation in image segmentation?

=> 	Data annotation provides ground truth labels for training segmentation models. It involves labeling pixels with corresponding class labels.


8.	How does Detectron2 simplify model training for object detection and segmentation tasks?

=> 	Detectron2 provides a unified framework for training and evaluating object detection and segmentation models, simplifying the process and reducing development time.


9.	Why is transfer learning valuable in training segmentation models?

=> 	Transfer learning allows us to leverage pre-trained models on large datasets (like ImageNet) to improve the performance of segmentation models on smaller datasets.


10.	How does Mask R-CNN improve the Faster R-CNN model architecture?

=> 	Mask R-CNN adds a mask branch to Faster R-CNN, enabling it to perform instance segmentation in addition to object detection.


11.	What is meant by "from bounding box to polygon masks" in image segmentation?

=> 	It refers to the process of refining coarse bounding box predictions into precise pixel-level segmentation masks.


12.	How does data augmentation benefit image segmentation model training?

=> 	Data augmentation increases the diversity of training data, preventing overfitting and improving model generalization.


13.	Describe the architecture of Mask R-CNN, focusing on the backbone, region proposal network (RPN), and segmentation mask head.

=> •	Backbone: Extracts features from the input image.
•	RPN: Proposes regions of interest (ROIs) that might contain objects.
•	Segmentation Mask Head: Predicts pixel-level segmentation masks for each ROI


14.	Explain the process of registering a custom dataset in Detectron2 for model training.

=> •	Involves creating a dataset catalog, defining data loading pipelines, and providing ground truth annotations.


15.	What challenges arise in understanding the concept of semantic segmentation, and how can Mask R-CNN address them?

=> •	Semantic segmentation can be challenging due to ambiguities and variations in object appearances. Mask R-CNN can address these challenges by leveraging its powerful architecture and training strategies.


16.	How is the IoU (Intersection over Union) metric used for evaluating segmentation models?

=> •	IoU measures the overlap between predicted and ground truth segmentation masks.


17.	Discuss the use of transfer learning in Mask R-CNN for improving segmentation on custom datasets.

=> •	Transfer learning allows us to leverage pre-trained Mask R-CNN models on large datasets to improve performance on smaller custom datasets.


18.	What is the purpose of evaluation curves, such as precision-recall curves, in segmentation model assessment?

=> •	Evaluation curves help visualize the trade-off between precision and recall at different thresholds.


19.	How do Mask R-CNN models handle occlusions or overlapping objects in segmentation?

=> •	Mask R-CNN uses a non-maximum suppression (NMS) algorithm to filter out overlapping predictions and assigns masks to individual objects.


20.	Explain the impact of learning rate on Mask R-CNN model training.

=> •	Learning rate affects the speed and stability of convergence. A well-tuned learning rate is crucial for achieving optimal performance.


21.	Describe the challenges of training segmentation models on custom datasets, particularly in the context of Detectron2.


=> •	Challenges include data quality, annotation accuracy, and the need for careful hyperparameter tuning.


22.	How does Mask R-CNN's segmentation output differ from a traditional object detector's output?


=> •	Mask R-CNN provides pixel-level segmentation masks for each detected object, while traditional object detectors only provide bounding boxes.
"""

# Practical Questions:

# Q1 Perform basic color-based segmentation on an image.
import cv2
import numpy as np

# Load the image
img = cv2.imread('your_image.jpg')

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the range of colors you want to segment (adjust these values as needed)
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# Create a mask for the color range
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Apply the mask to the original image
result = cv2.bitwise_and(img, img, mask=mask)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load the Image: Reads the image using cv2.imread().
# Convert to HSV: Converts the image from BGR color space to HSV (Hue, Saturation, Value) color space. HSV is often preferred for color-based segmentation as it is more intuitive for human perception of color.
# Define Color Range: Sets the lower and upper bounds for the color range you want to segment. In this case, we're targeting blue colors. Adjust these values to target different colors.
# Create a Mask: Creates a binary mask where pixels within the specified color range are set to 1, and others are set to 0.
# Apply the Mask: Applies the mask to the original image, isolating the desired color region.
# Display the Result: Displays the segmented image using cv2.imshow().

#Q2  Use edge detection with Canny to highlight object edges in an image loaded

import cv2

# Load the image
img = cv2.imread('your_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# Display the result
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Explanation:

# Load Image: Reads the image using cv2.imread().
# Convert to Grayscale: Converts the image to grayscale as Canny edge detection works on grayscale images.
# Apply Canny Edge Detection:
# cv2.Canny() takes the grayscale image and two threshold values as input.
# Threshold1: Lower threshold for edge detection.
# Threshold2: Higher threshold for edge detection.
# Edges with intensity gradient greater than threshold2 are considered as edges.
# Edges with intensity gradient between threshold1 and threshold2 are considered as edges only if they are connected to strong edges.
# Display the Result: Displays the image with detected edges.

# Q3  Load a pretrained Mask R-CNN model from PyTorch and use it for object detection and segmentation on an image

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval() 1

# Load the image
img = Image.open('your_image.jpg')

# Convert the image to a PyTorch tensor
img_tensor = torchvision.transforms.ToTensor()(img)[None]

# Perform inference
with torch.no_grad():
    prediction = model(img_tensor)

# Extract the predicted boxes, labels, and masks
boxes = prediction[0]['boxes']
labels = prediction[0]['labels']
masks = prediction[0]['masks']

# Visualize the results
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')

for i in range(len(boxes)):
    box = boxes[i].detach().numpy()
    mask = masks[i].detach().cpu().numpy().squeeze()

    # Draw the bounding box
    plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none'))

    # Overlay the mask
    plt.imshow(mask, alpha=0.5, cmap='gray')

plt.show()

# Q4 Generate bounding boxes for each object detected by Mask R-CNN in an image
#  1. Load the pre-trained Mask R-CNN model:

import torch
import torchvision

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 2. Load the image:

import cv2

img = cv2.imread('your_image.jpg')

# 3. Convert the image to a PyTorch tensor:

import torchvision.transforms as T

transform = T.Compose([T.ToTensor()])
img_tensor = transform(img)[None]

# 4. Perform inference:

with torch.no_grad():
    prediction = model(img_tensor)

    # 5. Extract bounding box coordinates:

    boxes = prediction[0]['boxes'].detach().numpy()

    # 6. Visualize the bounding boxes:

    import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')

for box in boxes:
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.imshow(img)
plt.show()

# Q5 Convert an image to grayscale and apply Otsu's thresholding method for segmentation

import cv2

# Load the image
img = cv2.imread('your_image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] 1

# Display the result
cv2.imshow('Original Image', img)
cv2.imshow('Thresholded Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

#  Q6 Perform contour detection in an image to detect distinct objects or shapes

import cv2
import numpy as np

# Load the image
img = cv2.imread('your_image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding (you can adjust the threshold value)
thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# Display the result
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Q7 Apply Mask R-CNN to detect objects and their segmentation masks in a custom image and display them
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval() 1

# Load the image
img = cv2.imread('your_image.jpg')

# Convert BGR to RGB and convert to PyTorch tensor
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255

# Perform inference
with torch.no_grad():
    prediction = model(img_tensor)

# Extract predictions
boxes = prediction[0]['boxes'].detach().cpu().numpy()
labels = prediction[0]['labels'].detach().cpu().numpy()
masks = prediction[0]['masks'].detach().cpu().numpy()

# Visualize the results
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')

for i in range(len(boxes)):
    box = boxes[i]
    mask = masks[i].squeeze()

    # Draw the bounding box
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    # Overlay the mask
    masked_img = img.copy()
    masked_img[mask == 0] = 0
    plt.imshow(masked_img, alpha=0.5)

plt.show()

# Q8  Apply k-means clustering for segmenting regions in an image

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load the image
img = cv2.imread('your_image.jpg')

# Convert to RGB and reshape for K-Means
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.reshape((-1, 3))

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(img)

# Get the cluster labels
labels = kmeans.labels_

# Reshape the labels back to the image shape
labels = labels.reshape(img.shape[0], img.shape[1])

# Create a segmented image
segmented_img = np.zeros_like(img)
for i in range(kmeans.n_clusters):
    segmented_img[labels == i] = kmeans.cluster_centers_[i]

# Convert back to BGR and display
segmented_img = cv2.cvtColor(segmented_img.astype('uint8'), cv2.COLOR_RGB2BGR)
cv2.imshow('Segmented Image', segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
