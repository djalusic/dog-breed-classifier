[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, your algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

Along with exploring state-of-the-art CNN models for classification and localization, you will make important design decisions about the user experience for your app.  Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  Your imperfect solution will nonetheless create a fun user experience!


## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/udacity/deep-learning-v2-pytorch.git
		cd deep-learning-v2-pytorch/project-dog-classification
	```
	
__NOTE:__ if you are using the Udacity workspace, you *DO NOT* need to re-download the datasets in steps 2 and 3 - they can be found in the `/data` folder as noted within the workspace Jupyter notebook.

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 
4. Make sure you have already installed the necessary Python packages according to the README in the program repository.
5. Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.
	
	```
		jupyter notebook dog_app.ipynb
	```

__NOTE:__ While some code has already been implemented to get you started, you will need to implement additional functionality to successfully answer all of the questions included in the notebook. __Unless requested, do not modify code that has already been included.__

__NOTE:__ In the notebook, you will need to train CNNs in PyTorch.  If your CNN is taking too long to train, feel free to pursue one of the options under the section __Accelerating the Training Process__ below.



## (Optionally) Accelerating the Training Process 

If your code is taking too long to run, you will need to either reduce the complexity of your chosen CNN architecture or switch to running your code on a GPU.  If you'd like to use a GPU, you can spin up an instance of your own:

#### Amazon Web Services

You can use Amazon Web Services to launch an EC2 GPU instance. (This costs money, but enrolled students should see a coupon code in their student `resources`.)

## Evaluation

Your project will be reviewed by a Udacity reviewer against the CNN project rubric.  Review this rubric thoroughly and self-evaluate your project before submission.  All criteria found in the rubric must meet specifications for you to pass.


## Project Submission

Your submission should consist of the github link to your repository.  Your repository should contain:
- The `dog_app.ipynb` file with fully functional code, all code cells executed and displaying output, and all questions answered.
- An HTML or PDF export of the project notebook with the name `report.html` or `report.pdf`.

Please do __NOT__ include any of the project data sets provided in the `dogImages/` or `lfw/` folders.

### Ready to submit your project?

Click on the "Submit Project" button in the classroom and follow the instructions to submit!

## PROJECT RUBRIC 

### Files Submitted

[x]  The submission includes all required, complete notebook files.

### Step 1: Detect Humans

Question 1: Assess the Human Face Detector

[x] The submission returns the percentage of the first 100 images in the dog and human face datasets that include a detected, human face.

### Step 2: Detect Dogs

#### Implement a Dog Detector

Use a pre-trained VGG16 Net to find the predicted class for a given image. Use this to complete a dog_detector function below that returns True if a dog is detected in an image (and False if not).

#### Question 2: Assess the Dog Detector

[x] The submission returns the percentage of the first 100 images in the dog and human face datasets that include a detected dog.

### Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

#### Specify DataLoaders for the Dog Dataset

[x] Write three separate data loaders for the training, validation, and test datasets of dog images. These images should be pre-processed to be of the correct size.

#### Question 3: Describe your chosen procedure for preprocessing the data.

[x] Answer describes how the images were pre-processed and/or augmented.

#### Model Architecture

The submission specifies a CNN architecture.

#### Question 4: Outline the steps you took to get to your final CNN architecture and your reasoning at each step.

[x] Answer describes the reasoning behind the selection of layer types.

#### Train the Model

[x] Choose appropriate loss and optimization functions for this classification task. Train the model for a number of epochs and save the "best" result.

#### Test the Model

[x] The trained model attains at least 10% accuracy on the test set.

### Step 4: Create a CNN Using Transfer Learning

#### Model Architecture

[x] The submission specifies a model architecture that uses part of a pre-trained model.

#### Question 5: Model Architecture

[x] The submission details why the chosen architecture is suitable for this classification task.

#### Train and Validate the Model

[x] Train your model for a number of epochs and save the result wth the lowest validation loss.

#### Test the Model

[x] Accuracy on the test set is 60% or greater.

#### Predict Dog Breed with the Model

[x] The submission includes a function that takes a file path to an image as input and returns the dog breed that is predicted by the CNN.

### Step 5: Write Your Algorithm

#### Write your Algorithm

[x] The submission uses the CNN from the previous step to detect dog breed. The submission has different output for each detected image type (dog, human, other) and provides either predicted actual (or resembling) dog breed.

### Step 6: Test Your Algorithm

#### Test Your Algorithm on Sample Images!

[x] The submission tests at least 6 images, including at least two human and two dog images.

#### Question 6: Weaknesses and Improvements

[x] Submission provides at least three possible points of improvement for the classification algorithm.

### Suggestions to Make Your Project Stand Out!
(Presented in no particular order ...)

(1) AUGMENT THE TRAINING DATA
Augmenting the training and/or validation set might help improve model performance.

(2) TURN YOUR ALGORITHM INTO A WEB APP
Turn your code into a web app using Flask!

(3) OVERLAY DOG EARS ON DETECTED HUMAN HEADS
Overlay a Snapchat-like filter with dog ears on detected human heads. You can determine where to place the ears through the use of the OpenCV face detector, which returns a bounding box for the face. If you would also like to overlay a dog nose filter, some nice tutorials for facial keypoints detection exist here.

(4) ADD FUNCTIONALITY FOR DOG MUTTS
Currently, if a dog appears 51% German Shephard and 49% poodle, only the German Shephard breed is returned. The algorithm is currently guaranteed to fail for every mixed breed dog. Of course, if a dog is predicted as 99.5% Labrador, it is still worthwhile to round this to 100% and return a single breed; so, you will have to find a nice balance.

(5) EXPERIMENT WITH MULTIPLE DOG/HUMAN DETECTORS
Perform a systematic evaluation of various methods for detecting humans and dogs in images. Provide improved methodology for the face_detector and dog_detector functions.
