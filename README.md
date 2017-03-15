## Project: **Build a Traffic Sign Recognition Classifier**
### Adam Tetelman

The goal of this project was to build a Convolutional Neural Network to identify Street Signs.

In this project I used the tools that were taught througout the "Introduction to Neural Networks," "MiniFlow," "Introduction to TensorFlow,"  "Deep Neural Networks," and "Convolutional Neural Network" lessons.

The project relies on the following packages:
 - python3
   - numpy (for data/array manipulation)
   - sklearn (for data pre-processing)
   - tensorflow (for deep learning)
   - jupyter (for this notebook)
   - matplotlib (for data exploration)
   - cv2 (for image processing - opencv2)  
   
The tools taught included:
  - Data Normalization
  - Neural Networks 
  - Convolutional Neural Networks
  - LeNet-5
  - Tensorflow NN implementation

The resources used included:
  - GTSRB training image dataset (downloaded in-line below)
  - A small collection of German street signs (located in ./datasets)


This project is broken down into 5 steps. 

Step 0 deals with downloading and parsing the data. During this phase I defined several helper functions to download, parse, etc. I download around 250MB of training data, convert it to numpy arrays, split it into a train, validation, and test set, and serialize it to a few pickle files.

Step 1 deals with exploring the dataset. During this phase I defined several helper functions to analyze and process the data. I process the data a bit, print some basic meta-data about the data, and plot a few stats about the training set. 

Step 2 deals with defining and training a model.  During this phase I defined several helper functions to evaluate, train, and predict. I also defined several helper tensors. I then define a  model based on the Convolution Neural Network (LeNet) demonstrated in previous labs. This section ends with a training segment that took roughly 400 minutes to run for 4000 Epochs on my Nvidia GTX 960M GPU.

Step 3 deals with testing the accuracy of that model. During this phase I calculated the accuracy on my train, validation, and test data. I imported several new test images from the Internet and scored/analyzed my acccuracy on that dataset. 

Lastly, step 4 is an optional exploratory stage looking into the learned features of the model. During this phase I define a few helper functions to print the activation of a convolutional layer of my network. I run several stimuli images through these functions and discuss what the output says about the learned features of my model.

Overall the results looked fairly good. I was able to perform with ~95 accuracy on my test set which was 10% of the overall dataset. I was only able to achieve ~80% accuracy over the new data and that drop in accuracy seemed in large part to pictures taken at odd angles, with too much cropping, or with odd coloring. I believe there may be some room for improvement in the pre-processing given the results of these new images. It is also likely that the model has overfit to the dataset a bit, and that re-training with a larger test set may yield more generalizable results.