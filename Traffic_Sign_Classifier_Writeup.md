# Self-Driving Car Engineer Nanodegree

## Writeup: **Build a Traffic Sign Recognition Classifier**

### Adam Tetelman

The goals / steps of this project are the following:
 - 0 Download, parse, and save data
 - 1 Explore and summarize data
 - 2 Build and train a model
 - 3 Test the model
 - 4 Explore layers of the network
 - 5 Summarize results in this written report

### Step 0 - Download, parse, and save data

In this step I download and verify the zip file containing all the image data. I resize all the images to (32, 32) using PIL. This is done under the `Download & Validate data` section.

In the `Split Data Into Train, Test, and Validation` section I define a size for my train data (60%), my validation data (30%) and my test data set (10 %).

I then proceed to the `Save, Restore, and Validate Data` section where I use pickle to serialize my datasets to disk for re-use and follow up with some basic validation. I then proceed to calculate the frequency of each label across the enire dataset and some more advanced numbers such as the max, min, mean, and std deviation of those counts.

### Step 1 Explore and summarize data

In this step I take a closer look at the dataset. I start off `Helper Functions` by defining some image processing functions. I use these later in my pre-processing pipeline, but I define them here so that I can visually demonstrate their effect on the sample images.

I begin the exploration in `Validate & Print Basic Dataset Information`  by printing off the general shape/size of each dataset and the number of labels.

- Total number of examples = 39209
- Number of training examples = 23525
- Number of cross validation examples = 11763
- Number of testing examples = 3921
- Image data shape = (32, 32, 3)
- Number of classes = 43

I next print a few random images: the original, grayscale, and normalized image.

![Sample images][signs]

The next line of code uses the previously calculated numbers to plot the counts of each label utilizing bar plots and a cumulative Pareto distribution. These plots demonstrate that some labels occur much more frequently than others, but that difference is not enough to cause much concern (~50% of the labels make up ~80% of the data).

![Label count][count1]
![Label Distribution][count2]

I finish off with a verbose frequency count of the sign labels and a brief analysis.

### Step 2 Build and train a model

I start this step off by defining my preprocessing using the previously defined functions and then process my X data in `Pre-process the X values`.

The next block of code contains a full definition of my model, followed by some helper functions and tensors that I use during training and evaulation. The training parameters are defined under `Define Parameters`.

Under `rain the Model` is where I do the training. I simple load up a Tensorflow session, initialize my variables, and train the specified number of `EPOCHS`. In each epoch I use my get_batches function to break the training data into batches. I run my `training_operation` tensor against each (x_batch, y_batch). At the end of each epoch I evaluate the accuracy against my validation set and save the model for later use/training.


### Step 3 Test the model

This step begins in `Run Against Test Datase` with a simple evaulation of the datasets. The results were 94.8% for test, 95.5% for validation, and 100% for train. The fact that test and validation scored similar makes me think the model is relatively general, however the 100% accuracy in train makes me think I either overfit the model or have a bug in my evaluation function.

Under the `Dig deeper into the accuracy` section I dig a bit deeper into the results of the test set to make sure nothing went wrong with the model. The plots show that there are a handful of signs that I incorrectly identify 10% to 20% of the time, while the remainder seem to be identified correctly around 95% of the time. It seems like the evaluation function was working correctly.

![Accuracy Percentage on Test][accuracy]

I finish up under `Finding & Loading New Data` Where I downloaded some additional images. Several of the images contain signs with types not included in the original training set, I was curious how the model would react. I load and resize these images using the same functions I used on the original set.

![New Sign With Predictions][newsign]

I proceed to `Make Predictions` where I run the images through my prediction function and print out the top 5 softmax probabilities. Many of the probabilities showed up as 100% or 0% which makes me wonder what the model is doing. However the 2 images that were incorrectly identified had somewhat lower probabilities, although still in the 90% range.

I was able to accurately predict a 'no passing sign', a 'no passing for vehicles over 3.5 metric tons' sign, a 'stop' sign, two 'no entry' signs, a 'general caution' sign, a 'bumpy road' sign, a 'road work' sign, and a 'no passing' sign.

I misidentified a flipped 'road work' sign with an all white background as a 'speed limit 30km/h' sign. This was rendered image, not a picture. The actual picture of a 'road work' sign was predicted correctly.

I misidentified a 'no passing' sign as a 'right of way at the next intersection' sign. I believe this was misidentified to largly to it being more cropped than most sample images.

I finish this section up by plotting each image along with the resized and preprocessed image and reuse my evaulation function under `Analyze Performance` to get an accuracy score of 80%.

Overall I was surprised by how well the model did. It had quirky behavior on the images that were from new labels and it was wrong with images that were heavily cropped or on bright background. But I did not cherrry pick these images or modify them beyond what has been described and there was an 80% accuracy. 

### Step 4 Explore layers of the network

In this last exploratory section I define the outputFeatureMap function to take in a stimuli image and network layer, and output a series of images displaying what each learned feature looks like within that layer for that image. I follow up with a quick analysis of the features. 

In summary, it appears as if the first convolutionsl layer has learned to use brightness to identify the edges and shapes of the sign. 

![Layer 1 maxpool identifying the sign borders][conv1]

It appears as though the second convolutional layer takes the sign image and then identifies basic lines and curves within the sign. I assume the last fully connected layer is able to determine which combination of lines and curves define each sign label. 

![Layer 2 relu identifying sign curves/edges][conv2]


### Pre-processing Pipeline Choice

I start off by resizing the images to 32x32x3 images. The images contained within the dataset were of varying sizes, and not too far off from this size. I did this to make pre-processing and model definition easier down the road. Additionally this allowed me to fit much more of the data in memory and my initial exploration into this parameter showed it did not decrease accuracy by much. *The variable name is `img_resize` it is set to `(32, 32)`*

After resizing the image I convert it to a numpy array. I specify the dtype as a np.float32. In some images this may have a downscaling effect. This makes the images more consistent, prevents some amount of overfitting, and allows me to fit more images in memory.

I then convert to grayscale. Processing a 32x32x1 image is easier and requires less memory than 32x32x3. This allowed me to more easily normalize the data and adds some level of noise to help ensure I don't overfit to the style of image in the dataset.

I then run my 32x32x1 grayscale image through the sklearn L2 normalization method. This normalizes the image values between 0 and 1, which helps to stabilize the network during training. It also adds a small amount of noise and causes images to look more similar to prevent overfitting.

### Model Choice
I chose to use the LeNet model for this classification problem. I initially planned to start off with this model and modify it as I went, however the initial results proved high enough for me to stick with Le-Net-5 as my final model.

The input from the pre-processing stage is a 4D numpy array of size (m, 32, 32, 1). Each row of this array contains a single processed grayscale image.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| mx32x32x1 np array of grayscale images  		| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs mx28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs mx14x14x6  				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs mx10x10x16 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs mx5x5x16   				|
| Flatten				| outputs  mx400								|
| Fully connected		| outputs, mx120        						|
| RELU					|												|
| Fully connected		| outputs mx84   								|
| RELU					|												|
| Fully connected		| outputs mx43   								|
|:---------------------:|:---------------------------------------------:| 

In addition to the model, I apply a softmax to the logits when training and making predictions.

### Training Parameter Choice

I chose a batch size of 512.  When I ran with a very large batch size I noticed that the signs that did not appear as frequently were misidentified more often. When I chose a very small batch size training took a long time. I chose 512 because training happened at a reasonable rate, it fit in memory on my laptop (versus using a cloud setup), and it gave a reasonable accuracy. The variable name is `BATCH_SIZE`

I chose a value of 4000 for epochs.I noticed that aroud 500 epochs was where accuracy got to around 80% and the cross validation accuracy slowly rose until about 4000 epochs when the increase in accuracy slowed dramatically. At around 10,000 epochs I saw a cross validation of above 97%, but I chose to go with 4000 epochs to avoid overfitting and decrease training time. The variable name is `EPOCHS`.

I chose a learning rate of .001. This is within the recommended range and gave relativly good results when training. I attempted a value of .01, during which I saw my training became unstable and accuracy jumped all over the place for ~50 epochs. .001  Gave stable training within a reasonable speed. It may have been worth exploring this deeper, however this value worked well for me.

I chose to use the Adam optimizer to reduce the mean of my cross_entropy function. This optimizer was recommended above using simple SGD and gave good results in a reasonable time. I did not play around much with this option, as I considered it a somewhat standard approach to training.


### Data distribution Choice

Before I serialize my data to disk I break it up into 3 datasets using the sklearn train_test_split function. I defined the breakup by ratio rather than sample count and used ratios based off what I have used in the past and has been recommended. I set aside 60% of the data for training. I set aside 30% of my data for validation. I set aside 10% of my data vor testing. I verify that each of the three datasets contains a sampling of each class.

I also use random seed values for the data split to ensure I get reproducable results.

Based on the results I got in the Testing phase of my project, I believe it would be wise to  grow my original dataset by including horizontally and vertically flipped versions of each image; this is something my network had trouble identifying.

### Problem solving approach

I began the process of solving this problem by getting my python fully functional. I verifed that with a single click I was able to run through all steps detailed above. My initial model and training parameters were fairly simple and copied from previous work.

Once I had a notebook that I knew worked I took what I learned from class and applied it to the problem. The general consesus on image classification seemed to indicate that converting to grayscale and normalizing your images was a good thing, and this step made the modelling a lot easier. The first thing I did was expand my preprocess function.

After writing the preprocess function I implemented my model. Based on suggestions from class I used my prior work with LeNet and implemented a similar model in this project.

I then set my epochs very high and played around with a few of the parameters and the model. The first thing I did was play with the activation functions. I tried replacing the relu activations with sigmoid activations; after several hundred epochs this didn't seem to be getting any accuracy on the training set.

I then tried increasing and decreasing the learning rate. A higher learning rate resulted in inconsistent accuracy growth/decline. A smaller learning rate did not seem to add much. I decided to go with my original rate selection.

I then did something similar with the batch size. I played around with low values (1, 4, 16, 32, 64) and higher values (10,000, 15,000). I decided on 512 because it fit in-memory on my laptop, did not overfit certain signs, and ran quicker than small values.

I did not play around at all with changing my data size ratios, but in retrospect this is something I could have done. I did, however, avoid running my model against the test set until I had chosen my model.

### Summary

I was given the task of identifying German Street signs given a training dataset.

I automated the process of downloading, parsing, and extracting the sample into something usable by Tensorflow.

I analyzed the image data and noted some peculiarities to be aware of.

I defined a Convolutional Neural Network based on prior research in image classification. I tuned the parameters of that network and trained it.

I took my trained model and applied it to my test set and a new test set for an accuracy of 95% and 80%.

I inspected and analyzed each layer of my network to gain insight into what happened in the "black box."

[signs]: ./image/signs.png "Sign Samples"
[count1]: ./image/count.png "Label Count"
[count2]: ./image/count2.png "Cumulative Distribution"
[accuracy]: ./image/accuracy.png "Test accuracy"
[newsign]: ./image/newsign.png "New Sign"
[covn1]: ./image/conv1.png "conv1"
[conv2]: ./image/conv2.png "conv2"