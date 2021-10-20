# Recognizing-Human-Emotion-using-CNN
Convolutional neural network is a type of neural network that is commonly used in image data, that can be used to detect and recognize objects in an image.Convolutional Neural Network uses the convolution process by moving a convolutional kernel (filter) of a certain size to an image. After that the computer gets new representative information from the multiplication of parts of the image with the filter used. The first step of the algorithm is to split the image into overlapping smaller images and then each smaller image will be the input which will be fed into the neural network to produce a feature representation. So CNN can recognize an object, wherever it appears in the image. After that, the results will be downsampled, which is useful for retrieving the largest pixel in each kernel pool. In that way, even when reducing the number of parameters, the most important information from that section is still retrieved. Then the final process is the fully connected layer that is used to decide whether the images match or not.

After prepare the dataset, we split our data to
training,validation and test data. The next step is reshape and scale the images to 48 x 48
pixel grayscale image of faces. The faces have been
automatically registered so that the face is more or less
centered and occupies about the same amount of space in
each image. After scaling the images, we define the
convolution neural network model with convolution layer, pooling layer, fully connected layer and using Rectified
Linear Unit(ReLU) activation function to increase nonlinearity in the model.After that, testing the model using test data and obtain
predicted value.The accuracy score is about 54% on the test set and to
improve the model in the future is to use a confusion matrix
to extract the details

