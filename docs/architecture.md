# Model Architecture

The model consists of an encoder-decoder architecture, where the model takes two inputs: BG and BG-FG and returns two outputs: Depth Map and Mask. The inputs are first individually processed through two encoder blocks each, which in turn reduces their size to 56x56, given the input shapes are 224x224. The reason for processing the two inputs separately for two encoder blocks is that:

- The output of these encoder blocks will be later fed as a skip connection to the last layers of the model, this might help the model in making better predictions as this way, the model will have the chance to see the two inputs separately in its last layers which may enhance its performance to identify the differences between the two inputs which is required to predict the mask and depth of the foreground object.
- An image of size 56x56 is sufficient for a convolutional model to extract information out of it. Thus we can apply two encoder blocks on the images separately before sending them to the network and not worry about losing information.

After encoding the inputs, they are merged together and sent through a series of encoder blocks until the image size becomes 7x7x512 (again a reminder, all these calculations are based on the fact that the input given to the model is 224x224).

After the encoding has been done, the encoded input is passed through two separate decoder networks which are responsible for predicting the depth map and the mask respectively. This encoder-decoder architecture is based on the famous convolutional neural network _UNet_.
The model architecutre can be seen below

![architecture](../images/architecture.jpg)

Each of the encoder and decoder blocks are based on a ResNet block. The detailed diagram explaining the flow of each block can be seen below

![blocks](../images/blocks.jpg)

The code for the full architecture can be found [here](../tensornet/model/dsresnet.py)
