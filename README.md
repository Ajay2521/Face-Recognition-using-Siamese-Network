# Face-Recognition-using-Siamese-Network

## Face Verification 

"Is this the claimed person?" given two images and you have to determine if they are of the same person. 
- The simplest way to do this is to compare the two images pixel-by-pixel. If the distance between the raw images is below a chosen threshold, it may be the same person! 
- Of course, this algorithm performs poorly, since the pixel values change dramatically due to variations in lighting, orientation of the person's face, minor changes in head position, and so on.

#### rather than using the raw image, you can learn an encoding,  𝑓(𝑖𝑚𝑔) 
#### By using an encoding for each image, an element-wise comparison produces a more accurate judgement

- Eg A mobile phone that unlocks using your face is also using face verification. This is a 1:1 matching problem.

## Face Recognition 

"Who is this person?" - Eg person details like name. This is a 1:K matching problem. 

Deep learning algorithms fail to work well if we have only one training example.

One-shot learning is a classification or object categorization task in which one or a few examples are used to classify many new examples.

The principle behind one-shot learning is Humans learn new concepts with very little supervision.

### Problem in ConvNet

- A small training set is really not enough to train a robust neural network for this task. The feature vectors trained do not contain important information that can be used for the recognition of future images.
- Retraining the ConvNet every time the number of classes or dataset is increased is way too time and resource consuming.

### Solutions to one-shot learning

#### 1. Siamese network for one-shot learning

![image-2.png](attachment:image-2.png)

#### Siamese networks are based on a similarity function, which does not require extensive training

#### Takes input two images—one being the actual image and other the candidate image—and outputs the similarity between the two.

The two input images, if very similar, output = lower value

The two input images, if not similar, output = Higher value

degree of difference between the two images is compared with the threshold value(𝜏) (which is a hyperparameter), 

if degree of difference between 2 image < threshold -> output = same person

if degree of difference between 2 image > threshold -> output = different person


#### Both the images to be compared are passed through the same networks, called sister networks, having the same parameters and shared weights.

images are passed through a sequence of convolutional, pooling, and fully connected layers, and end up with a feature vector of a fixed size, say, 128 denoted by f(x1) as an encoding of the input image x1.

## FaceNet

FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, you can then determine if two pictures are of the same person.

### Tech Used

- one-shot learning to solve a face recognition problem
- triplet loss function to learn a network's parameters in the context of face recognition
- face recognition as a binary classification problem
- Map face images into 128-dimensional encodings using a pretrained model
- Images will be of shape  (𝑚,𝑛𝐻,𝑛𝑊,𝑛𝐶) .

## Encoding Face Images into a 128-Dimensional Vector

### face embedding

FaceNet is a model that, when given a picture of a face, will extract high-quality features from it and predict a 128-element vector representation of these features, called a face embedding. ace embeddings can then be used as the basis for training classifier systems on standard face recognition benchmark datasets.

### Using a ConvNet to Compute Encodings

- FaceNet model takes a lot of data and a long time to train so we take the weigth which is already trained by others

- faceNet network uses 160x160 dimensional RGB images as its input. Specifically, a face image (or batch of  𝑚  face images) as a tensor of shape  (𝑚,𝑛𝐻,𝑛𝑊,𝑛𝐶)=(𝑚,160,160,3)

- The input images are originally of shape 96x96, thus, you need to scale them to 160x160. This is done in the img_to_encoding() function.

- The output is a matrix of shape  (𝑚,128)  that encodes each input face image into a 128-dimensional vector

### Triplet loss integration

![image.png](attachment:image.png)

#### compare pairs of images and learn the parameters of the neural network accordingly

one “anchor” image and get the distance between it and the “positive” (matching) image

distance of the anchor image with a “negative” (non-matching) example

Triplet loss is a loss function for machine learning algorithms where a baseline (anchor) input is compared to a positive (truthy) input and a negative (falsy) input.

![image-2.png](attachment:image-2.png)

The triplet loss function formalizes this, and tries to "push" the encodings of two images of the same person (Anchor and Positive) closer together, while "pulling" the encodings of two images of different persons (Anchor, Negative) further apart.

For an image  𝑥 , its encoding is denoted as  𝑓(𝑥) , where  𝑓  is the function computed by the neural network.

Training will use triplets of images  (𝐴,𝑃,𝑁) :

- A is an "Anchor" image--a picture of a person.
- P is a "Positive" image--a picture of the same person as the Anchor image.
- N is a "Negative" image--a picture of a different person than the Anchor image

You would thus like to minimize the following "triplet cost":

$$\mathcal{J} = \sum^{m}_{i=1} \large[ \small \underbrace{\mid \mid f(A^{(i)}) - f(P^{(i)}) \mid \mid_2^2}_\text{(1)} - \underbrace{\mid \mid f(A^{(i)}) - f(N^{(i)}) \mid \mid_2^2}_\text{(2)} + \alpha \large ] \small_+ \tag{3}$$
Here, the notation "$[z]_+$" is used to denote $max(z,0)$.

Here, the notation " [𝑧]+ " is used to denote  𝑚𝑎𝑥(𝑧,0) .

Notes:

The term (1) is the squared distance between the anchor "A" and the positive "P" for a given triplet; you want this to be small.
The term (2) is the squared distance between the anchor "A" and the negative "N" for a given triplet, you want this to be relatively large. It has a minus sign preceding it because minimizing the negative of the term is the same as maximizing that term.
𝛼  is called the margin. It's a hyperparameter that you pick manually. You'll use  𝛼=0.2 .

Since using a pretrained model, don't need to implement the triplet loss function

### Contrastive loss for dimensionality reduction

#### Dimensionality reduction involves reducing the dimensions of the feature vector. 

If the classes are the same,  loss function encourages  -> output = feature vectors that are similar

If the classes are the different,  loss function encourages  -> output = feature vectors that are less similar

## Face Recognition

- Compute the target encoding of the image from image_path
- Find the encoding from the database that has smallest distance with the target encoding.
- Initialize the min_dist variable to a large enough number (100). This helps you keep track of the closest encoding to the input's encoding.
- Loop over the database dictionary's names and encodings. To loop use for (name, db_enc) in database.items().
- Compute the L2 distance between the target "encoding" and the current "encoding" from the database. If this distance is less than the min_dist, then set min_dist to dist, and identity to name.
