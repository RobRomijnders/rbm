# Restricted Boltzman machines

When discussing the deep learning research, one topic will unanimously pop up: the restricted Boltzman machine (RBM). Jeff Hinton's lab at the University of Toronto used as a first attempt to do representation learning. The RBM specifies an energy model over hidden units. In representation learning, such hidden units are said to represent learned features about the inputs.

# Motivation
This implementation is part of my exploration of the machine learning fundamentals. People recently have ridiculed the _grad student descent_. This term describes machine learning papers that incrementally change neural network architectures to achieve small improvements and publish papers on it. Therefore, I started coding some of the fundamental algorithms in machine learning. I hope they give insight in the field of machine learning without reading about thousands of different neural network architectures.

# The equations for an RBM
The RBM defines an energy model over the visible units and hidden units. Here's an image of the associated graphical model:

![rbm_pgm](https://github.com/RobRomijnders/rbm/blob/master/im/Selection_554.png?raw=true)

The full equation reads:
![equation](https://latex.codecogs.com/gif.latex?p(v,h)&space;=&space;\frac{1}{Z}&space;e^{-E(v,h)}&space;\&space;\&space;\&space;\&space;\&space;E(v,h)&space;=&space;-b^Tv&space;-&space;c^Th&space;-&space;v^TWh)

#Inference in the RBM
To read about the inference in the RBM, I recommend:

  * Chapter 43 on Boltzmann machines of __Information theory, inference and learning algorithms__ by David Mackay
  * Chapter 20 on Deep generative models of __Deep learning__ by Goodfellow, Bengio and Courville
  * Section 28.2 on Deep generative models of __Machine learning, a probabilistic perspective__ by Kevin Murphy

# Training the RBM with contrastive Divergence
Another way to read the equation for the RBM is an energy model with a partition function. So with `E(v,h)` the energy of the current assignment and `Z` as the partition function. Using the framework of Maximum Likelihood Estimation, training an RBM is like walking over the weight surface and trying to find the _best_ peak. In order to do so, we follow the gradient:

![gradient](https://latex.codecogs.com/gif.latex?\nabla_\theta&space;log&space;\&space;p(x;\theta)&space;=&space;\nabla_\theta&space;log&space;\&space;\tilde{p}(x;\theta))&space;-&space;\nabla_\theta&space;log&space;\&space;Z(\theta))

Following the math in [chapter 18 of the Deep Learning book](http://www.deeplearningbook.org/contents/partition.html) we rewrite this gradient to:

![gradient rewritten](https://latex.codecogs.com/gif.latex?\nabla_\theta&space;log&space;\&space;p(x;\theta)&space;=&space;\bold{E}_{x&space;\sim&space;data}&space;\nabla_\theta&space;log&space;\&space;\tilde{p}(x)&space;-&space;\bold{E}_{x&space;\sim&space;p(x)}&space;\nabla_\theta&space;log&space;\&space;\tilde{p}(x))

The left part of the gradient is named the _positive phase_ and the right part is named the _negative phase_. This has an intuitive interpretation. In the _positive phase_ we sample from the data. So we increase the probability of these samples. In the _negative phase_ we sample from the model. The left part has a negative sign. So we decrease the probability of these samples. Here is the image from [the Deep Learning book](http://www.deeplearningbook.org/contents/partition.html)] to visualize this:

![pos_neg_phase](https://github.com/RobRomijnders/rbm/blob/master/im/pos_neg_phase.png?raw=true)

# Data
The code works with the Google quick draw dataset. [Here](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap/?pli=1) you can download the numpy bitmaps

# Results
You can run it for example on the images of cars:
![car_data](https://github.com/RobRomijnders/rbm/blob/master/im/car_data.png?raw=true)
![car_samples](https://github.com/RobRomijnders/rbm/blob/master/im/car_samples.png?raw=true)
![dolphin_data](https://github.com/RobRomijnders/rbm/blob/master/im/dolphin_data.png?raw=true)
![dolphin_data](https://github.com/RobRomijnders/rbm/blob/master/im/dolphin_samples.png?raw=true)

As always, I am curious to any comments and questions. Reach me at romijndersrob@gmail.com

