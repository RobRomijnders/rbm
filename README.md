# Restricted Boltzman machines

When discussing the deep learning research, one topic will unanimously pop up: the restricted Boltzman machine (RBM). Jeff Hinton's lab at the University of Toronto used as a first attempt to do representation learning. The RBM specifies an energy model over hidden units. In representation learning, such hidden units are said to represent learned features about the inputs.

# Motivation
This implementation is part of my exploration of the machine learning fundamentals. People recently have ridiculed the _grad student descent_. This term describes machine learning papers that incrementally change neural network architectures to achieve small improvements and publish papers on it. Therefore, I started coding some of the fundamental algorithms in machine learning. I hope they give insight in the field of machine learning without reading about thousands of different neural network architectures.

# The equations for an RBM
The RBM defines an energy model over the visible units and hidden units. Here's an image of the associated graphical model:

