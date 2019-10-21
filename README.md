# Hidden-Tree-Markov-Network
A library which implements the neuro-probabilistic model of the Hidden Tree Markov Network described in the paper "[Hidden Tree Markov Networks: Deep and Wide Learning for Structured Data](https://arxiv.org/abs/1711.07784)" by D. Bacciu.

## Content
This repository contains:
  - The implementation of the Hidden Tree Markov Network;
  - The implementation of the Graph Hidden Tree Markov Network, as a recurrent version of the previous model which generalizes the approach to graphs;
  - The preprocessing tools which allow to easily use both the versions of the model;
  - An example of usage and training of the Hidden Tree Markov Network on the INEX2005 and INEX2006 datasets;
 
## But also...
This model is an extension of the Relative Density Net, developed by A. D. Brown and G. E. Hinton in the paper "[Relative Density Nets: A New Way to Combine Backpropagation with HMM's](https://pdfs.semanticscholar.org/943d/5f9deb552d25d6b1377b09ad3fa00cbe7b3e.pdf)", and the modularity of the implementation allows to easily make use of different HMM's as input generative model.

### Author
This implementation has been developed as a final project for the Bachelor's Degree in Computer Science at the University of Pisa by Valerio De Caro.
