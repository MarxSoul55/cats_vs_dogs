Keep constant:
* 3x3 convolutions with zero-padding to keep dimensions constant
* pooling layers order and structure (in order to progressively reduce dimensions)
* flatten layer followed by dense output (2 nodes)
* elu activation function
* orthogonal initialization

Let change:
* amount of convolutional layers and the number of filters (up to a maximum of 128 for
  efficiency's sake) and order in which they are connected
* skip-connections between convolutional layers
