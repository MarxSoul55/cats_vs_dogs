# Requested Contributions
Please make all contributions compliant with [PEP 8](https://pep8.org/).
* The preprocessing method
`cats_vs_dogs.src.pytorch_impl.src.preprocessing.ImageDataPipeline.preprocess_classes` relies on
incrementing a flag `step` to indicate when the `while` loop should be terminated. At first glance,
it may seem like an easy fix—just use a `for` loop instead of the `while` loop! However, it's not
that simple. Doing something like `for _ in range(steps)` causes an entire run through each class
to be considered *one step*. This is incorrect—each run through an *image* is what I'm considering
a step to be (i.e. a gradient update).
    * The code I wrote achieves this vision, but it's quite ugly. If someone could refactor it or
    figure out a different way to approach the issue, that would be great!
    * Note that this is effectively duplicated in the TensorFlow implementation, but the PyTorch
    version is the "primary" implementation that I'm focused on. Focus your contributions in that
    one, and then I'll see to updating the TensorFlow one.
* Figure out a way to get TensorFlow's
[SavedModel](https://www.tensorflow.org/programmers_guide/saved_model#save_and_restore_models)
to work as a replacement for `tf.train.Saver` for saving and loading not just the model, but also
all relevant variables and operations (e.g. objective function, summary ops, etc.).
    * Ensure that you're working with the *TensorFlow* implementation, not the *PyTorch* one!
* A general refactoring of the TensorFlow version is desirable. The TensorFlow library, while
powerful, is very complicated and sometimes very unintuitive. My implementation was based off of my
own understanding of the library, but it may not fit how the authors of TensorFlow see it. I
encourage you to go through the code and refactor it where you see fit! This is very much
appreciated!
