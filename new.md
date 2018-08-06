![Logo](https://github.com/MarxSoul55/cats_vs_dogs/blob/master/logo/Logotype%20CatsvsDogs%20Horizontal.png)

*Thanks to [nunojesus](https://github.com/nunojesus) for designing this logo!*
# Hello World!  :earth_americas:
This repository contains code for a computer-vision program to discern the difference between two
common household animals. The program uses a machine learning algorithm called a *convolutional
neural network*.

This algorithm has been implemented in two popular software libraries—[PyTorch](https://pytorch.org/)
and [TensorFlow](https://www.tensorflow.org). The data used to train the model was [Kaggle's dataset of cat and dog pictures](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data).

### Want to contribute?  :blush:
See the following file in the root of this repo: `CONTRIBUTING.md`
# Want to use and modify this program?  :heart_eyes:
### If you have no idea what you're doing here...  :grimacing:
If you want to use and modify this program to your fullest potential, you'll need to have *a lot*
of background knowledge about artificial neural-networks and machine-learning in general. There's
no way around these requirements, unfortunately. You have to put in the time, read lots of books
and blog-posts, take online-courses, etc. Going beyond this point, I'll assume you have that
knowledge, since I do not have anywhere near the time to explain everything. I will list a few
resources, however:
* http://neuralnetworksanddeeplearning.com/
    * Great online book. Many people will list this as a resource if you ask around. Well-written
      and very informative.
* http://cs231n.github.io/
    * Lecture-notes and other resources from a course at Stanford University.
    * Also see: https://www.youtube.com/watch?v=NfnWJUyUJYU&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=0
        * These are video-lectures for the notes in question! Very, very helpful!
* https://www.coursera.org/learn/machine-learning
    * Another well-respected resource like the first one listed. If you want a more traditional
      "lecture and listen" setting, this will suffice.
* http://www.deeplearningbook.org/
    * A bit of a newbie on the field, but has some top-notch authors behind it. They're well-known
      and well-respected in the field of AI.
* https://www.youtube.com/watch?v=aircAruvnKk
    * This video was created by 3Blue1Brown as part of a series, which is actually being created as
      I speak (October 15th, 2017). 3Blue1Brown has a reputation of teaching things in an intuitive
      manner with lots of nice visualizations.
* https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
    * A short introduction on the inner-workings of convolutional neural networks, a very popular
      machine learning algorithm (as of 2018) and the algorithm that this program uses for image
      recognition!
* https://www.youtube.com/watch?v=bxe2T-V8XRs&list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU
    * This video is part of a short series on artificial neural-networks. It's just a small
      overview of some of the general concepts. It does throw in a few implementation-level
      concepts, like gradient-descent with matrices, which is a nice feature.

Even having a solid background in machine-learning, you still need to (obviously) know how to
program in Python and how to use the TensorFlow framework. There are so many tutorials for both of
these that I won't even bother to list any, as a google-search will yield everything you seek.
### If you do have an idea of what you're doing...  :sunglasses:
*...follow these steps:*
### *Setting Up*
1. Download [Kaggle's dataset of cats and dogs](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data).
2. Download the release: [v0.6](https://github.com/MarxSoul55/cats_vs_dogs/releases/tag/v0.6)
    * **Note that these directions are all based off of v0.6**
3. You're going to want to keep most of the directory's structure as in the release you downloaded.
You should add the following folders inside of the package `cats_vs_dogs` (folder where the
`__main__.py` is):
    * bin/saved/pytorch_impl/saved (location of saved model for PyTorch implementation)
    * bin/saved/tensorflow_impl (location of tensorboard files and saved model for TF
      implementation)
    * data/train/cat (will contain the images of cats from Kaggle)
    * data/train/dog (will contain the images of dogs from Kaggle)
    * data/test (will contain the test images from Kaggle)
### *Using the CLI*
4. Now you need to learn how to use `model.py`, which acts as the interface by which you'll train
and test the model. There are five commands you should know:
    * `python model.py --train --steps 25000`
        * The `--train` tells the program to train the model, and the `--steps` tells the program how many
          "steps" it should take. By steps, I mean parameter-updates, which are performed
          image-after-image. In more technical terms, I'm doing stochastic gradient-descent with
          `batch_size=1`. Why did I not implement minibatch gradient-descent? Lack of computing
          resources. I don't personally have a lot of RAM on my machine, so I didn't bother.
        * In this example, I train on all 25000 images from Kaggle's training-set. You can tell
          because of the `--steps 25000`. But what if I used something like `--steps 26000`? The Kaggle
          dataset only contains 25000 images total! Simple, my preprocessing-function that I wrote
          in `preprocessing.py` will simply start over at the beginning of the directory. It's
          seamless, so don't worry about it.
        * During training, the preprocessor will preprocess an image for the model to train on from
          the first class (in this case either `cats` or `dogs`) and will then preprocess an image
          from the second class. After it does so, it will loop back around to the first class and
          pick some image that it hasn't trained on yet (order of picking is random, since we
          want to train on everything). Then it'll do the same for the second class, and continue
          looping over and over until it reaches the amount of steps that was specified by the
          user.
        * After the training is complete, the program generate (or overwrite if exists) two
          directories:
            * saved_model :floppy_disk: (self-explanatory)
            * tensorboard (VERY IMPORTANT—this is how you monitor your error/loss/objective function)
                * After a session of training for the model, navigate to the directory containing
                  the `tensorboard` subdirectory and run `tensorboard --logdir=tensorboard`. After
                  a minute or two, the terminal will spit out a URL which you can follow to see the
                  error function :chart_with_downwards_trend: plotted over time.
                * See https://www.tensorflow.org/get_started/summaries_and_tensorboard for details.
    * `python model.py --train --resuming --steps 25000`
        * Almost identical to the first command. Only thing different is the `--resuming` flag.
        * `--resuming` tells the program that it's *resuming* training (obviously).
        * With this flag, the training will start where the model's paramters' values last were.
          Those values are stored in the files in `saved` directory. It will overwrite `saved` and
          `tensorboard`.
    * `python model.py --classify --source data/test/1.jpg`
        * The `--classify` tells the program to test the model on an image, whose path is given by
          the portion after the `--source` flag: `data/test/1.jpg`. The program will print a string:
          either "cat" or "dog".
    * `python model.py --classify --source data/test`
        * Similar to the previous command, except this time you're going through every image in the
          `test` directory! This will print out a long dictionary of format {filename: prediction}.
        * Note that this command ignores subdirectories and files [whose formats aren't supported by
          OpenCV](http://amin-ahmadi.com/2016/09/24/list-of-image-formats-supported-by-opencv/).
        * Note that the way the dictionary is printed out is quite messy, and so it'll be hard to
          read for a big directory like `test` (which holds thousands of images from Kaggle).
          That's because this command is only here for testing purposes, so I didn't bother making
          it look nicer. If you want to print out the dictionary in a nicer way, you are welcome to
          change the code! If you want the actual dictionary (as a Python object), you should get
          it through `model.classify`.
    * `python model.py --classify --source https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg`
        * This time, the source is a URL! Notice the `.jpg` at the end, which indicates that this
          URL points to a picture! The program will grab the image and then print either "cat" or
          "dog", just like when you make `--source` point to a file on disk!
### *Some Final Words*
5. **Note that everything above will use the "default settings" which I've specified for my own
endeavors.** This means the architecture of the neural-network, its optimizer, its objective
function, etc. are all based off of my personal choices for my own experimentation. **The point is
that you are welcome AND encouraged to change the code for yourself.** In step 3 above, I've
provided some short descriptions of each file, but I believe that the code is fairly readable, and
I've provided lots of docstrings and some comments that can help you make sense of the code so that
you may modify it easily.
6. That's all there is to it! I highly encourage you to tinker around with my code and change it as
it suits your purposes—it currently suits mine very well, but for you, it may not! In previous
versions, I believe modifications were difficult, but as of **v0.3**, I think the code has become
much more beautiful, readable, and thus modifiable. Happy classifying!  :heart:
