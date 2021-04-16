# ISPR
ISPR midterms taken during the course 2020/2021
## Midterm 1 - Assignment 6 - Points 1.5/3
Implement the convolution of a set of edge detection filters with an image and apply it to one face image and one tree image of your choice from the dataset. Implement Roberts, Prewitt and Sobel filters (see here, Section 5.2, for a reference) and compare the results (it is sufficient to do it visually).  You should not use the library functions for performing the convolution or to generate the Sobel filter. Implement your own and show the code!
## Midterm 2 - Assingment 3 (non terminato)
Implement from scratch an RBM and apply it to DSET3. The RBM should be implemented fully by you (both CD-1 training and inference steps) but you are free to use library functions for the rest (e.g. image loading and management, etc.).

1. Train an RBM with 100 hidden neurons (single layer) on the MNIST data (use the training set split provided by the website).

2. Use the trained RBM to encode all the images using the corresponding activation of the hidden neurons.

3. Train a simple classifier (e.g. any simple classifier in scikit) to recognize the MNIST digits using as inputs their encoding obtained at step 2. Use the standard training/test split. Show the resulting confusion matrices (training and test) in your presentation.

(Alternative to step 3, optional) Step 3 can as well be realized by placing a softmax layer after the RBM: if you are up for it, feel free to solve the assignment this way.
