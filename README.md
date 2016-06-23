# Deep Columnar Convolutional Neural Network

DCCNN is a Convolutional neural network architecture which is inspired by the <a href="http://people.idsia.ch/~juergen/cvpr2012.pdf">Multi Column Deep Neural Network of Ciresan (2012).</a> 

Using improvements from recent papers such as Batch Normalization, Leaky Relu, Inception BottleNeck blocks and Convolutional Subsampling,
the network uses very few parameters in order to acheive near state of the art performance on various datasets such as
MNIST, CIFAR 10/100, and SHVN. 

Although it does not improve on the state of the art, it shows that smaller architectures with far fewer parameters can rival the performance of large ensemble networks.

# Architectures
## DCCNN MNIST

This architecture is simple enough for the MNIST dataset, which contains small grayscale images. It acheives an error rate of <b>0.23%</b> after 500 epochs.
The weights for this model are available in the weights folder.

![alt tag](https://raw.githubusercontent.com/titu1994/Deep-Columnar-Convolutional-Neural-Network/master/architectures/DCCNN%20MNIST.png)

## DCCNN SVHN

This architecture is similar to the MNIST dataset, but uses the SVHN dataset, of nearly 600,000 color images. It acheives an error rate of
<b>1.92%</b>. It was not possible to use the larger DCCNN CIFAR 100 architecture for this model, due to insufficient GPU memory.

Note that while similar, pooling is accomplished by using Convolutional Subsampling rather than Max Pooling.

![alt tag](https://raw.githubusercontent.com/titu1994/Deep-Columnar-Convolutional-Neural-Network/master/architectures/DCCNN%20SVHN.png)

## DCCNN CIFAR 10

This architecture is different from the above two, and is used for the CIFAR 10 datasets. It acheives a low error rate of <b>6.90%</b>.
This is higher than the state of the art <b>3.47%</b> but that is a extremely deep 18 layer network.

![alt tag](https://raw.githubusercontent.com/titu1994/Deep-Columnar-Convolutional-Neural-Network/master/architectures/DCCNN%20Cifar10.png)

## DCCNN CIFAR 100

This architecture is similar to the one above, but has a more forks at the last level. It acheives a low error rate of <b>28.63%</b>.
This is higher than the state of the art <b>24.28%</b> which has 50 million parameters and requires over 160,000 epochs. 

![alt tag](https://raw.githubusercontent.com/titu1994/Deep-Columnar-Convolutional-Neural-Network/master/architectures/DCCNN%20Cifar100.png)
