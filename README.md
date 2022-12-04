# Summary

This project aims to develop an BTD system from MRI (Magnetic Resonance Imaging) scans. This system will help clinicians to make an accurate diagnosis in order to improve the survival of patients. This system must provide a set of requirements and must be designed according to a deep learning architecture . It must explore different MRI scans and must make use of a set of well - defined metrics for its performance evaluation.

### Choice of Datasets
For the realization of our Project, we opted for data available on "Kaggle". 
For our  Deep learning models we choose mainly “Brain MRI Images dataset”.
Link:https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Choice of Models
     	In modern days, checking the huge number of MRI (magnetic resonance imaging) images and finding a brain tumor manually by a human is a very tedious and inaccurate task. It can affect the proper medical treatment of the patient. Again, it can be a hugely time-consuming task as it involves a huge number of image datasets. There is a good similarity between normal tissue and brain tumor cells in appearance, so segmentation of tumor regions becomes a difficult task to do. So there is an essentiality for a highly accurate automatic tumor detection method.
      Deep Learning Models : 
## CNN : 
  Deep learning is a handy and efficient method for image classification. Deep learning has been widely applied in various fields including medical imaging, because its application does not require the reliability of an expert in the related field, but requires the amount of data and diverse data to produce good classification results. Convolutional Neural Network (CNN) is the deep learning technique to perform image classification.
 Brain tumor detection using convolutional neural networks (CNN) CNN presents a segmentation-free method that eliminates the need for 

hand-crafted feature extractor techniques. For this reason, different CNN architectures have been proposed by several researchers.
Our CNN based model will help the doctors to detect brain tumors in MRI images accurately, so that the speed in treatment will increase a lot.
### Techniques that will be used To Improve The Accuracy of CNN :
While we develop the Convolutional Neural Networks (CNN) to classify the images, It is often observed that the model starts overfitting when we try to improve the accuracy. Very frustrating, Hence I list down the following techniques which would improve the model performance without overfitting the model on the training data.
1.Data normalization: The normalization of an image consists in dividing each of its pixel values by the maximum value that a pixel can take (255 for an 8-bit image).
2.Batch normalization: After each convolutional layer, we added a batch normalization layer, which normalizes the outputs of the previous layer. This is somewhat similar to data normalization, except it’s applied to the outputs of a layer, and the mean and standard deviation are learned parameters.
Architecture Preview :
![image](https://user-images.githubusercontent.com/84160502/205516984-4192000e-0d67-4a70-b169-064233f68419.png)


##Vgg16 : 
  A convolutional neural network is also known as a ConvNet, which is a kind of artificial neural network. A convolutional neural network has an input layer, an output layer, and various hidden layers. VGG16 is a type of CNN (Convolutional Neural Network) that is considered to be one of the best computer vision models to date. The creators of this model evaluated the networks and increased the depth using an architecture with very small (3 × 3) convolution filters, which showed a significant improvement on the prior-art configurations. They pushed the depth to 16–19 weight layers making it approx — 138 trainable parameters
Why Vgg16 ?
VGG16 is an object detection and classification algorithm which is able to classify 1000 images of 1000 different categories with 92.7% accuracy. It is one of the popular algorithms for image classification and is easy to use with transfer learning.
### Techniques that will be used To Improve The Accuracy of Vgg16 :
Like CNN we will use the techniques for improving the accuracy + data augmentation
Architecture Preview :
![image](https://user-images.githubusercontent.com/84160502/205516996-c05d575f-aaf2-494e-be84-f7bfeb8c522e.png)


