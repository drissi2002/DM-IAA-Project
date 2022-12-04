# Summary

This project aims to develop an <b> BTD </b> system from <b> MRI (Magnetic Resonance Imaging) scans </b>. This system will help clinicians to make an accurate diagnosis in order to improve the survival of patients. This system must provide a set of requirements and must be designed according to a deep learning architecture . It must explore different MRI scans and must make use of a set of well - defined metrics for its performance evaluation.

### Choice of Datasets
For the realization of our Project, we opted for data available on <b> "Kaggle" </b>. 
For our  Deep learning models we choose mainly “Brain MRI Images dataset”. <br><br/>
<b> Link <b/>: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
![téléchargement (2)](https://user-images.githubusercontent.com/84160502/205517138-1fd60817-d0af-46f7-a804-7fd621e3a24a.png)

## Choice of Models
In modern days, checking the huge number of MRI (magnetic resonance imaging) images and finding a brain tumor manually by a human is a very tedious and inaccurate task. It can affect the proper medical treatment of the patient. Again, it can be a hugely time-consuming task as it involves a huge number of image datasets. There is a good similarity between normal tissue and brain tumor cells in appearance, so segmentation of tumor regions becomes a difficult task to do. So there is an essentiality for a highly accurate automatic tumor detection method.
## Deep Learning Models : 
## CNN : 
  Deep learning is a handy and efficient method for image classification. Deep learning has been widely applied in various fields including medical imaging, because its application does not require the reliability of an expert in the related field, but requires the amount of data and diverse data to produce good classification results. <b> Convolutional Neural Network (CNN) </b>is the deep learning technique to perform image classification.
 Brain tumor detection using convolutional neural networks (CNN) CNN presents a segmentation-free method that eliminates the need for 

hand-crafted feature extractor techniques. For this reason, different CNN architectures have been proposed by several researchers.
Our CNN based model will help the doctors to detect brain tumors in MRI images accurately, so that the speed in treatment will increase a lot.
### Techniques that will be used To Improve The Accuracy of CNN :
While we develop the Convolutional Neural Networks (CNN) to classify the images, It is often observed that the model starts overfitting when we try to improve the accuracy. Very frustrating, Hence I list down the following techniques which would improve the model performance without overfitting the model on the training data.
1.Data normalization: The normalization of an image consists in dividing each of its pixel values by the maximum value that a pixel can take (255 for an 8-bit image).
2.Batch normalization: After each convolutional layer, we added a batch normalization layer, which normalizes the outputs of the previous layer. This is somewhat similar to data normalization, except it’s applied to the outputs of a layer, and the mean and standard deviation are learned parameters.
Architecture Preview :
![téléchargement](https://user-images.githubusercontent.com/84160502/205517128-c04eb818-4f77-494e-9070-c340e428595c.png)



## Vgg16 : 
  A convolutional neural network is also known as a <b>ConvNet</b>, which is a kind of artificial neural network. A convolutional neural network has an input layer, an output layer, and various hidden layers. <b>VGG16</b> is a type of CNN (Convolutional Neural Network) that is considered to be one of the best computer vision models to date. The creators of this model evaluated the networks and increased the depth using an architecture with very small (3 × 3) convolution filters, which showed a significant improvement on the prior-art configurations. They pushed the depth to <b>16–19 weight</b> layers making it approx — 138 trainable parameters
## Why Vgg16 ?
VGG16 is an object detection and classification algorithm which is able to classify 1000 images of 1000 different categories with <b>92.7% accuracy</b>. It is one of the popular algorithms for image classification and is easy to use with <b>transfer learning</b>.
### Techniques that will be used To Improve The Accuracy of Vgg16 :
Like CNN we will use the techniques for improving the accuracy + data augmentation
Architecture Preview :
![téléchargement (1)](https://user-images.githubusercontent.com/84160502/205517133-0b1ddc33-140e-4e00-a25c-51bba857c9ee.png)


