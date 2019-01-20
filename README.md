# Doodle Camera

This project is developed by Awesome Name team in less than 4 days during the Google AI ML WinterCamp Shanghai.

<img src="https://github.com/laylalaisy/Doodle_Camera--AwesomeName/raw/master/phone.png" width="200"/>

Using Doodle Camera, you can transfer your daily photos into the 'ugly but cute' doodle style, where the doodle images are drawn by real humans.

Video Demo [Here][video].

Online Demo [Here][demo]. (Server is Down)

Presentation Slides [Here][presentation].

Poster [Here][poster].

[video]: https://drive.google.com/file/d/1kOy6MtoV0b7BqVu6PDHgSA5XepzeqVLx/view
[demo]: http://165.227.22.95/index
[presentation]: https://docs.google.com/presentation/d/1j50bNXnlmWNMyljWRQDWrnEB-gHcQ4TBbbPqAnHsDx8/edit#slide=id.p1
[poster]: https://docs.google.com/presentation/d/1Cey5EeBEdEqRYktxKW8tOywhIxM0poXg36-HQU4ueUc/edit?usp=sharing

<br/>

## Technical Design

### Overview

<img src="https://github.com/laylalaisy/Doodle_Camera--AwesomeName/raw/master/overview.png" width="700" style="display:inline"/>

In our project, we have two data input. One is the photo uploaded by users, another is the pre-processed dataset of doodle images based on Google Quick Draw which has 340 classes of doodle images.

As shown in the graph, there are mainly three processes of our project. First, we use our classifier as a filter to find out good doodle images. Then, once users use upload a photo, we do object detection. Then, we use similar matching method to choose the most similar doodle image to replace the original object.

As mentioned in PPT , there are three key challenges in our project and I will simply introduce them. 

1. How to recognize objects in the image? (We use pretrained YOLO-v3 to do object detection)
2. How to select **recognizable** doodle images? (Because in the original dataset, a lot of images are quite abstract and even incomplete. )
3. How to choose **suitable** doodle images? (Suitable images not only require doodle and object belongs to the same class, but we also would like to find doodle image with similar shape and direction.)



### Training a image classifier

**Code related to training the classifier is under the cky branch.**

We firstly focus on training a classifier on the Google QuickDRAW Dataset, because if we have a robust classifier, we can gain an understanding of doodle images, which can help us filter images that are not recognizable, and help us select suitable doodle images. So I will firstly talk about how we train a classifier and how we achieve top 12% on the Kaggle Competition.

For an image classification task, we naturally think about using a convolutional network like ResNet or DenseNet. An essential problem is how we generate the images. The Google QuickDRAW dataset contains the information of how people draw the doodle image, specifically, the timestamps for every stroke of the drawing. To utilize the time data, we colorize images according to the timestamps of strokes. For example, the butterfly shown below, the person firstly draws its wings, which is marked in red, and then draws its wings, which is marked in black.

<img src="https://github.com/laylalaisy/Doodle_Camera--AwesomeName/raw/master/butterfly.png" width="200"/>

Here are the properties and final performance of the models that we trained during the winter camp. 

<img src="https://github.com/laylalaisy/Doodle_Camera--AwesomeName/raw/master/chart.png" width="600"/>

For data augmentation, we not only do data augmentation during training for better generalization ability, we also use the technic named Test-Time Augmentation to help the deciding process.

Afterwards, we ensemble all these models and checkpoints using a blending algorithm. Notably, there is a simple but effective trick given the fact that the testset is balanced, which means that every class appears the same number of times. We assign different weights to the label predictions to generate more balanced predictions.

Our final ensembled model achieves top 12% of the Kaggle Competiation, but still 2.4 percent lower than the best model. If we have more time to train our models, the performance will be better.



### Selecting Recognizable and Suitable Images

To recall, we need to select recognizable images, and suitable images. 

To select recognizable images, we assume that the images that are easy to recognize for models are also somehow easy to recognize for humans, so we select top 100 doodle images according to the model's probability from each class and it seems to be working well. 

To select suitable images, we firstly store the feature maps of doodle images from the dataset, and once we get an object image after the object detection process, we use the Euclidean distance in feature space to find the most similar doodle image. One essential problem is that the image after object detection is from real world, where the style is totally different from the doodle images in the dataset. So before we feed the object image to the classifier, we use canny edge detection to first convert the real image to lines and edges, which is much similar to the doodle style. We also use a perceptual hash algorithm when the first solution can not find a suitable image.

</br>

## Acknowledgements

We thank Google for holding this AI ML Winter Camp, where Google provided comfortable workplace, cloud server, delicious lunch and teabreaks. 

Thanks to Google's awesome HR team, especially Manyi, Queenie, Irene, Helen and Brett for their great work. 

Thanks to Google's speakers and engineers for their in-depth lecture and strong support.

