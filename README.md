# Mask-Detection-using-CNN
This project is centered around COVID-19,  the major health concern the world is facing currently. The  focus of reducing the spread of the virus is around face  masks, and the laws that many world institutions are  putting in place mandating their wear in public places. This  project proposes a method of image analysis that uses  Convolution Neural Networks (CNN) to classify images into  three separate categories: face masks, no face mask, or  incorrectly worn face mask. In this project, a tool is created  that anyone can use to gather information if the individuals  are wearing face masks, not wearing a face mask, or  wearing their face mask incorrectly quickly and easily.
## Dataset
The dataset that was used for this project was obtained from Kaggle. The dataset is named “Face Mask Detection” and it contains two sets of data: images and annotations. The dataset contains 853 images that belong to 3 different classes which include: with mask, without mask, and mask worn incorrectly. The images are in the .png file type and the annotations are in the .xml. These .xml annotation files contained information regarding the images that they were tied to. The images contained people that met the description of one of the three different classifications. Some of these images had more than one individual, and each were made as an object under the annotation file. In addition, each one of these objects, sometimes more than one per image, contained coordinates of a box that focused on the face and a label designating the with mask, without mask, and mask worn incorrectly labels. 
## Files
### Prepocessing.py
Preprocessing of the data was necessary so that the data of the images can be entered into the CNN. The data needed to be separated out and labeled individually to allow CNN to run flawlessly. First, the code generated a matrix that would hold the image title with all its corresponding objects and their labels of mask. The mask data of without mask, with mask, and mask worn incorrectly was labeled 0,1,2, respectively. To do this, boxes, labels, and image ID matrices were created with all the aforementioned information and appended into a target matrix with all their information under the correct images. Next, the boxes from the objects that were found in the images, were extracted as a .png file of only the boxes and saved into a separate file, box images. Once all the objects were gathered, they were then labeled with their correct original image number, box/object number, and the label corresponding to its mask classification. Lastly, the images were all resized to a 50x50x3 image so that the CNN could work properly, and the white box was removed from the images.
### Main.py
This preprocessing.py file was imported into the main file with the CNN model. The extracted images and extracted labels were called upon and organized into two different arrays: images data and labels. Then, the images data was split into 80% training and 20% test sets with a random state of 42. Lastly, the CNN model was applied on the training and test datasets outputting the accuracy after 25 epochs. An early stopping mechanism was put in place so if the accuracy of the model did not change for 3 iterations of epoch, then it would stop the epoch and output the evaluated accuracy and loss. This also outputted images of whether the person in the image did not wear a mask (0), wore a mask (1), or wore the mask incorrectly (2). The model was able to output the actual and predicted labels in the terminal that correspond to the respective image. The accuracy of the model varies between 90-92% with a mean and median of approximately 91% after 25 epochs. 
