# Embryo Classification of Gardner Score

This study aims to develop a series of machine learning model to predict the Gardner Score of an embryo. The Gardner Score is a value assigned to the three main regions of the embryo, meaning, the Expansion (size of the embryo and width of the Pellucid Zone), the Inner Cellular Mass (main cluster of cells), and Trophectoderm (cells which serve as energy supply for the embryo). We consider a database of 249 embryos in the blastocyst stage (5th-6th day after insemination) each with its corresponding labeled masks for each region, and its Gardner Score. Images are segmented using a U-net and three methods are developed to find which one makes the best classification:

1 - In the first method, handcrafted features are build from the masks and the superpositio of the mask and the embryo image. These features can be divided in textural-related features (using the gray level co-ocurrence matrix), geometric features (measuring the volume and perimeter of the diferent masks of each embryo), and topographical features (obtained for the polar unwrapping of the embryo, and building a signal of peaks which are a correspondence of the cells inside the Inter Cellular Mass and Trophoectodherm. The image of each region is classified by a series of different models, and the one with best accuracy is chosen. Between these there are Decision Trees, Random Forest, SVM, Naive Bayes, a Multi-Layered Perceptron and XGBoost. We report the results of the mean value of a 10-fold cross validation, and the features importance derived from the different classifiers.

2 - The second method uses image embeddings obtained by convolutional neural networks (VGG, Resnet50, InceptionV3) and two different-sized autoencoders. The features are extracted from specific layers of the mentioned models, followed by a classification stage by the same models mentioned in the first method.

3 - The third method uses an end-to-end multiple neural network to perform the classification directly, using the same CNN networks as the second method. 

## First method

(See notebooks/segmented_embryo_classification)

This first method is composed by a segmentation, followed by three feature extraction methods stages. The segmentation is carried out by the database provided by Dr. Parvaneh Saeedi from Simon Fraser University (check article [https://ieeexplore.ieee.org/document/8803139]).

One classification is executed with an array with all the features, and another one with a previously dimensionality reduced process (PCA).

Also, classification is carried out for each region (for example: ICM) with features of all regions (ICM, TE, and ZP), and another classification is carried out with the features of the chosen region (only ICM's features, if ICM region is chosen).

### Textural feature extraction

It follows the depiction of the next schema:

<img width="273" alt="Captura de pantalla 2023-11-26 a la(s) 16 01 00" src="https://github.com/JuanDiegoYoung/embryo_classification/assets/81267941/8ec09bd2-d86e-4a6c-820e-23cae7f84d82">

### Geometric feature extraction

The geomeric features consist of the volume and perimeter of the three different zones, and all possible combinations of the zones (see image). Also, as extra features, the quotientes between volume and perimeters of relevant zones (ex: volume of ICM / volume of the full blast, perimeter of ICM / volume of ICM, volume of ZP / volume of full blast, etc), and the radius of the full blast and pellucid zone is calculated. The inverse of all these features is added as well.

<img width="375" alt="Captura de pantalla 2023-11-26 a la(s) 16 14 28" src="https://github.com/JuanDiegoYoung/embryo_classification/assets/81267941/532974c8-6df0-4dbf-9011-73126936bb79">

<img width="375" alt="Captura de pantalla 2023-11-26 a la(s) 16 14 45" src="https://github.com/JuanDiegoYoung/embryo_classification/assets/81267941/3688e7f9-605e-4d9b-a3b3-d096da8eeef0">

<img width="368" alt="Captura de pantalla 2023-11-26 a la(s) 16 14 59" src="https://github.com/JuanDiegoYoung/embryo_classification/assets/81267941/c3bb3797-800b-40c4-ae80-afc32e43b2db">

### Topographical feature extraction

These features are the result of unwrapping the masks from the center of the embryo in a polar fashion. The result is a signal for the ICM, another one for the TE, and another for the ZP. From there different measures can be obtained: maximum width, maximum height, also the minimum, the mean, and the standard deviation. Also, a cellular analysis is performed: it consist in the detection of peaks and valleys in the ICM and TE signals. Each cells consist in a peak and its two adjacent valleys, and then some features are extracted: height, width, parabolic coefficient, and extendend width (approximating the cell as a parabole). Lastly, the number of cells its calculated.

<img width="281" alt="Captura de pantalla 2023-11-26 a la(s) 16 13 47" src="https://github.com/JuanDiegoYoung/embryo_classification/assets/81267941/21668382-8cb8-4232-ad07-bfe733dff3e8">

## Second method

(See notebooks/embedded_embryo_classification)

This method uses different networks (VGG, Resnet50 and InceptionV3) removing the last fully connected layers, keeping the convolutional architecture, and saving the last output as features.

Again, the classification branches with and without PCA, and branches again with all features, and the region specific features. It branches one more time using segmented images, and not segmented images.

<img width="712" alt="Captura de pantalla 2023-11-26 a la(s) 16 28 52" src="https://github.com/JuanDiegoYoung/embryo_classification/assets/81267941/4ee4f8e6-5cf1-4450-9225-57410170fe9d">

In this example there is a VGG architecture, and the final fully connected layers (purple ones) are removed.

## Third method

(See notebooks/end_to_end_embryo_classification)

This last method is the most straigthforward: using the embryo images to train the different networks, and make a classification for the quality of each region.

<img width="657" alt="Captura de pantalla 2023-11-26 a la(s) 16 32 10" src="https://github.com/JuanDiegoYoung/embryo_classification/assets/81267941/7436bcf0-c928-4591-8a73-5e3932564613">

This method also branches using the segmented images, and the not segmented images.

## Remaining things to do

 - Correct a bug where the fully connected layers are not removed in the embedding process
 - Add the segmented embedding extraction
 - Add the segmented images for the end to end classification

 - Add all single-region classifications

 - Run all experiments and choose the best result

 - Carry out uncertainty estimation

 - Realize unitary tests

 - Create src files

 - Create another repository with the database provided by the CEM (Centro de Esterilidad de Montevideo), and execute the classification of the Gardnger score, prediction of probability of pregnancy, and uncertainty estimation
