
## AI - Astronomy Research

### Galaxy Classifier
#### How to run?

First, get the galaxy data for `sdss_images_1000.npy` and `sdss_labels_1000.npy` on [astroML's GitHub](https://github.com/astroML/astroML-data/tree/main/datasets).  
Then copy this data to the root folder of the project.

Execute `galaxy_classifier_tf2.py`. This will create the weight data for all the galaxies present on the files.

---

### Comet identifier
### How to run?

First, get the data from the NASA AWS bucket, the details can be found on [this website](https://www.topcoder.com/challenges/fcad16e0-9ca6-4510-8bd9-af3ed026b140?utm_source=community&utm_campaign=NASASoho&utm_medium=promotion).  
Then copy this data to the root folder of the project.

Execute comet_classifier.py. This will run the model for the classifier and give the overall accuracy based on the input images.
