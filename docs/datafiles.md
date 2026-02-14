The `microfading` package aims to facilitate the manipulation of microfading data. Consequently, the microfading data files play a central role in the package. Like in most analytical techniques, several microfading devices have been developed over the years. Each device creates its own specfic rawfiles that are usually very different from one device to another. To compare the results obtained with different devices, one solution is to convert each rawfile into a unique file format, which is the method that has been chosen in this package. 

This unique file format, called *interim* file, is simply an excel file with a specific inner structure to organize the data. Each excel file is composed of three sheets:

1. **info** : contains the metadata related to the measurements, the object, and the project
2. **CIELAB** : contains the colorimetric values
3. **spectra** : contains the spectral values

An exemple of an interim file can be found in the microfading package [(get_datasets() function)](https://g-patin.github.io/microfading/retrieve-test-datasets/).

The overall landscape of the package can be viewed as a tree or an hourglass shape. 


---

© 2024 Gauthier Patin. All rights reserved. | Last updated: 2024-09-20

