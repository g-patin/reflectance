In this section, you will learn how to use the databases that come along with the `reflectance` package and inside which you have the possibility to store information. The main purpose of the databases is to connect the metadata with the analytical data. Additionally, the databases will allow you to conveniently reuse information and will also enable you to perform queries based on the information stored in the databases. For example, one could retrieve all the data obtained on paintings created by a given artist. 


This section implies that you have created the database files. If that is not the case, look at the section on how to [create empty databases](https://g-patin.github.io/reflectance/create-databases/). If you want to know whether databases were already created, use the following function:

```python
import reflectance as rf

# Check whether databases were created.
rf.is_DB()
```

<div class="output-area">
<pre>
True
All the databases were created and can be found in the following directory: /home/john/Documents/RS/databases
</pre>
</div>

&nbsp;

## **Description**

The databases ecosystem contains two main parts which are thoroughly described in the following sub-sections. On the one hand, there are data files inside which you will accumulate information about the measurements. On the other hand, one configure certain parameters of the packages by saving your preferences in the db_config.json file.

### **Data files**

The databases simply consists of csv and txt files (Table 1) inside which you will accumulate information related to the microfading measurements. The management of the data files lies around two main actions:

1. You want to learn how to **add / remove info**.
2. You want to learn how to **retrieve info**. 

The first things to do is to import the microfading package as showed in the code box above. You will access the databases functions as followed: `mf.<function name>`

```python
# for example, this will retrieve all the measuring devices that you registered. 
mf.get_institutions()
```

### **db_config file**

The db_config.json file contains information about the p


Every function name start with a verb followed by an underscore and a noun or an adjective providing more information about the action that will be performed. There are only 4 verbs:

- **Get**: retrieve information contains in the databases
- **Add**: add a new item in the databases (in future versions, I will implement *remove* functions)
- **Update**: modify the recorded parameters of the databases
- **Set**: set default values for some key parameters

&nbsp;

Table 1. Description of the database files.

| <div style="width:205px">Filename</div> | <div style="width:250px">Description</div>
| :--------| :---------
|DB_objects.csv | Contain information related to each microfaded object 
|DB_projects.csv | Contain information related to each project 
|institutions.txt | Contain information about the owners (institutions) of the objects 
|MFT_devices.txt | Contain information about the microfading devices
|objects_creators.txt | A list of persons (surname and name) that created the objects
|objects_supports.txt | A list of a materials used to create the objects 
|objects_techniques.txt | A list of techniques used to create the objects 
|objects_types.txt | A list of global categories to define the objects
|persons.txt | A list of persons (name, surname, and initials) performing the analyses
|white_standards.txt | Contain information related to white reference standards

&nbsp;

## **Get functions**

Table 2. Description of the *get* functions.

| <div style="width:205px">Function name</div> | <div style="width:250px">Description</div>
| :--------| :---------
|`get_path_DB()` | returns the absolute path where the databases are stored on your local computer 
|`get_DB()` | eturns the databases as pandas dataframes
|`get_colorimetric_info()` | returns the observer and illuminant default values
|`get creators()` | returns a list of names and surnames corresponding to the persons that created the microfaded objects
|`get_datasets()` | returns a list of microfading example data filenames that can be used to try the package
|`get_institutions()` | returns a list of institutions that own the microfaded objects
|`get_lighting_conditions()` | returns the default light exposure conditions default values 
|`get_objects()` | returns a list of the microfaded objects
|`get_persons()` | returns a list of persons that performed the microfading analyses
|`get_devices()` | returns a list of the registered microfading devices
|`get_white_standards()` | returns a list of the registered white standard references


## **Add functions**

Table 3. Description of the *add* functions.

| <div style="width:205px">Function name</div> | <div style="width:250px">Description</div>
| :--------| :---------
|`add_new_creator()` | record information about a new object creator (artist, institution, etc.) 
|`add_new_institution()` | record information about a new institution
|`add_new_object()` | record information about a new object
|`add_new_person()` | record information about a new person performing microfading measurements.
|`add_new_project()` | record information about a new project
|`add_devices()` | add a new microfading device
|`add_reference()` | add a new white standard reference 


## **Set functions**

Table 4. Description of the *set* functions.

| <div style="width:205px">Function name</div> | <div style="width:250px">Description</div>
| :--------| :---------
|`set_colorimetric_info()` | set the observer and illuminant default values
|`set_DB()`  | set the use of databases (True or False)
|`set_dose_unit()`  | set the unit of the light dose (MJ/m2, Mlxh, or sec)
|`set_folder_DB()` | set the path of the databases files should be stored
|`set_lighting_conditions()` | set the light exposure conditions default values

The values defined in the *set* functions are stored inside the *db_config.json* file which you can find on your local computer inside the *site-packages* folder of Anaconda. The folder *site-packages* is where the python libraries are installed on your computer (see below).

```python
import microfading as mf

# This command will indicate you where is the microfading package installed on your computer
mf

```
<div class="output-area">
<pre>
 < module 'microfading' from '/home/john/anaconda3/lib/python3.11/site-packages/microfading/__init__.py'>
</pre>
</div>

Whenever a function will need a parameter stored inside the db_config.json, it will retrieve it and use it (See code below for an example).

![Alt text](images/mf_get_Lab.png){: .img-Large align=left }
/// caption
Retrieve the Lab values.
///


## **Update functions**

Table 5. Description of the *update* functions.

| <div style="width:205px">Function name</div> | <div style="width:250px">Description</div>
| :--------| :---------
|`update_DB_objects()` | add new categories to the objects database
|`update_DB_projects()`| add new categories to the projects database



### Update object parameters


### Update project parameters

