# coding: utf-8
# Author: Gauthier Patin
# Licence: GNU GPL v3.0

import os
import pandas as pd
import numpy as np
import colour
import json
from typing import Optional, Union, List, Tuple
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt
from uncertainties import ufloat, ufloat_fromstr, unumpy
from pathlib import Path
import itertools
import importlib.resources as pkg_resources
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import ipywidgets as ipw
from ipywidgets import *
from IPython.display import display, clear_output

# underlying modules of the  microfading package
from . import plotting
from . import databases
from . import process_rawfiles

####### DEFINE GENERAL PARAMETERS #######

D65 = colour.CCS_ILLUMINANTS["cie_10_1964"]["D65"]

labels_eq = {
    'dE76': r'$\Delta E^*_{ab}$',
    'dE94': r'$\Delta E^*_{94}$',
    'dE00': r'$\Delta E^*_{00}$',         
    'dR_vis': r'$\Delta R_{vis}$',
    'dL*' : r'$\Delta L^*$',
    'da*' : r'$\Delta a^*$',
    'db*' : r'$\Delta b^*$',
    'dC*' : r'$\Delta C^*$',
    'dh' : r'$\Delta h$',    
    'L*' : r'$L^*$',
    'a*' : r'$a^*$',
    'b*' : r'$b^*$',
    'C*' : r'$C^*$',
    'h' : r'$h$',          
}


#### DATABASES RELATED FUNCTIONS ####


def DB():
    "Check whether the databases files were created."

    # instantiate a DB class object
    DB = databases.DB()

    if DB.folder_db.stem == "folder_path":
        print('The databases files have not been created. To create them, run the command "mf.create_DB(<folder_path_of_your_choice>)".')
        return
    
    else:
        db_files = ['DB_projects.csv', 'DB_objects.csv','institutions.txt', 'persons.txt','object_types.txt', 'object_techniques.txt', 'object_supports.txt', 'object_creators.txt']
        
        if all(list(map(os.path.isfile, [str(Path(DB.folder_db)/x) for x in db_files]))):
            print(f'All the databases were created and can be found in the following directory: {DB.folder_db}')

        else:
            print('The databases files were created, but one or several files are currently missing.')
            print(f'The files should be located in the following directory: {DB.folder_db}')


def get_datasets(device:Optional[str] = 'KM', rawfiles:Optional[bool] = False, stdev:Optional[bool] = False):
    """Retrieve exemples of dataset files. These files are meant to give the users the possibility to test the MFT class and its functions.  

    Parameters
    ----------
    device : Optional[str], optional
        Device that has been used to obtain the files, by default 'KM'
        One can choose a single option among the following choices: 'Avt', 'KM', 'OO', 'Tidas' 
        'Avt' corresponds to the Avantes spectrometer ....
        'KM' corresponds Konica Minolta photospectrometer CM-2600d.
        'OO' corresponds to the Ocean Optics spectrometer ....
        'Tidas' corresponds to the .....

    rawfiles : Optional[bool], optional
        Whether to get rawdata files, by default False      

    stdev : Optional[bool], optional
        Whether to have measurements wiht standard deviation values, by default False
        It only works if the rawfiles parameters is set to 'False'. The rawfiles do not have standard deviation values.

    Returns
    -------
    list
        It returns a list of strings, where each string corresponds the absolute path of a txt file. Subsequently, one can use the list as input for the RS class. 
    """

    # Whether to select files with standard deviation values
    if stdev:
        if device == 'sMFT':
            data_files = [
                '2024-144_MF.BWS0024.G02_avg_BW1_model_2024-08-02_MFT1.xlsx',
                '2024-144_MF.BWS0025.G02_avg_BW2_model_2024-08-02_MFT1.xlsx',
                '2024-144_MF.BWS0026.G02_avg_BW3_model_2024-08-02_MFT1.xlsx',                
            ]

        elif device == 'fotonowy':
            data_files = [
                '2024-144_MF.BWS0024.G01_avg_BW1_model_2024-07-30_MFT2.xlsx',
                '2024-144_MF.BWS0025.G01_avg_BW2_model_2024-08-02_MFT2.xlsx',
                '2024-144_MF.BWS0026.G01_avg_BW3_model_2024-08-07_MFT2.xlsx',
                '2024-144_MF.dayflower4.G01_avg_0h_model_2024-07-30_MFT2.xlsx',
                '2024-144_MF.indigo3.G01_avg_0h_model_2024-08-02_MFT2.xlsx',
            ]
        
    else:
        if device == 'sMFT':
            data_files = [
                '2024-144_MF.BWS0026.04_G02_BW3_model_2024-08-02_MFT1.xlsx',
                '2024-144_MF.BWS0025.04_G02_BW2_model_2024-08-02_MFT1.xlsx',
                '2024-144_MF.BWS0024.04_G02_BW1_model_2024-08-02_MFT1.xlsx',
                '2024-144_MF.yellowwood.01_G01_yellow_model_2024-08-01_MFT1.xlsx',
                '2024-144_MF.vermillon.01_G01_red_model_2024-07-31_MFT1.xlsx',
            ]

        elif device == 'fotonowy':
            data_files = [
                '2024-144_MF.BWS0024.01_G01_BW1_model_2024-07-30_MFT2.xlsx',
                '2024-144_MF.BWS0025.01_G01_BW2_model_2024-08-02_MFT2.xlsx',
                '2024-144_MF.BWS0026.01_G01_BW3_model_2024-08-07_MFT2.xlsx',
                '2024-144_MF.vermillon3.01_G01_0h_sample_2024-07-31_MFT2.xlsx',
                '2024-144_MF.yellowwood4.01_G01_0h_model_2024-08-01_MFT2.xlsx',
            ]
 
    # Whether to select rawfiles according to a choosen device
    if rawfiles:
        if device == 'sMFT':
            data_files = [
                '2024-144_BWS0024_04_G02_BW1_c01_000001.txt',
                '2024-144_yellowwood_01_G01_yellow_c01_000001.txt',
            ]

        elif device == 'fotonowy':
            data_files = [
                '2024-8200 P-001 G01 uncleaned_01-spect_convert.txt',
                '2024-8200 P-001 G01 uncleaned_01-spect.txt',
                '2024-8200 P-001 G01 uncleaned_01.txt',
                '2024-8200 P-001 G01 uncleaned_01.rfc',
                '2024-144 BWS0024 G01 BW1_01-spect_convert.txt',
                '2024-144 BWS0024 G01 BW1_01-spect.txt',
                '2024-144 BWS0024 G01 BW1_01.txt',
                '2024-144 BWS0024 G01 BW1_01.rfc',
            ]    

    # Get the paths to the data files within the package
    file_paths = []
    for file_name in data_files:
        
        with pkg_resources.path('microfading.datasets', file_name) as data_file:
             file_paths.append(data_file)


    return file_paths
   

def create_DB(folder:str):
    """Initiate the creation of databases.

    Parameters
    ----------
    folder : str
        Absolute path of the folder where the databases will be stored.

    Returns
    -------
        It creates two empty databases as .csv file (DB_projects.csv and DB_objects.csv), as well as six .txt files in the folder given as input.
    """

    # instantiate a DB class object and then use the create_db function
    DB = databases.DB()
    DB.create_db(folder_path=folder)


def get_objects(project_id:Union[str,list] = 'all'):
    """Retrieve object Id numbers according to project Id.

    Parameters
    ----------
    project_id : Union[str,list], optional
        The Id number of projects for which the objects should be retrieved, by default 'all'
        You can enter a string if there is only a single project, or a list of strings if there are several projects.
        When 'all', it returns all the objects registered in the DB_objects.csv

    Returns
    -------
    a dictionary
        It returns a dictionary where the keys are the project Id number and the values are the object Id number given inside a list. If there is only one project id, then it directly returns the list of objects.
    """
    # instantiate a DB class object
    DB = databases.DB() 

    db_objects = DB.get_db(db='objects')
    projects_objects = {}

    if project_id == 'all':
        pass
    elif isinstance(project_id, str):
        project_id = [project_id]
        db_objects = db_objects[db_objects['project_id'].isin(project_id)]
    elif isinstance(project_id, list):
        db_objects = db_objects[db_objects['project_id'].isin(project_id)]
    
    project_ids = sorted(set(db_objects['project_id'].values))
    for Id in project_ids:
        df_project = db_objects[db_objects['project_id'] == Id]
        objects = sorted(set(df_project['object_id'].values))
        projects_objects[Id] = objects   

    if len(project_id) == 1:
        projects_objects = projects_objects[project_id[0]]

    return projects_objects


def get_path_DB():
    """Retrieve the absolute path of the folder where the databases are located.

    Returns
    -------
    string or None
        If dabases have been created, it will return the absolute path as a string. Otherwise, it will only print a statement indicating no databases were found.    
    """
    
    # instantiate a DB class object
    DB = databases.DB()   

    
    if DB.folder_db.stem == "folder_path":
        print('Databases have not been created or have been deleted. Please, create databases by running the function "create_DB" from the reflectance package.')
        return None
    
    else:    
        if 'DB_projects.csv' in os.listdir(DB.folder_db) and 'DB_objects.csv' in os.listdir(DB.folder_db):

            print(f'DB_projects.csv and DB_objects.csv files can be found in the following folder: {DB.folder_db}')    
            return DB.folder_db       

        else:
            print('Databases have not been created or have been deleted. Please, create databases by running the function "create_DB" from the reflectance package.')
            return None


def get_creators():
    """Retrieve the list of object creators that have been registered in the object_creators.txt file

    Returns
    -------
    pandas dataframe
        It returns the list of creators inside a pandas dataframe with two columns: 'surname', 'name'
    """
    
    DB = databases.DB()
    return DB.get_creators()


def get_DB(db:Optional[str] = 'all'):
    """Retrieve the databases

    Parameters
    ----------
    db : Optional[str], optional
        Choose which databases to retrieve, by default 'all'
        When 'projects', it returns the DB_projects.csv file
        When 'objects', it returns the DB_objects.csv file
        When 'all', it returns both file as a tuple

    Returns
    -------
    pandas dataframe or tuple
        It returns the databases as a pandas dataframe or a tuple if both dataframes are being asked.
    """

    # instantiate a DB class object and return the databases if they exist.
    DB = databases.DB()
    return DB.get_db(db=db)


def get_DB_config():

    DB = databases.DB()
    return DB.get_db_config()


def get_institutions():
    """Retrieve the list of institutions that have been registered in the institutions.txt file. These institutions are the owner of the objects on which the reflectance analyses were performed.

    Returns
    -------
    pandas dataframe
        It returns the list of institutions inside a pandas dataframe with two columns: 'name', 'acronym'
    """

    DB = databases.DB()
    return DB.get_institutions()


def get_persons():
    """Retrieve the list of persons that have been registered in the persons.txt file. These persons are the one related to the creation of the measurement files.

    Returns
    -------
    pandas dataframe
        It returns the list of persons inside a pandas dataframe with three columns: 'name', 'surname', 'initials'
    """

    DB = databases.DB()
    return DB.get_persons()


def get_devices():
    """Retrieve the list of microfading devices that have been registered in the devices.txt file.

    Returns
    -------
    pandas dataframe
        It returns the list of devices inside a pandas dataframe with four columns: 'Id', 'name', 'description', 'process_function'
    """

    DB = databases.DB()    
    return DB.get_devices()


def get_colorimetry_info():
    """Retrieve the colorimetric information (observer and illuminant) recorded in the db_config.json file of the reflectance package.

    Returns
    -------
    pandas dataframe or string
        It returns the information inside a dataframe if they have been recorded.
    """

    DB = databases.DB()    
    return DB.get_colorimetry_info()



def get_white_references():
    """Retrieve the list of white standard references that have been registered in the white_references.txt file.

    Returns
    -------
    pandas dataframe
        It returns the list of references inside a pandas dataframe with two columns: 'Id', 'description'
    """

    DB = databases.DB()    
    return DB.get_white_references()


def add_new_creator():
    """Record a new object creator inside the object_creators.txt file.
    """

    DB = databases.DB()    
    return DB.add_new_creator()


def add_new_institution():
    """Record a new institution inside the institutions.txt file.
    """

    DB = databases.DB()    
    return DB.add_new_institution()


def add_new_project():
    """Record the information about a new project inside the DB_projects.csv file.
    """

    DB = databases.DB()    
    return DB.add_new_project()


def add_new_object():
    """Record the information about a new object inside the DB_objects.csv file.
    """

    DB = databases.DB()    
    return DB.add_new_object()


def add_new_person():
    """Record the information of a new person inside the pesons.txt file.
    """

    DB = databases.DB()    
    return DB.add_new_person()


def update_DB_objects(new: str, old:Optional[str] = None):
    """Add a new column or modify an existing one in the DB_objects.csv file.

    Parameters
    ----------
    new : str
        value of the new column

    old : Optional[str], optional
        value of the old column to be replaced, by default None        
    """    

    DB = databases.DB()
    DB.update_db_objects(new=new, old=old) 


def update_DB_projects(new: str, old:Optional[str] = None):
    """Add a new column or modify an existing one in the DB_projects.csv file.

    Parameters
    ----------
    new : str
        value of the new column
        
    old : Optional[str], optional
        value of the old column to be replaced, by default None        
    """

    DB = databases.DB()
    DB.update_db_projects(new=new, old=old) 


def add_devices():
    """
    Register measurement devices.
    """

    DB = databases.DB()

    style = {"description_width": "initial"}
    RS_process_functions = ['MFT_fotonowy']
    
    # Define ipython widgets
    nb_widget = ipw.Text(        
        value='',
        placeholder='Id number of the device or just a number',
        description='Id',
        style=style,               
    )

    name_widget = ipw.Text(        
        value='',
        placeholder='One word description',
        description='Name',
        style=style,               
    )

    description_widget = ipw.Text(        
        value='',
        placeholder='One line device description',
        description='Description',
        style=style,               
    )

    RS_function_widget = ipw.Dropdown(        
        value=RS_process_functions[0],
        options=RS_process_functions,
        description='Process function name',
        style=style,               
    )

    registering = ipw.Button(
        description='Register the device',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me', 
        style=style,           
    )      

    button_record_output = ipw.Output()

    def button_record_pressed(b):
        """
        Save the device info in the devices.txt file.
        """

        button_record_output.clear_output(wait=True)

        device_id = nb_widget.value.strip()
        device_name = name_widget.value.strip()
        device_description = description_widget.value.strip()
        RS_function = RS_function_widget.value.strip()

        df_devices = pd.read_csv(f'{DB.folder_db}/devices.txt')
        device_Ids = df_devices['Id'].values

        if device_id not in device_Ids:

            df_devices = pd.concat([df_devices, pd.DataFrame(data=[device_id,device_name,device_description,RS_function], index=['Id','name','description','process_function']).T])
            df_devices.to_csv(f'{DB.folder_db}/devices.txt', index=False)

            with button_record_output:
                print(f'Device registered in the following file: {DB.folder_db}/devices.txt')

        else:
            with button_record_output:
                print('The Id number you entered is already assigned to another device. Please choose another Id number.')
            

    registering.on_click(button_record_pressed)

    display(ipw.VBox([nb_widget,name_widget,description_widget,RS_function_widget]))
    display(ipw.HBox([registering,button_record_output]))


def add_references():
    """
    Register white standard references.
    """

    DB = databases.DB()
    style = {"description_width": "initial"}

    # Define ipython widgets
    nb_widget = ipw.Text(        
        value='',
        placeholder='Id number of the item or just a number',
        description='Id',               
    )

    description_widget = ipw.Text(        
        value='',
        placeholder='One line device description',
        description='Description',               
    )   

    registering = ipw.Button(
        description='Register the item',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me', 
        style=style,           
    )      

    button_record_output = ipw.Output()

    def button_record_pressed(b):
        """
        Save the reference info in the white_references.txt file.
        """

        button_record_output.clear_output(wait=True)

        reference_id = nb_widget.value.strip()        
        reference_description = description_widget.value.strip()
        

        df_references = pd.read_csv(f'{DB.folder_db}/white_references.txt')
        reference_Ids = df_references['Id'].values

        if reference_id not in reference_Ids:

            df_references = pd.concat([df_references, pd.DataFrame(data=[reference_id,reference_description], index=['Id','description']).T])
            df_references.to_csv(f'{DB.folder_db}/white_references.txt', index=False)

            with button_record_output:
                print(f'Item registered in the following file: {DB.folder_db}/white_references.txt')

        else:
            with button_record_output:
                print('The Id number you entered is already assigned to another white reference. Please choose another Id number.')
            

    registering.on_click(button_record_pressed) 

    display(ipw.VBox([nb_widget,description_widget]))
    display(ipw.HBox([registering,button_record_output]))



def process_rawdata(files: list, device: str, filenaming:Optional[str] = 'none', folder:Optional[str] = '.', db:Optional[bool] = False, comment:Optional[str] = '', wanted_wl:Optional[tuple] = 'default', authors:Optional[str] = 'XX', white_reference:Optional[str] = 'default', observer:Optional[str] = 'default', illuminant:Optional[str] = 'default', delete_files:Optional[bool] = True, return_filename:Optional[bool] = True):
    """Process the reflectance spectroscopy raw files created by the software that performed the analysis. 

    Parameters
    ----------
    files : list
        A list of string that corresponds to the absolute path of the raw files.
    
    device : str
        Define the device that has been used to generate the raw files ('KM', 'OO', 'Tidas').
    
    filenaming : [str | list], optional
        Define the filename of the output excel file, by default 'none' 
        When 'none', it uses the filename of the raw files
        When 'auto', it creates a filename based on the info provided by the databases
        A list of parameters provided in the info sheet of the excel output can be used to create a filename   

    folder : str, optional
        Folder where the final data files should be saved, by default '.'
    
    db : bool, optional
        Whether to make use of the databases, by default False
        When True, it will populate the info sheet in the interim file (the output excel file) with the data found in the databases.
        Make sure that the databases were created and that the information about about the project and the objects were recorded.
    
    comment : str, optional
        Whether to include a comment in the final excel file, by default ''
    
    authors : str, optional
        Initials of the persons that performed and processed the measurements, by default 'XX' (unknown).
        Make sure that you registered the persons in the persons.txt file (see function 'add_new_person').
        If there are several persons, use a dash to connect the initials (e.g: 'JD-MG-OL').       

    observer : str, optional
        Reference CIE *observer* in degree ('10deg' or '2deg'). by default 'default'.
        When 'default', it fetches the observer value recorded in the db_config.json file of the package. If no value has been recorded, then it sets the observer value to '10deg'. 

    illuminant : (str, optional)  
        Reference CIE *illuminant*. It can be any value of the following list: ['A', 'B', 'C', 'D50', 'D55', 'D60', 'D65', 'D75', 'E', 'FL1', 'FL2', 'FL3', 'FL4', 'FL5', 'FL6', 'FL7', 'FL8', 'FL9', 'FL10', 'FL11', 'FL12', 'FL3.1', 'FL3.2', 'FL3.3', 'FL3.4', 'FL3.5', 'FL3.6', 'FL3.7', 'FL3.8', 'FL3.9', 'FL3.10', 'FL3.11', 'FL3.12', 'FL3.13', 'FL3.14', 'FL3.15', 'HP1', 'HP2', 'HP3', 'HP4', 'HP5', 'LED-B1', 'LED-B2', 'LED-B3', 'LED-B4', 'LED-B5', 'LED-BH1', 'LED-RGB1', 'LED-V1', 'LED-V2', 'ID65', 'ID50']. by default 'default'.
        When 'default', it fetches the illuminant value recorded in the db_config.json file of the package. If no value has been recorded, then it sets the illuminant value to 'D65'.      

    delete_files : bool, optional
        Whether to delete the raw files

    Returns
    -------
    Excel file
        It returns an excel file composed of three tabs (info, CIELAB, spectra).
    """

    return process_rawfiles.RS_Tidas(files=files, filenaming=filenaming, folder=folder, db=db, comment=comment, device_ID=device, wanted_wl=wanted_wl, authors=authors, white_reference=white_reference, observer=observer, illuminant=illuminant, delete_files=delete_files, return_filename=return_filename)




def set_colorimetry_info():
    """Record the colorimetric information (observer and illuminant) in the db_config.json file of the reflectance package.
    """

    DB = databases.DB()
    return DB.set_colorimetry_info()


def set_config_info():

    DB = databases.DB()
    return DB.set_config_info()


def set_comments_info():
    """Record the comments order in the db_config.json file of the reflectance package. 
    It is only relevant if the software of the device has a "comment" entry where you can insert information.
    """

    DB = databases.DB()
    return DB.set_comment_info()


def set_devices_info(device_ID):
    """Record devices-related information in the db_config.json file.
    """

    DB = databases.DB()
    return DB.set_devices_info(device_ID)


def set_devices_keys():

    DB = databases.DB()
    return DB.set_devices_keys()


def set_fibers_info():
    """Record the fiber optics information in the db_config.json file.
    """

    DB = databases.DB()
    return DB.set_fibers_info()


#### REFLECTANCE CLASS ####         

class RS(object):

    def __init__(self, files:list, ) -> None:
        """Instantiate a Reflectance Spectroscopy (RS) class object in order to manipulate and visualize reflectance data.

        Parameters
        ----------
        files : list
            A list of string, where each string corresponds to the absolute path of text or csv file that contains the data and metadata of a single measurement. The content of the file requires a specific structure, for which an example can be found in "datasets" folder of the reflectance package folder (Use the get_datasets function to retrieve the precise location of such example files). If the file structure is not respected, the script will not be able to properly read the file and access its content.
        
        """
        self.files = files           

       
    def __repr__(self) -> str:
        return f'Reflectance data class - Number of files = {len(self.files)}'
       
    
    def get_spectra(self, wl_range:Union[int, float, list, tuple] = 'all', spectral_mode:Optional[str] = 'rfl', smoothing:Optional[list] = [1,0]):
        """Retrieve the reflectance spectra related to the input files.

        Parameters
        ----------
        wl_range : Union[int, float, list, tuple], optional
            Select the wavelengths for which the spectral values should be given with a two-values tuple corresponding to the lowest and highest wavelength values, by default 'all'
            When 'all', it will returned all the available wavelengths contained in the datasets.
            A single wavelength value (an integer or a float number) can be entered.
            A list of specific wavelength values as integer or float can also be entered.
            A tuple of two or three values (min, max, step) will take the range values between these two first values. By default the step is equal to 1.
       
        spectral_mode : string, optional
            When 'rfl', it returns the reflectance spectra
            When 'abs', it returns the absorption spectra using the following equation: A = -log(R)

        smoothing : list of two integers, optional
            Whether to smooth the reflectance data using the Savitzky-Golay filter from the Scipy package, by default [1,0]
            The first integer corresponds to the window length and should be less than or equal to the size of a reflectance spectrum. The second integer corresponds to the polyorder parameter which is used to fit the samples. The polyorder value must be less than the value of the window length.


        Returns
        -------
        A list of pandas dataframes
            It returns a list of pandas dataframes where the columns correspond to the dose values and the rows correspond to the wavelengths.
        """

        data_sp = []
        files = self.read_files(sheets=['spectra']) 
          

        for file in files:
            df_sp = file[0]            

            # whether to compute the absorption spectra
            if spectral_mode == 'abs':
                df_sp = np.log(df_sp) * (-1)
                                       

            # Set the wavelengths
            if isinstance(wl_range, tuple):
                if len(wl_range) == 2:
                    wl_range = (wl_range[0],wl_range[1],1)
                
                wavelengths = np.arange(wl_range[0], wl_range[1], wl_range[2])                               

            elif isinstance(wl_range, list):
                wavelengths = wl_range                               

            elif isinstance(wl_range, int):
                wl_range = [wl_range]
                wavelengths = wl_range  

            else:
                wavelengths = df_sp.index          
                
            df_sp = df_sp.loc[wavelengths]

            
            # Smooth the data            
            df_sp = pd.DataFrame(savgol_filter(df_sp.T.values, window_length=smoothing[0], polyorder=smoothing[1]).T, columns=df_sp.columns, index=wavelengths)
            
            
            # append the spectral data
            data_sp.append(df_sp) 

        return data_sp
    
    
    def get_cielab(self, coordinates:Optional[list] = 'all', index:Optional[bool] = True):
        """Retrieve the colourimetric values.

        Parameters
        ----------
        coordinates : Optional[list], optional
            Select one or multiple colourimetric coordinates from the following list: ['L*', 'a*','b*', 'C*', 'h', 'x', 'y'], by default 'all'       

        index : Optional[bool], optional
            Whether to set the index of the returned dataframes, by default False

        Returns
        -------
        A list of pandas dataframes
            It returns the values of the wanted colour coordinates inside dataframes where each coordinate corresponds to a column.
        """
                
                    
        # Retrieve the data        
        cielab_data = self.read_files(sheets=['CIELAB'])
        cielab_data = [x[0] for x in cielab_data]


        index_data = [x.set_index(x.columns[0]) for x in cielab_data]
        if coordinates == 'all':
            coordinates = ['L*', 'a*','b*', 'C*', 'h', 'x', 'y']

        wanted_data = [x.loc[coordinates] for x in index_data]

        if index == False:
            wanted_data = [x.reset_index() for x in wanted_data]
        
        return wanted_data

        # Create an empty list with all the colorimetric data
        all_data = []
              
        # Compute the delta LabCh values and add the data into the list all_data  
        for data in cielab_data:

            # for data with std values
            if sorted(set(data.columns.get_level_values(1))) == ['mean', 'std']:
                data_dLabCh = delta_coord = [unumpy.uarray(d[coord, 'mean'], d[coord, 'std']) - unumpy.uarray(d[coord, 'mean'], d[coord, 'std'])[0] for coord in ['L*', 'a*', 'b*', 'C*', 'h'] for d in [data]]

                delta_means = [unumpy.nominal_values(x) for x in delta_coord]
                delta_stds = [unumpy.std_devs(x) for x in delta_coord]

                delta_coord_mean = [(f'd{coord}', 'mean') for coord in ['L*', 'a*', 'b*', 'C*', 'h']]
                delta_coord_std = [(f'd{coord}', 'std') for coord in ['L*', 'a*', 'b*', 'C*', 'h']]

                for coord_mean,delta_mean,coord_std,delta_std in zip(delta_coord_mean,delta_means, delta_coord_std,delta_stds):                    
                    data[coord_mean] = delta_mean
                    data[coord_std] = delta_std

                    all_data.append(data)          
                
            # for data without std values
            else:
                data_LabCh = data[['L*','a*','b*','C*','h']]
                data_dLabCh = data_LabCh - data_LabCh.iloc[0,:]
                data_dLabCh = data_dLabCh.rename(columns={'L*': 'dL*', 'a*': 'da*' ,'b*': 'db*','C*': 'dC*','h': 'dh'}, level=0)
                all_data.append(pd.concat([data,data_dLabCh], axis=1))

                            

        # Whether to set the index
        if index:
            all_data = [x.set_index(x.columns[0]) for x in all_data]
        
        return all_data       
           
   
    def read_files(self, sheets:Optional[list] = ['info', 'CIELAB', 'spectra']):
        """Read the data files given as argument when defining the instance of the MFT class.

        Parameters
        ----------
        sheets : Optional[list], optional
            Name of the excel sheets to be selected, by default ['info', 'CIELAB', 'spectra']

        Returns
        -------
        A list of list of pandas dataframes
            The content of each input data file is returned as a list pandas dataframes (3 dataframes maximum, one dataframe per sheet). Ultimately, the function returns a list of list, so that when there are several input data files, each list - related a single file - corresponds to a single element of a list.            
        """
        
        files = []        
                
        for file in self.files:
            
            df_info = pd.read_excel(file, sheet_name='info')
            df_sp = pd.read_excel(file, sheet_name='spectra', header=[0,1], index_col=0)
            df_cl = pd.read_excel(file, sheet_name='CIELAB', header=[0,1])                      


            if sheets == ['info', 'CIELAB', 'spectra']:
                files.append([df_info, df_cl, df_sp])

            elif sheets == ['info']:
                files.append([df_info])

            elif sheets == ['CIELAB']:
                files.append([df_cl])

            elif sheets == ['spectra']:
                files.append([df_sp])

            elif sheets == ['spectra', 'CIELAB']:
                files.append([df_sp, df_cl])

            elif sheets == ['CIELAB','spectra']:
                files.append([df_cl, df_sp])

            elif sheets == ['info','CIELAB']:
                files.append([df_info, df_cl])

            elif sheets == ['info','spectra']:
                files.append([df_info, df_sp])

        return files
     

    def get_data(self, data:Union[str, list] = 'all', xarray:Optional[bool] = False):
        """Retrieve the microfading data.

        Parameters
        ----------
        data : str|list, optional
            Possibility to select the type of data, by default 'all'.
            When 'all', it returns all the data (spectral and colorimetric).
            When 'sp', it only returns the spectral data.
            When 'cl', it only returns the colorimetric data.  
            When 'Lab', it returns the CIE L*a*b* values.
            A list of strings can be entered to select specific colourimetric data among the following: ['dE76,'dE00','dR_vis', 'L*', 'a*', 'b*', 'C*', 'h'].

        xarray : bool, optional
            When True, the data are returned as an xarray.Dataset object, else as pandas dataframe object, by default False.

        Returns
        -------
        It returns a list of pandas dataframes or xarray.Dataset objects
        """

        all_files = self.read_files(sheets=['spectra','CIELAB'])
        all_data = []
        data_sp = [] 
        data_cl = [] 

        for data_file in all_files:

            df_sp = data_file[0]
            df_cl = data_file[1]

            if sorted(set(df_sp.columns.get_level_values(1))) == ['mean', 'std']:
                sp_n = df_sp.xs('mean', level=1, axis=1).values
                sp_s = df_sp.xs('std', level=1, axis=1).values

                L_n = df_cl["L*","mean"].values
                a_n = df_cl["a*","mean"].values
                b_n = df_cl["b*","mean"].values
                C_n = df_cl["C*","mean"].values
                h_n = df_cl["h","mean"].values
                dE76_n = df_cl["dE76","mean"].values
                dE00_n = df_cl["dE00","mean"].values
                dR_vis_n = df_cl["dR_vis","mean"].values

                L_s = df_cl["L*","std"].values
                a_s = df_cl["a*","std"].values
                b_s = df_cl["b*","std"].values
                C_s = df_cl["C*","std"].values
                h_s = df_cl["h","std"].values
                dE76_s = df_cl["dE76","std"].values
                dE00_s = df_cl["dE00","std"].values
                dR_vis_s = df_cl["dR_vis","std"].values
                
            else:
                sp_n = df_sp.xs('value', level=1, axis=1).values
                sp_s = df_sp.xs('value', level=1, axis=1)
                sp_s.loc[:,:] = 0
                sp_s = sp_s.values

                L_n = df_cl["L*","value"].values
                a_n = df_cl["a*","value"].values
                b_n = df_cl["b*","value"].values
                C_n = df_cl["C*","value"].values
                h_n = df_cl["h","value"].values
                dE76_n = df_cl["dE76","value"].values
                dE00_n = df_cl["dE00","value"].values
                dR_vis_n = df_cl["dR_vis","value"].values

                L_s = np.zeros(len(L_n))
                a_s = np.zeros(len(a_n))
                b_s = np.zeros(len(b_n))
                C_s = np.zeros(len(C_n))
                h_s = np.zeros(len(h_n))
                dE76_s = np.zeros(len(dE76_n))
                dE00_s = np.zeros(len(dE00_n))
                dR_vis_s = np.zeros(len(dR_vis_n))
            
            wl = data_file[0].iloc[:,0].values
            He = data_file[1]['He_MJ/m2','value'].values
            Hv = data_file[1]['Hv_Mlxh','value'].values
            t = data_file[1]['t_sec','value'].values
            
            spectral_data = xr.Dataset(
                {
                    'sp': (['wavelength','dose'], sp_n),
                    'sp_s': (['wavelength','dose'], sp_s)                
                },
                coords={
                    'wavelength': wl,   
                    'dose': He,
                    'He': ('dose', He),
                    'Hv': ('dose', Hv),  # Match radiant energy
                    't': ('dose', t)  # Match radiant energy
                }
            )

            color_data = xr.Dataset(
                {
                    'L*': (['dose'], L_n),
                    'a*': (['dose'], a_n),
                    'b*': (['dose'], b_n),
                    'C*': (['dose'], C_n),
                    'h': (['dose'], h_n),
                    'dE76': (['dose'], dE76_n),
                    'dE00': (['dose'], dE00_n),
                    'dR_vis': (['dose'], dR_vis_n),
                    'L*_s': (['dose'], L_s),
                    'a*_s': (['dose'], a_s),
                    'b*_s': (['dose'], b_s),
                    'C*_s': (['dose'], C_s),
                    'h_s': (['dose'], h_s),
                    'dE76_s': (['dose'], dE76_s),
                    'dE00_s': (['dose'], dE00_s),
                    'dR_vis_s': (['dose'], dR_vis_s),
                },
                coords={                    
                    'He': ('dose',He),
                    'Hv': ('dose',Hv),
                    't': ('dose',t),
                }
            )                
                    
            sp = spectral_data.set_xindex(["He","Hv","t"])
            cl = color_data.set_xindex(["He","Hv","t"])
            combined_data = xr.merge([sp, cl])

        all_data.append(combined_data)            
        
        
        if data == 'all':
            if xarray == False:                
                [data_sp.append(x[0]) for x in all_files]
                [data_cl.append(x[1]) for x in all_files]
                return data_sp, data_cl
            
            else:
                return all_data

        elif data == 'sp':
            if xarray == False:                
                [data_sp.append(x[0]) for x in all_files]                                           
            else:                
                data_sp = [x.sp for x in all_data]
                

            return data_sp
        
        elif data == 'cl':
            if xarray == False:
                [data_cl.append(x[1]) for x in all_files]
            else:
                data_cl = [x[['L*','a*','b*','C*','h','dE76','dE00','dR_vis']] for x in all_data]
            
            return data_cl
        
        elif data == 'Lab':
            if xarray == False:
                [data_cl.append(x[1][['L*','a*','b*']]) for x in all_files]
            else:
                data_cl = [x[['L*','a*','b*']] for x in all_data]

            return data_cl
        
        elif isinstance(data,list):
            if xarray == False:
                dic_doses = {'He': 'He_MJ/m2', 'Hv':'Hv_Mlxh', 't':'t_sec'}
                data = [dic_doses[x] if x in dic_doses.keys() else x for x in data]
                [data_cl.append(x[1][data]) for x in all_files]

            else:
                data = [elem for elem in data if elem not in ['Hv','He','t']]
                data_cl = [x[data] for x in all_data]
            
            return data_cl
        
        else:
            print("Enter a valid data parameter. It can either be a string ('sp', 'cl', 'Lab', 'all') or a list of strings ['dE00','dE76', 'L*', 'a*', 'b*', 'C*', 'h']")
            return None

    
    def get_metadata(self, labels:Optional[list] = 'all'):
        """Retrieve the metadata.

        Parameters
        ----------
        labels : Optional[list], optional
            A list of strings corresponding to the wanted metadata labels, by default 'all'
            The metadata labels can be found in the 'info' sheet of the excel files.
            When 'all', it returns all the metadata

        Returns
        -------
        pandas dataframe
            It returns the metadata inside a pandas dataframe where each column corresponds to a single file.
        """
        
        df = self.read_files()
        metadata = [x[0] for x in df]

        df_metadata = pd.DataFrame(index = metadata[0].set_index('parameter').index)

        for m in metadata:
            m = m.set_index('parameter')
            Id = m.loc['meas_id']['value']
            
            df_metadata[Id] = m['value']

        if labels == 'all':
            return df_metadata
        
        else:            
            return df_metadata.loc[labels]
       

    def get_Lab(self, illuminant:Optional[str] = 'default', observer:Optional[str] = 'default'):
        """
        Retrieve the CIE L*a*b* values.

        Parameters
        ----------
        illuminant : (str, optional)  
            Reference *illuminant* ('D65', or 'D50'). by default 'default'.
            When 'default', it fetches the illuminant value recorded in the db_config.json file of the package. If no value has been recorded, then it sets the illuminant value to 'D65'.
 
        observer : (str|int, optional)
            Reference *observer* in degree ('10' or '2'). by default 'default'.
            When 'default', it fetches the observer value recorded in the db_config.json file of the package. If no value has been recorded, then it sets the observer value to '10'.        

            
        Returns
        -------
        pandas dataframe
            It returns the L*a*b* values inside a dataframe where each column corresponds to a single file.
        """    
        DB = databases.DB()

        if observer == 'default':
            if isinstance(DB.get_colorimetry_info(), str):
                observer = '10deg'
            else:
                observer = DB.get_colorimetry_info().loc['observer']['value']

        else:
            observer = f'{str(observer)}deg'


        if illuminant == 'default':
            if isinstance(DB.get_colorimetry_info(), str):
                illuminant = 'D65'
            else:
                illuminant = DB.get_colorimetry_info().loc['illuminant']['value']
        
        
        observers = {
            '10deg': 'cie_10_1964',
            '2deg' : 'cie_2_1931',
        }
        cmfs_observers = {
            '10deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1964 10 Degree Standard Observer"],
            '2deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1931 2 Degree Standard Observer"] 
            }
        
        ccs_ill = colour.CCS_ILLUMINANTS[observers[observer]][illuminant]

        meas_ids = self.get_meas_ids               
        df_sp = self.get_spectra()   
        df_sp_nominal = [
            df.loc[:, pd.IndexSlice[:, 'mean']] if 'mean' in df.columns.get_level_values(1)
            else df.loc[:, pd.IndexSlice[:, 'value']]
            for df in df_sp
        ]

        df_Lab = []

        for df, meas_id in zip(df_sp_nominal, meas_ids):   
            
            Lab_values = pd.DataFrame(index=['L*','a*','b*']).T           
            
            for col in df.columns:
                
                sp = df[col]
                wl = df.index
                sd = colour.SpectralDistribution(sp,wl)                

                XYZ = colour.sd_to_XYZ(sd,cmfs_observers[observer], illuminant=colour.SDS_ILLUMINANTS[illuminant])        
                Lab = np.round(colour.XYZ_to_Lab(XYZ/100,ccs_ill),3)               
                Lab_values = pd.concat([Lab_values, pd.DataFrame(Lab, index=['L*','a*','b*']).T], axis=0)
                Lab_values.index = np.arange(0,Lab_values.shape[0])

            Lab_values.columns = pd.MultiIndex.from_product([[meas_id], Lab_values.columns])  
            df_Lab.append(Lab_values)

        return pd.concat(df_Lab, axis=1)           
              
     
    @property
    def get_meas_ids(self):
        """Return the measurement id numbers corresponding to the input files.
        """
        info = self.get_metadata()        
        return info.loc['meas_id'].values

    @property
    def get_objects(self):
        """Return the object id numbers corresponding to the input files.
        """

        metadata_parameters = self.get_metadata().index

        if 'object_id' in metadata_parameters:

            df_info = self.get_metadata(labels=['object_id'])
            objects = sorted(set(df_info.values[0]))

            return objects
                   
        else:
            print(f'The info tab of the microfading interim file(s) {self.files} does not contain an object_id parameter.')
            return None


    def compute_mean(self, return_data:Optional[bool] = True, criterion:Optional[str] = 'group', save:Optional[bool] = False, folder:Optional[str] = '.', filename:Optional[str] = 'default'):
        """Compute mean and standard deviation values of several microfading measurements.

        Parameters
        ----------
        return_data : Optional[bool], optional
            Whether to return the data, by default True        

        criterion : Optional[str], optional
            _description_, by default 'group'            

        save : Optional[bool], optional
            Whether to save the average data as an excel file, by default False

        folder : Optional[str], optional
            Folder where the excel file will be saved, by default 'default'
            When 'default', the file will be saved in the same folder as the input files
            When '.', the file will be saved in the current working directory
            One can also enter a valid path as a string.

        filename : Optional[str], optional
            Filename of the excel file containing the average values, by default 'default'
            When 'default', it will use the filename of the first input file
            One can also enter a filename, but without a filename extension.

        Returns
        -------
        tuple, excel file
            It returns a tuple composed of three elements (info, CIELAB data, spectral data). When 'save' is set to True, an excel is created to stored the tuple inside three distinct excel sheet (info, CIELAB, spectra).

        Raises
        ------
        RuntimeError
            _description_
        """

        if len(self.files) < 2:        
            raise RuntimeError('Not enough files. At least two measurement files are required to compute the average values.')
        

        def mean_std_with_nan(arrays):
            '''Compute the mean of several numpy arrays of different shapes.'''
            
            # Find the maximum shape
            max_shape = np.max([arr.shape for arr in arrays], axis=0)
                    
            # Create arrays with NaN values
            nan_arrays = [np.full(max_shape, np.nan) for _ in range(len(arrays))]
                    
            # Fill NaN arrays with actual values
            for i, arr in enumerate(arrays):
                nan_arrays[i][:arr.shape[0], :arr.shape[1]] = arr
                    
            # Calculate mean
            mean_array = np.nanmean(np.stack(nan_arrays), axis=0)

            # Calculate std
            std_array = np.nanstd(np.stack(nan_arrays), axis=0)
                    
            return mean_array, std_array
        
        
        def to_float(x):
            try:
                return float(x)
            except ValueError:
                return x


        ###### SPECTRAL DATA #######

        data_sp = self.get_spectra()        

        # Average the spectral data
        sp = mean_std_with_nan(data_sp)
        sp_mean = sp[0]
        sp_std = sp[1]
              
        
        # Retrieve the wavelength range
        wl = self.get_wavelength.iloc[:,0].values
        
        Id = 'meas_ID'
        # Create a multi-index pandas DataFrame
        header_tuples = [(Id, 'mean'),(Id,'std')]
        multiindex_cols = pd.MultiIndex.from_tuples(header_tuples, names=['meas_id', 'value'])
        
        data_df_sp = np.empty((len(wl), 2))       
        data_df_sp[:, 0::2] = sp_mean
        data_df_sp[:, 1::2] = sp_std
        df_sp_final = pd.DataFrame(data_df_sp,columns=multiindex_cols, index=wl)
        df_sp_final.index.name = 'wavelength_nm'
            
           
        ###### COLORIMETRIC DATA #######

        data_cl = self.get_cielab()        
        index_cl = data_cl[0].index

        # Average the colorimetric data    
        cl = mean_std_with_nan(data_cl)
        cl_mean = cl[0]
        cl_std = cl[1]

        # Create a multi-index pandas DataFrame
        cl_tuples = [(Id, 'mean'),(Id,'std')]
        multiindex_cols = pd.MultiIndex.from_tuples(cl_tuples, names=['meas_id', 'stats'])
        
        data_df_cl = np.empty((cl_mean.shape[0], cl_mean.shape[1] * 2))       
        data_df_cl[:, 0::2] = cl_mean
        data_df_cl[:, 1::2] = cl_std
        df_cl_final = pd.DataFrame(data_df_cl,columns=multiindex_cols, index=index_cl)
        df_cl_final.index.name = 'coordinates'
        
        
        ###### INFO #######

        data_info = self.get_metadata().fillna(' ')

        # Select the first column as a template
        df_info = data_info.iloc[:,0]
        

        # Rename title file
        df_info.rename({'[SINGLE REFLECTANCE MEASUREMENT]': '[MEAN REFLECTANCE MEASUREMENT]'}, inplace=True)

        # Date time
        most_recent_dt = max(data_info.loc['date_time'])
        df_info.loc['date_time'] = most_recent_dt
        
        # Project data info
        df_info.loc['project_id'] = '_'.join(sorted(set(data_info.loc['project_id'].values)))
        df_info.loc['project_leader'] = '_'.join(sorted(set(data_info.loc['project_leader'].values)))
        df_info.loc['co-researchers'] = '_'.join(sorted(set(data_info.loc['co-researchers'].values)))
        df_info.loc['start_date'] = '_'.join(sorted(set(data_info.loc['start_date'].values)))
        df_info.loc['end_date'] = '_'.join(sorted(set(data_info.loc['end_date'].values)))
        df_info.loc['keywords'] = '_'.join(sorted(set(data_info.loc['keywords'].values)))

        # Object data info
        if len(set([x.split('_')[0] for x in data_info.loc['institution'].values])) > 1:
            df_info.loc['institution'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['institution'].values])))
        
        df_info.loc['object_id'] = '_'.join(sorted(set(data_info.loc['object_id'].values)))
        df_info.loc['object_category'] = '_'.join(sorted(set(data_info.loc['object_category'].values)))
        df_info.loc['object_type'] = '_'.join(sorted(set(data_info.loc['object_type'].values)))
        df_info.loc['object_technique'] = '_'.join(sorted(set(data_info.loc['object_technique'].values)))
        df_info.loc['object_title'] = '_'.join(sorted(set(data_info.loc['object_title'].values)))
        df_info.loc['object_name'] = '_'.join(sorted(set(data_info.loc['object_name'].values)))
        df_info.loc['object_creator'] = '_'.join(sorted(set(data_info.loc['object_creator'].values)))
        df_info.loc['object_date'] = '_'.join(sorted(set(data_info.loc['object_date'].values)))
        df_info.loc['object_support'] = '_'.join(sorted(set(data_info.loc['object_support'].values)))
        df_info.loc['color'] = '_'.join(sorted(set(data_info.loc['color'].values)))
        df_info.loc['colorants'] = '_'.join(sorted(set(data_info.loc['colorants'].values)))
        df_info.loc['colorants_name'] = '_'.join(sorted(set(data_info.loc['colorants_name'].values)))
        df_info.loc['binding'] = '_'.join(sorted(set(data_info.loc['binding'].values)))
        df_info.loc['ratio'] = '_'.join(sorted(set(data_info.loc['ratio'].values)))
        df_info.loc['thickness_um'] = '_'.join(sorted(set(data_info.loc['thickness_um'].values)))
        df_info.loc['status'] = '_'.join(sorted(set(data_info.loc['status'].values)))

        # Device data info
        if len(set(data_info.loc['device_ID'].values)) > 1:
            df_info.loc['device_ID'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['device_ID'].values])))
        
        df_info.loc['model'] = '_'.join(sorted(set(data_info.loc['model'].values)))
        df_info.loc['brand'] = '_'.join(sorted(set(data_info.loc['brand'].values)))
        df_info.loc['measurement_mode'] = '_'.join(sorted(set(data_info.loc['measurement_mode'].values)))
        df_info.loc['zoom'] = '_'.join(sorted(set(data_info.loc['zoom'].values)))
        df_info.loc['iris'] = '_'.join(sorted(set(str(data_info.loc['iris'].values))))
        df_info.loc['geometry'] = '_'.join(sorted(set(data_info.loc['geometry'].values)))
        df_info.loc['distance_ill_mm'] = '_'.join(sorted(set(str(data_info.loc['distance_ill_mm'].values))))
        df_info.loc['distance_coll_mm'] = '_'.join(sorted(set(str(data_info.loc['distance_coll_mm'].values))))       

        
        if len(set(data_info.loc['fiber_ill'].values)) > 1:
            df_info.loc['fiber_ill'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['fiber_ill'].values])))

        if len(set(data_info.loc['fiber_coll'].values)) > 1:
            df_info.loc['fiber_coll'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['fiber_coll'].values])))

        
        if len(set(data_info.loc['lamp'].values)) > 1:
            df_info.loc['lamp'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['lamp'].values])))
        

        if len(set(data_info.loc['filter_ill'].values)) > 1:
            df_info.loc['filter_ill'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['filter_ill'].values])))

        if len(set(data_info.loc['white_reference'].values)) > 1:
            df_info.loc['white_reference'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['white_reference'].values])))
        

        # Analysis data info
        
        criterion_value = df_info.loc[criterion]
        object_id = df_info.loc['object_id']
        if criterion == 'group':            
            df_info.loc['meas_id'] = f'MF.{object_id}.{criterion_value}'
        elif criterion == 'object' or criterion == 'project':
             df_info.loc['meas_id'] = f'MF.{criterion_value}'
        else:
            print('Choose one of the following options for the criterion parameter: ["group", "object", "project"]')

        meas_nbs = '-'.join([x.split('.')[-1] for x in self.get_meas_ids])
        df_info.loc['group'] = f'{"-".join(sorted(set(data_info.loc["group"].values)))}_{meas_nbs}'    
        df_info.loc['group_description'] = '_'.join(sorted(set(data_info.loc['group_description'].values)))
        df_info.loc['background'] = '_'.join(sorted(set(data_info.loc['background'].values)))  

        if len(set(data_info.loc['specular_component'].values)) > 1:
            df_info.loc['specular_component'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['specular_component'].values]))) 

        
        df_info.loc['integration_time_ms'] = np.round(np.mean(data_info.loc['integration_time_ms'].astype(float).values),1)
        df_info.loc['average'] = '_'.join([str(x) for x in sorted(set(data_info.loc['average'].astype(str).values))])         
        df_info.loc['measurements_N'] = '_'.join([str(x) for x in sorted(set(data_info.loc['measurements_N'].astype(str).values))])
        df_info.loc['illuminant'] = '_'.join(sorted(set(data_info.loc['illuminant'].values)))
        df_info.loc['observer'] = '_'.join(sorted(set(data_info.loc['observer'].values)))
                  
        
        # Rename the column
        df_info.name = 'value'
                
        
        ###### SAVE THE MEAN DATAFRAMES #######
        
        if save:  

            # set the folder
            if folder == ".":
                folder = Path('.')  

            elif folder == 'default':
                folder = Path(self.files[0]).parent

            else:
                if Path(folder).exists():
                    folder = Path(folder)         

            # set the filename
            if filename == 'default':
                filename = f'{Path(self.files[0]).stem}_MEAN{Path(self.files[0]).suffix}'

            else:
                filename = f'{filename}.xlsx'

            
            # create a excel writer object
            with pd.ExcelWriter(folder / filename) as writer:

                df_info.to_excel(writer, sheet_name='info', index=True)
                df_cl_final.to_excel(writer, sheet_name="CIELAB", index=True)
                df_sp_final.to_excel(writer, sheet_name='spectra', index=True)
        

        ###### RETURN THE MEAN DATAFRAMES #######
            
        if return_data:
            return df_info, df_cl_final, df_sp_final   
    


    def plot_CIELAB(self, stds=[], colors:Union[str,list] = None, title:Optional[str] = None, fontsize:Optional[int] = 20, legend_labels:Union[str,list] = 'default', legend_position:Optional[str] = 'in', legend_fontsize:Optional[int] = 20, legend_title:Optional[str] = None, obs_ill:Optional[bool] = True, save:Optional[bool] = False, path_fig:Optional[str] = 'cwd'):
        """Plot the Lab values related to the microfading analyses.

        Parameters
        ----------
        stds : list, optional
            A list of standard variation values respective to each element given in the data parameter, by default []
          
        title : Optional[str], optional
            Whether to add a title to the plot, by default None

        fontsize : Optional[int], optional
            Fontsize of the plot (title, ticks, and labels), by default 24

        legend_labels : Union[str, list], optional
            A list of labels respective to each element given in the data parameter that will be shown in the legend. When the list is empty there is no legend displayed, by default 'default'
            When 'default', each label will composed of the Id number of the number followed by a short description

        legend_position : Optional[str], optional
            Position of the legend, by default 'in'
            The legend can either be inside the figure ('in') or outside ('out')

        legend_fontsize : Optional[int], optional
            Fontsize of the legend, by default 24

        legend_title : Optional[str], optional
            Add a title above the legend, by default ''

        save : bool, optional
            Whether to save the figure, by default False

        path_fig : str, optional
            Absolute path required to save the figure, by default 'cwd'
            When 'cwd', it will save the figure in the current working directory.

        Returns
        -------
        _type_
            It returns a figure with 4 subplots that can be saved as a png file.
        """

        data_Lab = self.get_cielab(coordinates=['L*', 'a*', 'b*'])
        data_Lab = [x.T.values for x in data_Lab]
       

        # Retrieve the metadata
        info = self.get_metadata()
        ids = [x for x in self.get_meas_ids if 'BW' not in x] 

        if 'group_description' in info.index:                
            group_descriptions = info.loc['group_description'].values

        else:
            group_descriptions = [''] * len(self.files)
               
        

        # Define the colour of the curves
        if colors == 'sample':
            pass           

        elif isinstance(colors, str):
            colors = [colors] * len(self.files)

        elif colors == None:
            colors = [None] * len(self.files)
        
        # Define the labels
        if legend_labels == 'default':
            legend_labels = [f'{x}-{y}' for x,y in zip(self.get_meas_ids,group_descriptions)]
            legend_title = 'Measurement $n^o$'

        # Whether to plot the observer and illuminant info
        if obs_ill:
            DB = databases.DB()
            if isinstance(DB.get_colorimetry_info(), str):
                observer = '10deg'
                illuminant = 'D65'
            else:
                observer = DB.get_colorimetry_info().loc['observer']['value']
                illuminant = DB.get_colorimetry_info().loc['illuminant']['value']

            dic_obs = {'10deg':'$\mathrm{10^o}$', '2deg':'$\mathrm{2^o}$'}            
            obs_ill = f'{dic_obs[observer]}-{illuminant}'
        
        else:
            obs_ill = None

        return plotting.CIELAB(data=data_Lab, legend_labels=legend_labels, colors=colors, title=title, fontsize=fontsize, legend_fontsize=legend_fontsize, legend_position=legend_position, legend_title=legend_title, obs_ill=obs_ill, save=save, path_fig=path_fig)


    def plot_swatches_circle(self, light_doses: Optional[list] = [0,0.5,1,2,5,15], JND:Optional[list] = [1,2,3,5,10], dose_unit:Union[str,tuple] = 'Hv', dE:Optional[bool] = True, fontsize: Optional[int] = 24, equation:Optional[str] = 'c0*(x**c1) + c2', initial_params:Optional[List[float]] = [0.1, 0.1], save:Optional[bool] = False, path_fig:Optional[str] = 'cwd', title:Optional[str] = None, report:Optional[bool] = False): 
        """Plot the microfading data with circular colored patches. 

        Parameters
        ----------
        light_doses : list, optional
            Light doses in Mlxh for which a coloured patches will be created, by default [0,0.5,1,2,5,1]
            There has been at least two numerical values in the list. The first value corresponds to the color background of the plot and is usually set to 0. The other values will be plotted as circular patches.

        JND : list, optional
            Whether to plot circular patches of just noticeable differences, by default [1,2,3,5,10]
            NOT YET IMPLEMENTED

        dose_unit : [str, tuple], optional
            Unit of the light energy dose, by default 'Hv'
            Any of the following units can be used: 'He', 'Hv', 't'. Where 'He' corresponds to radiant energy (MJ/m2), 'Hv' to exposure dose (Mlxh), and 't' to times (hours) (exh,50,10,365)
        
        dE : bool, optional
            Whether to include the dE00 value between the background and each circular patche, by default True

        fontsize : int, optional
            Fontsize of the plot (title, ticks, and labels), by default 24        

        equation : str, optional
            Mathematical equation used to fit the coordinate values, by default 'c0*(x**c1) + c2'.
            Any others mathematical can be given. The following equation can also be used for fitting microfading data: '((x) / (c0 + (c1*x)))'.

        initial_params : List[float], optional
            Initial guesses of the 'c' parameters given in the equation (c0, c1, c2, etc.), by default [0.1, 0.0]
            In the default values, only c0 and c1 are provided ; c2 is retrieved from the initial value of each colorimetric coordinate plot.

        save : bool, optional
            Whether to save the figure, by default False

        path_fig : str, optional
            Absolute path required to save the figure, by default 'cwd'
            When 'cwd', it will save the figure in the current working directory.

        title : str, optional
            Whether to add a title to the plot, by default None, by default None        

        report : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
       

        if title == 'default':
            title = list(self.get_meas_ids)

        x_range=(0, light_doses[-1]+0.05, 0.05)   
        x_values = np.arange(*x_range)     
        Lab = self.get_cielab(coordinates=['L*','a*','b*'], dose_unit='Hv')          
        data_Lab = [] 

        # Define the function to fit
        def fit_function(x, *params):
            param_dict = {f'c{i}': param for i, param in enumerate(params)}
            param_dict['x'] = x
            return eval(equation, globals(), param_dict) 

        
        
        for el in Lab:

            x = el.index.values
            df_Lab_extrapolated = pd.DataFrame(index=x_values) 

            for c in ['L*','a*','b*']:                    
                
                y = el[c].values.flatten()
                initial_value = y[0]
                initial_params=initial_params + [initial_value]

                def fit_function(x, *params):
                    param_dict = {f'c{i}': param for i, param in enumerate(params)}
                    param_dict['x'] = x
                    return eval(equation, globals(), param_dict) 

                #bounds = ([-np.inf] * len(initial_params), [np.inf, 1]) if len(initial_params) == 2 else ([-np.inf] * len(initial_params), [np.inf, 1, np.inf])

                # perform the curve fitting
                optimized_params, _ = curve_fit(fit_function, x, y, p0=initial_params) # bounds=bounds)
                
                # generate fitted y data
                fitted_y = fit_function(x_values, *optimized_params)
                #return fitted_y

                # Calculate R-squared value
                residuals = y - fit_function(x, *optimized_params)
                ss_res, ss_tot = np.sum(residuals**2), np.sum((y - np.mean(y))**2)        
                r_squared = np.round(1 - (ss_res / ss_tot), 3)

                #print(f'1st, {c}, R_sq = {r_squared}')

                if r_squared < 0.1:

                    initial_params=[0,0.1, initial_value]

                    def fit_function(x, *params):
                        param_dict = {f'c{i}': param for i, param in enumerate(params)}
                        param_dict['x'] = x
                        return eval(equation, globals(), param_dict) 
                    
                    # perform the curve fitting
                    optimized_params, _ = curve_fit(fit_function, x, y, p0=initial_params) # bounds=bounds)
                    
                    # generate fitted y data
                    fitted_y = fit_function(x_values, *optimized_params)

                    # Calculate R-squared value
                    residuals = y - fit_function(x, *optimized_params)
                    ss_res, ss_tot = np.sum(residuals**2), np.sum((y - np.mean(y))**2)        
                    r_squared = np.round(1 - (ss_res / ss_tot), 3)

                    #print(f'2nd, {c}, R_sq = {r_squared}')

                    if r_squared < 0.1:
                        initial_params=[-0.1,0.1, initial_value]

                        def fit_function(x, *params):
                            param_dict = {f'c{i}': param for i, param in enumerate(params)}
                            param_dict['x'] = x
                            return eval(equation, globals(), param_dict) 
                        
                        # perform the curve fitting
                        optimized_params, _ = curve_fit(fit_function, x, y, p0=initial_params) # bounds=bounds)
                        
                        # generate fitted y data
                        fitted_y = fit_function(x_values, *optimized_params)

                        # Calculate R-squared value
                        residuals = y - fit_function(x, *optimized_params)
                        ss_res, ss_tot = np.sum(residuals**2), np.sum((y - np.mean(y))**2)        
                        r_squared = np.round(1 - (ss_res / ss_tot), 3)
                    

                #extrapolated_values = self.compute_fitting(return_data=True, x_range=x_range, dose_unit='Hv', coordinate=c, initial_params=[0.1,0.1,initial_value], equation=equation)

                df_Lab_extrapolated[c] = fitted_y



            #return df_Lab_extrapolated  
            df_Lab_extrapolated.index = np.round(df_Lab_extrapolated.index,2)
            wanted_Lab = df_Lab_extrapolated.loc[light_doses].values
            #wanted_srgb = colour.XYZ_to_sRGB(colour.Lab_to_XYZ(wanted_Lab), d65).clip(0, 1)
            #data_srgb.append(wanted_srgb)
            data_Lab.append(wanted_Lab)
            #print(wanted_Lab)
        
        
            plotting.swatches_circle(data=[wanted_Lab], data_type='Lab', light_doses=light_doses, dE=dE, fontsize=fontsize, save=save, title=title, path_fig=path_fig)


    def plot_delta(self, stds:Optional[bool] = True, coordinates:Optional[list] = ['dE00'], dose_unit:Optional[str] = 'He', legend_labels:Union[str, list] = 'default', initial_values:Optional[bool] = False, colors:Union[str,list] = None, lw:Union[int,list] = 'default', title:Optional[str] = None, fontsize:Optional[int] = 24, legend_fontsize:Optional[int] = 24, legend_title:Optional[str] = None, xlim:Optional[tuple] = None, save:Optional[bool] = False, path_fig:Optional[str] = 'cwd'):
        """Plot the delta values of choosen colorimetric coordinates related to the microfading analyses.

        Parameters
        ----------
        stds : bool, optional
            Whether to show the standard deviation values if any, by default True.

        coordinates : list, optional
            List of colorimetric coordinates, by default ['dE00']
            Any of the following coordinates can be added to the list: 'dE76', 'dE00', 'dR_vis' , 'L*', 'a*', 'b*', 'C*', 'h'.

        dose_unit : str, optional
            Unit of the light energy dose, by default 'He'
            Any of the following units can be used: 'He', 'Hv', 't'. Where 'He' corresponds to radiant energy (MJ/m2), 'Hv' to exposure dose (Mlxh), and 't' to times (sec)

        legend_labels : Union[str, list], optional
            A list of labels respective to each element given in the data parameter that will be shown in the legend. When the list is empty there is no legend displayed, by default 'default'
            When 'default', each label will composed of the Id number of the number followed by a short description

        colors : Union[str, list], optional
            Define the colors of the curves, by default None
            When 'sample', the color of each line will be based on srgb values computed from the reflectance values. Alternatively, a single string value can be used to define the color (see matplotlib colour values) and will be applied to all the lines. Or a list of matplotlib colour values can be used. With a single coordinate, the list should have the same length as measurement files. With multiple coordinates, the list should have the same length as coordinates.

        lw : Union[int,list], optional
            Width of the lines, by default 'default'
            When 'default', it attributes a given a width according to each coordinates, otherwise it gives a value of 2.
            A single value (an integer) can be entered and applied to all the lines.
            A list of integers can also be entered. With a single coordinate, the list should have the same length as measurement files. With multiple coordinates, the list should have the same length as coordinates.

        title : str, optional
            Whether to add a title to the plot, by default None

        fontsize : int, optional
            Fontsize of the plot (title, ticks, and labels), by default 24

        legend_fontsize : int, optional
            Fontsize of the legend, by default 24

        legend_title : str, optional
            Add a title above the legend, by default ''

        xlim : tuple, optional
            A tuple of two integers that define the left and right limits of the x-axis , by default None.

        save : bool, optional
            Whether to save the figure, by default False

        path_fig : str, optional
            Absolute path required to save the figure, by default 'cwd'
            When 'cwd', it will save the figure in the current working directory.
        """

        # Retrieve the data
        if xlim == None:
            dose_values = 'all'
        elif isinstance(xlim, tuple):
            dose_values = (xlim[0], xlim[1], 0.05)
                
        all_data = self.compute_delta(coordinates=coordinates, dose_unit=dose_unit, dose_values=dose_values)
        nominal_data = []
        stdev_data = []

        for data in all_data:

            if sorted(set(data.columns.get_level_values(1))) == ['mean', 'std']:
                nominal = data.xs(key='mean', axis=1, level=1)
                if stds:
                    stdev = data.xs(key='std', axis=1, level=1)   
                else:
                    stdev = nominal.copy()   
                    stdev.iloc[:,:] = 0         

            else:
                                
                nominal = data
                stdev = data.copy()
                stdev.iloc[:,:] = 0
                

            nominal_data.append(nominal.reset_index().T.values)
            stdev_data.append(stdev.T.values)

        
        
        # Retrieve the metadata
        info = self.get_metadata()
        ids = [x for x in self.get_meas_ids]
        meas_nbs = [x.split('.')[-1] for x in ids]

        if 'group_description' in info.index:                
            group_descriptions = info.loc['group_description'].values

        else:
            group_descriptions = [''] * len(self.files)
               

        # Set the labels values
        if legend_labels == 'default':                       
            legend_labels = [f'{x}-{y}' for x,y in zip(ids, group_descriptions)] 

        elif legend_labels == '':
            legend_labels = []
        
        elif isinstance(legend_labels, list):
            legend_labels = legend_labels
            '''
            labels_list = []
            for i,Id in enumerate(self.get_meas_ids):
                label = Id.split('.')[-1]
                for el in labels:
                    label = label + f'-{self.get_metadata().loc[el].values[i]}'
                labels_list.append(label)

            labels = labels_list
            '''

        # Add the initial values of the colorimetric coordinates
        
        if initial_values:  
            initial_values = {}          
            for coord in coordinates:
                if coord in ['dL*', 'da*', 'db*', 'dC*', 'dh']:
                    initial_value = self.get_cielab(coordinates=[coord[1:]])[0][coord[1:]].iloc[0,:].values[0]
                    initial_values[coord[1:]] = initial_value
        else:
            initial_values = {}  

        if len(meas_nbs) > 1:
            initial_values = {}

        # Set the color of the lines according to the sample
        if colors == 'sample':
            colors = list(self.get_sRGB(dose_values=0).values.reshape(len(meas_nbs),-1))
            colors = colors * len(coordinates)
        
        # Whether to add a title or not
        if title == 'default':
            title = 'MFT'            
        elif title == 'none':
            title = None
        else:
            title = title 

        # Define the saving folder in case the figure should be saved
        filename = ''
        if save:
            if path_fig == 'default':
                path_fig = self.get_dir(folder_type='figures') / filename                

            if path_fig == 'cwd':
                path_fig = f'{os.getcwd()}/{filename}' 
        
        
        plotting.delta(data=nominal_data, yerr=stdev_data, dose_unit=[dose_unit], coordinates=coordinates, initial_values=initial_values, colors=colors, lw=lw, title=title, fontsize=fontsize, legend_labels=legend_labels, legend_fontsize=legend_fontsize, legend_title=legend_title, save=save, path_fig=path_fig)


    def plot_sp(self, stdev:Optional[bool] = False, spectra:Optional[str] = 'i', spectral_mode:Optional[str] = 'R', legend_labels:Union[str,list] = 'default', title:Optional[str] = None, fontsize:Optional[int] = 24, fontsize_legend:Optional[int] = 24, legend_title='', wl_range:Optional[tuple] = None, colors:Union[str,list] = None, lw:Union[int, list] = 2, ls:Union[str, list] = '-', save=False, path_fig='cwd', derivation=False, smoothing=(1,0), report:Optional[bool] = False):
        """Plot the reflectance spectra corresponding to the associated microfading analyses.

        Parameters
        ----------
        stdev : bool, optional
            Whether to show the standard deviation values, by default False

        spectra : Optional[str], optional
            Define which spectra to display, by default 'i'
            'i' for initial spectral, 
            'f' for final spectra,
            'i+f' for initial and final spectra, 
            'all' for all the spectra, 
            'doses' for spectra at different dose values indicated by the dose_unit and dose_values parameters
        
        spectral_mode : string, optional
            When 'R', it returns the reflectance spectra            
            When 'A', it returns the absorption spectra using the following equation: A = -log(R)

        legend_labels : Union[str, list], optional
            A list of labels respective to each element given in the data parameter that will be shown in the legend. When the list is empty there is no legend displayed, by default 'default'
            When 'default', each label will composed of the Id number of the number followed by a short description

        title : str, optional
            Whether to add a title to the plot, by default None

        fontsize : int, optional
            Fontsize of the plot (title, ticks, and labels), by default 24

        fontsize_legend : int, optional
            Fontsize of the legend, by default 24

        legend_title : str, optional
            Add a title above the legend, by default ''

        wl_range : tuple, optional
            Define the wavelength range with a two-values tuple corresponding to the lowest and highest wavelength values, by default None

        colors : Union[str, list], optional
            Define the colors of the reflectance curves, by default None
            When 'sample', the color of each line will be based on srgb values computed from the reflectance values. Alternatively, a single string value can be used to define the color (see matplotlib colour values) or a list of matplotlib colour values can be used. 

        lw : Union[int, list], optional
            Define the width of the plot lines, by default 2
            It can be a single integer value that will apply to all the curves. Or a list of integers can be used where the number of integer elements should match the number of reflectance curves.

        ls : Union[str, list], optional
            Define the line style of the plot lines, by default '-'
            It can be a string ('-', '--', ':', '-.') that will apply to all the curves. Or a list of string can be used where the number of string elements should match the number of reflectance curves.

        save : bool, optional
            Whether to save the figure, by default False

        path_fig : str, optional
            Absolute path required to save the figure, by default 'cwd'
            When 'cwd', it will save the figure in the current working directory.

        derivation : bool, optional
            Wether to compute and display the first derivative values of the spectra, by default False

        smooth : bool, optional
            Whether to smooth the reflectance curves, by default False

        smooth_params : list, optional
            Parameters related to the Savitzky-Golay filter, by default [10,1]
            Enter a list of two integers where the first value corresponds to the window_length and the second to the polyorder value. 

        report : Optional[bool], optional
            Configure some aspects of the figure for use in a report, by default False

        Returns
        -------
        _type_
            It returns a figure that can be save as a png file.
        """

        # Retrieve the metadata
        info = self.get_metadata()

        if 'group_description' in info.index:                
            group_descriptions = info.loc['group_description'].values

        else:
            group_descriptions = [''] * len(self.files)


        # Define the colour of the curves
        if colors == 'sample':
            colors = self.get_sRGB().iloc[0,:].values.clip(0,1).reshape(len(self.files),-1)

        elif isinstance(colors, str):
            colors = [colors] * len(self.files)

        elif colors == None:
            colors = [None] * len(self.files)
            
        # Define the labels
        if legend_labels == 'default':
            legend_labels = [f'{x}-{y}' for x,y in zip(self.get_meas_ids,group_descriptions)]
            legend_title = 'Measurement $n^o$'

        # Select the spectral data
        if spectra == 'i':            
            data_sp_all = self.get_spectra(wl_range=wl_range, smoothing=smoothing)
            data_sp = [x[x.columns.get_level_values(0)[0]] for x in data_sp_all]            

            text = 'Initial spectra'

        elif spectra == 'f':
            data_sp_all = self.get_spectra(wl_range=wl_range, smoothing=smoothing)
            data_sp =[x[x.columns.get_level_values(0)[-1]] for x in data_sp_all] 

            text = 'Final spectra'

        elif spectra == 'i+f':
            data_sp_all = self.get_spectra(wl_range=wl_range, smoothing=smoothing)
            data_sp = [x[x.columns.get_level_values(0)[[0]+[-1]]] for x in data_sp_all]            
            
            ls = ['-', '--'] * len(data_sp)
            lw = [3,2] * len(data_sp)
            black_lines = ['k'] * len(data_sp)            
            colors = list(itertools.chain.from_iterable(zip(colors, black_lines)))            
            

            if legend_labels == 'default':
                meas_labels = [f'{x}-{y}' for x,y in zip(self.get_meas_ids,group_descriptions)]
            else:
                meas_labels = legend_labels
            none_labels = [None] * len(meas_labels)
            legend_labels = [item for pair in zip(meas_labels, none_labels) for item in pair]

            text = 'Initial and final spectra (black dashed lines)'
              

        else:
            print(f'"{spectra}" is not an adequate value. Enter a value for the parameter "spectra" among the following list: "i", "f", "i+f", "doses".')
            return           
                                
        # whether to compute the absorption spectra
        if spectral_mode == 'abs':
            data_sp = [np.log(x) * (-1) for x in data_sp]
        
        # Reset the index
        data = [x.reset_index() for x in data_sp]
        
        # Whether to compute the first derivative
        if derivation:
            data = [pd.concat([x.iloc[:,0], pd.DataFrame(np.gradient(x.iloc[:,1:], axis=0))], axis=1) for x in data]

        # Compile the spectra to plot inside a list
        wanted_data = []  
        wanted_std = []

        # Set the wavelength column as index
        data = [x.set_index(x.columns.get_level_values(0)[0]) for x in data]          
             
        # Add the std values
        if stdev:            
            try:     
                
                values_data = [x.T.iloc[::2].values for x in data]
                values_wl = [x.index for x in data]
                for el1, wl in zip(values_data, values_wl):
                    for el2 in el1:
                        wanted_data.append((wl,el2))

                values_std = [x.T.iloc[1::2].values for x in data]                
                for el1 in values_std:
                    for el2 in el1:
                        wanted_std.append(el2)
            except IndexError:
                wanted_std = []
            
        else:
            for el in data:                
                data_values = [ (el.index,x) for x in el.T.values]
                wanted_data = wanted_data + data_values 
            wanted_std = []

        
        return plotting.spectra(data=wanted_data, stds=wanted_std, spectral_mode=spectral_mode, legend_labels=legend_labels, title=title, fontsize=fontsize, fontsize_legend=fontsize_legend, legend_title=legend_title, x_range=wl_range, colors=colors, lw=lw, ls=ls, text=text, save=save, path_fig=path_fig, derivation=derivation)
       

    def plot_sp_delta(self,spectra:Optional[tuple] = ('i','f'), dose_unit:Optional[str] = 'Hv', legend_labels:Union[str,list] = 'default', title:Optional[str] = None, fontsize:Optional[int] = 24, legend_fontsize:Optional[int] = 24, legend_title='', wl_range:Union[int,float,list,tuple] = None, colors:Union[str,list] = None, spectral_mode:Optional[str] = 'dR', derivation=False, smoothing=(1,0)):

        if spectra == ('i','f'):

            sp_data = [x.iloc[:,[0,-1]] for x in self.get_spectra(wl_range=wl_range, spectral_mode=spectral_mode)]
            sp_delta = [x.iloc[:,1] - x.iloc[:,0] for x in sp_data]
            wanted_data = [(x.index, x.values) for x in sp_delta]

        elif spectra[0] == 'i':
            
            sp1 = [x.iloc[:,0] for x in self.get_spectra(wl_range=wl_range, spectral_mode=spectral_mode)]
            sp2 = [x.values.flatten() for x in self.get_spectra(dose_unit=dose_unit, dose_values=float(spectra[1]),wl_range=wl_range, spectral_mode=spectral_mode)]
            
            wanted_data = [(x.index,np.array(y)-np.array(x)) for x,y in zip(sp1,sp2)]

        elif spectra[1] == 'f':

            sp1 = [x.values.flatten() for x in self.get_spectra(dose_unit=dose_unit, dose_values=float(spectra[0]), wl_range=wl_range, spectral_mode=spectral_mode)]            
            sp2 = [x.iloc[:,-1] for x in self.get_spectra(wl_range=wl_range, spectral_mode=spectral_mode)]            
            
            wanted_data = [(y.index,np.array(y)-np.array(x)) for x,y in zip(sp1,sp2)]
        
        else:

            wavelengths = self.get_wavelength.T.values
            sp1 = [x.values.flatten() for x in self.get_spectra(dose_unit=dose_unit, dose_values=float(spectra[0]),wl_range=wl_range, spectral_mode=spectral_mode)]
            sp2 = [x.values.flatten() for x in self.get_spectra(dose_unit=dose_unit, dose_values=float(spectra[1]),wl_range=wl_range, spectral_mode=spectral_mode)]
            
            wanted_data = [(w,np.array(y)-np.array(x)) for w,x,y in zip(wavelengths,sp1,sp2)]   
                 
        # Retrieve the metadata
        info = self.get_metadata()

        if 'group_description' in info.index:                
            group_descriptions = info.loc['group_description'].values

        else:
            group_descriptions = [''] * len(self.files)        
        
        
        # Define the colour of the curves
        if colors == 'sample':
            colors = self.get_sRGB().iloc[0,:].values.clip(0,1).reshape(len(self.files),-1)

        elif isinstance(colors, str):
            colors = [colors] * len(self.files)

        elif colors == None:
            colors = [None] * len(self.files)

        # Define the labels
        if legend_labels == 'default':
            legend_labels = [f'{x}-{y}' for x,y in zip(self.get_meas_ids,group_descriptions)]
            legend_title = 'Measurement $n^o$'
             
        # Whether to compute the first derivative
        if derivation:
            pass  # to implement
            #wanted_data = [x.reset_index() for x in wanted_data]
            #wanted_data = [pd.concat([x.iloc[:,0], pd.DataFrame(np.gradient(x.iloc[:,1:], axis=0))], axis=1) for x in wanted_data]
            #wanted_data = [x.set_index(x.columns.get_level_values(0)[0]) for x in wanted_data] 

        
        #return wanted_data
        plotting.spectra(data=wanted_data, spectral_mode=spectral_mode, x_range=wl_range, colors=colors, fontsize_legend=legend_fontsize, legend_labels=legend_labels, legend_title=legend_title, title=title, fontsize=fontsize, derivation=derivation)


    def get_illuminant(self, illuminant:Optional[str] = 'D65', observer:Optional[str] = '10'):
        """Set the illuminant values

        Parameters
        ----------
        illuminant : Optional[str], optional
            Select the illuminant, by default 'D65'
            It can be any value within the following list: ['A', 'B', 'C', 'D50', 'D55', 'D60', 'D65', 'D75', 'E', 'FL1', 'FL2', 'FL3', 'FL4', 'FL5', 'FL6', 'FL7', 'FL8', 'FL9', 'FL10', 'FL11', 'FL12', 'FL3.1', 'FL3.2', 'FL3.3', 'FL3.4', 'FL3.5', 'FL3.6', 'FL3.7', 'FL3.8', 'FL3.9', 'FL3.10', 'FL3.11', 'FL3.12', 'FL3.13', 'FL3.14', 'FL3.15', 'HP1', 'HP2', 'HP3', 'HP4', 'HP5', 'LED-B1', 'LED-B2', 'LED-B3', 'LED-B4', 'LED-B5', 'LED-BH1', 'LED-RGB1', 'LED-V1', 'LED-V2', 'ID65', 'ID50', 'ISO 7589 Photographic Daylight', 'ISO 7589 Sensitometric Daylight', 'ISO 7589 Studio Tungsten', 'ISO 7589 Sensitometric Studio Tungsten', 'ISO 7589 Photoflood', 'ISO 7589 Sensitometric Photoflood', 'ISO 7589 Sensitometric Printer']

        observer : Optional[str], optional
            Standard observer in degree, by default '10'
            It can be either '2' or '10'

        Returns
        -------
        tuple
            It returns a tuple with two set of values: the chromaticity coordinates of the illuminants (CCS) and the spectral distribution of the illuminants (SDS).
        """

        observers = {
            '10': "cie_10_1964",
            '2' : "cie_2_1931"
        }
       
        CCS = colour.CCS_ILLUMINANTS[observers[observer]][illuminant]
        SDS = colour.SDS_ILLUMINANTS[illuminant]

        return CCS, SDS

     
    def get_observer(self, observer:Optional[str] = '10'):
        """Set the observer.

        Parameters
        ----------
        observer : Optional[str], optional
            Standard observer in degree, by default '10'
            It can be either '2' or '10'

        Returns
        -------        
            Returns the x_bar,  y_bar, z_bar spectra between 360 and 830 nm.
        """

        observers = {
            '10': "CIE 1964 10 Degree Standard Observer",
            '2' : "CIE 1931 2 Degree Standard Observer"
        }

        return colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER[observers[observer]]
    

    def compute_sp_derivate(self):
        """Compute the first derivative values of reflectance spectra.

        Returns
        -------
        a list of pandas dataframes
            It returns the first derivative values of the reflectance spectra inside dataframes where each column corresponds to a single spectra.
        """

        sp = self.get_data(data='sp')                    

        sp_derivation = [pd.DataFrame(pd.concat([pd.DataFrame(np.gradient(x.iloc[:,:], axis=0), index=pd.Series(x.index), columns=x.columns)], axis=1),index=pd.Series(x.index), columns=x.columns) for x in sp]

        return sp_derivation
    

    def get_sRGB(self, illuminant='default', observer='default', clip:Optional[bool] = True):
        """Compute the sRGB values. 

        Parameters
        ----------
        illuminant : (str, optional)  
            Reference *illuminant* ('D65', or 'D50'). by default 'default'.
            When 'default', it fetches the illuminant value recorded in the db_config.json file of the package. If no value has been recorded, then it sets the illuminant value to 'D65'.
 
        observer : (str|int, optional)
            Reference *observer* in degree ('10' or '2'). by default 'default'.
            When 'default', it fetches the observer value recorded in the db_config.json file of the package. If no value has been recorded, then it sets the observer value to '10'.        

        clip : Optional[bool], optional
            Whether to constraint the srgb values between 0 and 1.

        Returns
        -------
        pandas dataframe
            It returns the sRGB values inside a dataframe where each column corresponds to a single file.
        """

        DB = databases.DB()

        if observer == 'default':
            if isinstance(DB.get_colorimetry_info(), str):
                observer = '10deg'
            else:
                observer = DB.get_colorimetry_info().loc['observer']['value']

        else:
            observer = f'{str(observer)}deg'


        if illuminant == 'default':
            if isinstance(DB.get_colorimetry_info(), str):
                illuminant = 'D65'
            else:
                illuminant = DB.get_colorimetry_info().loc['illuminant']['value']
        
        
        observers = {
            '10deg': 'cie_10_1964',
            '2deg' : 'cie_2_1931',
        }
        cmfs_observers = {
            '10deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1964 10 Degree Standard Observer"],
            '2deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1931 2 Degree Standard Observer"] 
            }
        
        ccs_ill = colour.CCS_ILLUMINANTS[observers[observer]][illuminant]

        meas_ids = self.get_meas_ids 

        df_sp = self.get_spectra()   
        df_sp_nominal = [
            df.loc[:, pd.IndexSlice[:, 'mean']] if 'mean' in df.columns.get_level_values(1)
            else df.loc[:, pd.IndexSlice[:, 'value']]
            for df in df_sp
        ] 

        df_srgb = []
        

        for df, meas_id in zip(df_sp_nominal, meas_ids):
            
            srgb_values = pd.DataFrame(index=['R','G','B']).T            

            for col in df.columns:
                
                sp = df[col]
                wl = df.index
                sd = colour.SpectralDistribution(sp,wl)                

                XYZ = colour.sd_to_XYZ(sd,cmfs_observers[observer], illuminant=colour.SDS_ILLUMINANTS[illuminant]) 
                srgb = np.round(colour.XYZ_to_sRGB(XYZ / 100, illuminant=ccs_ill), 4)                        
                srgb_values = pd.concat([srgb_values, pd.DataFrame(srgb, index=['R','G','B']).T], axis=0)
                srgb_values.index = np.arange(0,srgb_values.shape[0])

            srgb_values.columns = pd.MultiIndex.from_product([[meas_id], srgb_values.columns])
            
            if clip:
                srgb_values = srgb_values.clip(0,1)

            df_srgb.append(srgb_values)


        return pd.concat(df_srgb, axis=1)


    @property
    def get_wavelength(self):
        """Return the wavelength range of the microfading measurements.
        """
        data = self.get_spectra()

        wavelengths = pd.concat([pd.Series(x.index.values) for x in data], axis=1)
        wavelengths.columns = self.get_meas_ids

        return wavelengths


    def get_XYZ(self, illuminant:Optional[str] = 'default', observer:Union[str,int] = 'default'):
        """Compute the XYZ values. 

        Parameters
        ----------
        illuminant : (str, optional)  
            Reference *illuminant* ('D65', or 'D50'). by default 'default'.
            When 'default', it fetches the illuminant value recorded in the db_config.json file of the package. If no value has been recorded, then it sets the illuminant value to 'D65'.
 
        observer : (str|int, optional)
            Reference *observer* in degree ('10' or '2'). by default 'default'.
            When 'default', it fetches the observer value recorded in the db_config.json file of the package. If no value has been recorded, then it sets the observer value to '10'.        

        Returns
        -------
        pandas dataframe
            It returns the XYZ values inside a dataframe where each column corresponds to a single file.
        """

        DB = databases.DB()

        if observer == 'default':
            if isinstance(DB.get_colorimetry_info(), str):
                observer = '10deg'
            else:
                observer = DB.get_colorimetry_info().loc['observer']['value']

        else:
            observer = f'{str(observer)}deg'


        if illuminant == 'default':
            if isinstance(DB.get_colorimetry_info(), str):
                illuminant = 'D65'
            else:
                illuminant = DB.get_colorimetry_info().loc['illuminant']['value']               
        
        cmfs_observers = {
            '10deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1964 10 Degree Standard Observer"],
            '2deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1931 2 Degree Standard Observer"] 
            } 
        
        meas_ids = self.get_meas_ids                
        df_sp = self.get_spectra()   
        df_sp_nominal = [
            df.loc[:, pd.IndexSlice[:, 'mean']] if 'mean' in df.columns.get_level_values(1)
            else df.loc[:, pd.IndexSlice[:, 'value']]
            for df in df_sp
        ] 
          
        df_XYZ = []
        

        for df, meas_id in zip(df_sp_nominal, meas_ids):
            
            XYZ_values = pd.DataFrame(index=['X','Y','Z']).T

            for col in df.columns:
                
                sp = df[col]
                wl = df.index
                sd = colour.SpectralDistribution(sp,wl)                

                XYZ = np.round(colour.sd_to_XYZ(sd,cmfs_observers[observer], illuminant=colour.SDS_ILLUMINANTS[illuminant]),3)
                XYZ_values = pd.concat([XYZ_values, pd.DataFrame(XYZ, index=['X','Y','Z']).T], axis=0)
                XYZ_values.index = np.arange(0,XYZ_values.shape[0])

            XYZ_values.columns = pd.MultiIndex.from_product([[meas_id], XYZ_values.columns])
            df_XYZ.append(XYZ_values)

        return pd.concat(df_XYZ, axis=1)


    def get_xy(self, illuminant:Optional[str] = 'default', observer:Union[str, int] = 'default'):
        """Compute the xy values. 

        Parameters
        ----------
        illuminant : (str, optional)  
            Reference *illuminant* ('D65', or 'D50'). by default 'default'.
            When 'default', it fetches the illuminant value recorded in the db_config.json file of the package. If no value has been recorded, then it sets the illuminant value to 'D65'.
 
        observer : (str|int, optional)
            Reference *observer* in degree ('10' or '2'). by default 'default'.
            When 'default', it fetches the observer value recorded in the db_config.json file of the package. If no value has been recorded, then it sets the observer value to '10'.    

        Returns
        -------
        pandas dataframe
            It returns the xy values inside a dataframe where each column corresponds to a single file.
        """
        DB = databases.DB()

        if observer == 'default':
            if isinstance(DB.get_colorimetry_info(), str):
                observer = '10deg'
            else:
                observer = DB.get_colorimetry_info().loc['observer']['value']

        else:
            observer = f'{str(observer)}deg'


        if illuminant == 'default':
            if isinstance(DB.get_colorimetry_info(), str):
                illuminant = 'D65'
            else:
                illuminant = DB.get_colorimetry_info().loc['illuminant']['value']               
        
        cmfs_observers = {
            '10deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1964 10 Degree Standard Observer"],
            '2deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1931 2 Degree Standard Observer"] 
            }       
        
        
        meas_ids = self.get_meas_ids                
        df_sp = self.get_spectra()   
        df_sp_nominal = [
            df.loc[:, pd.IndexSlice[:, 'mean']] if 'mean' in df.columns.get_level_values(1)
            else df.loc[:, pd.IndexSlice[:, 'value']]
            for df in df_sp
        ]     
        df_xy = []
        

        for df, meas_id in zip(df_sp_nominal, meas_ids):
            
            xy_values = pd.DataFrame(index=['x','y']).T           

            for col in df.columns:
                
                sp = df[col]
                wl = df.index
                sd = colour.SpectralDistribution(sp,wl)                

                XYZ = colour.sd_to_XYZ(sd,cmfs_observers[observer], illuminant=colour.SDS_ILLUMINANTS[illuminant])
                xy = np.round(colour.XYZ_to_xy(XYZ),4)
                xy_values = pd.concat([xy_values, pd.DataFrame(xy, index=['x','y']).T], axis=0)
                xy_values.index = np.arange(0,xy_values.shape[0])

            xy_values.columns = pd.MultiIndex.from_product([[meas_id], xy_values.columns])
            df_xy.append(xy_values)

        return pd.concat(df_xy, axis=1)

    