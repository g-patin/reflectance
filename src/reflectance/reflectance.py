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
import msdb

# underlying modules of the  microfading package
from . import plotting
from . import config
from . import process_rawfiles
from . import utils

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


def is_DB(info_msg:Optional[bool] = True):
    "Check whether the databases files were created."

    # instantiate a DB class object
    
    DB_config = config.get_config_info()
    folder_db = DB_config['databases']['path_folder']

    if len(DB_config['databases']) == 0:
        print('The databases have not been registered nor configured. If you wish to create the database files, use the function "create_DB". If the database files have been created but you just want to set the path of the databases folder, then use the function "set_DB".')
        
        return False

    db_files = ['projects_info.csv', 'objects_info.csv','institutions.txt', 'users_info.txt','object_types.txt', 'object_techniques.txt', 'object_materials.txt', 'object_creators.txt']
        
    if all(list(map(os.path.isfile, [str(Path(folder_db)/x) for x in db_files]))):
        if info_msg:
            print(f'A database, called {DB_config["databases"]["db_name"]}, has been created for which the corresponding files can be found in the following directory: {folder_db}')            
        return True

    else:
        if info_msg:
            print('The databases files were created, but one or several files are currently missing.')
            print(f'The files should be located in the following directory: {folder_db}')

        return False


def get_datasets(device:Optional[str] = 'KM', rawfiles:Optional[bool] = False, stdev:Optional[bool] = False):
    """Retrieve exemples of dataset files. These files are meant to give the users the possibility to test the MFT class and its functions.  

    Parameters
    ----------
    device : Optional[str], optional
        Device that has been used to obtain the files, by default 'KM'
        One can choose a single option among the following choices: 'Avt', 'KM', 'OO', 'Tidas' 
        'AVT' corresponds to the Avantes spectrometer ....
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

    devices = ['AVT','KM', 'OO', 'Tidas']
    if device not in devices:
        print(f'The device values you entered ({device}) is not valid. Please enter a device value from the list : {devices}.')
        return


    # Whether to select rawfiles according to a choosen device
    if rawfiles:
        if device == 'AVT':
            data_files = ['2023-103_2751MK_01_G01_edge_2024-06-04.txt']

        elif device == 'KM':
            data_files = [
                '2024-03-26_noProject_GreyScales_SNV195805.txt',                
            ] 

        elif device == 'OO':
            data_files = [
                'LightBoxExp_0030_10_BW1-dark_Reflection_1.txt',
                'LightBoxExp_0030_10_BW1-dark_z_1.txt'
            ]   

        elif device == 'Tidas':
            data_files = [
                '2024-144_dayflower4_01_G01_0h_c01_000001.txt'
            ]


    else:    
        # Whether to select files with standard deviation values
        if stdev:
            if device == 'AVT':
                data_files = [
                    ''              
                ]

            elif device == 'KM':
                data_files = [
                    ''
                ]

            elif device == 'OO':
                data_files = [
                    ''
                ]
            
            elif device == 'Tidas':
                data_files = [
                    ''
                ]
            
        else:
            if device == 'AVT':
                data_files = [
                    '2023-103_RS.2751MK.01_G01_edge_heritage_Shirley-Temple_2024-06-04_SP04.xlsx',
                ]

            elif device == 'KM':
                data_files = [
                    '',                
                ]

            elif device == 'OO':
                data_files = [
                    '',
                ]
            
            elif device == 'Tidas':
                data_files = ['2024-144_RS.cochineal1.01_G01_35h_print_2024-10-08_SP03.xlsx']
   

    # Get the paths to the data files within the package
    file_paths = []
    for file_name in data_files:
        
        with pkg_resources.path('reflectance.datasets', file_name) as data_file:
             file_paths.append(data_file)


    return file_paths
   

def get_config(key:Optional[str] = 'all'):
    """Retrieve the content of the config_info.json file

    Parameters
    ----------
    key : Optional[str], optional
        Give you the possibility to retrieve a specific category of information, by default 'all'
        One can enter a key value among the following list: ['databases', 'devices', 'comments', 'colorimetry', 'fibers', 'filters', 'lamps', 'objectives']

    Returns
    -------
    dict
        It returns the information inside a dictionary.
    """

    # Retrieve the path of the config_info file
    config_file=Path(__file__).parent / 'config_info.json'
    
    # Load folder path from JSON file if it exists
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            config = json.load(file)

            if key != 'all':
                return config[key]
            else:
                return config
        
    else:
        print('The config_info.json has been deleted ! Please re-install the microfading package.')
        return None   


def get_colorimetry_info():
    """Retrieve the colorimetric information (observer and illuminant) recorded in the db_config.json file of the reflectance package.

    Returns
    -------
    pandas dataframe or string
        It returns the information inside a dataframe if they have been recorded.
    """

    if is_DB(info_msg=False):
        return config.get_colorimetry_info()


def get_institution_info():
    """Retrieve the information about the institution of the users.  
    """
    return config.get_institution_info()


def process_rawdata(
    files: list, 
    device: str, 
    filenaming:Optional[str] = 'default', 
    folder:Optional[str] = '.', 
    db:Optional[bool] = 'default', 
    preconfig:Optional[bool] = 'default', 
    comment:Optional[str] = '', 
    splice_correction:Union[tuple, str] = ([1000,1800], 10),
    interpolation_wl:Optional[tuple] = 'default', 
    rounding:Union[int, tuple, str] = (4,4), 
    authors:Optional[str] = 'XX', 
    organization:Optional[str] = 'XX',
    white_standard:Optional[str] = 'default',
    average:Optional[int] = 'unknown', 
    observer:Optional[str] = 'default', 
    background:Optional[str] = 'unknown',
    spot_size:Optional[float] = 'unknown',
    illuminant:Optional[str] = 'default', 
    output_format:Optional[str] = 'xlsx',
    delete_files:Optional[bool] = False, 
    return_data:Optional[bool] = False
    ):
    """Process reflectance spectroscopy raw files. 

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

    interpolation_wl : tuple, optional

    rounding : Union[int, tuple, str], optional
        Rounding the spectral and colorimetric values, by default (4,3)
        The integers correspond to the amount of digits after the decimal separator.
        When an integer is provided, it is applied to both the spectral and colorimetric values.
        When a tuple is provided, the first value relates to the spectral values while the second relates to the colorimetric values.
    
    authors : str, optional
        Initials of the persons that performed and processed the measurements, by default 'XX' (unknown).
        Make sure that you registered the persons in the persons.txt file (see function 'add_new_person').
        If there are several persons, use a dash to connect the initials (e.g: 'JD-MG-OL'). 

    organization : str, optional
        Name of the institution whithin which the measurements have been performed, by default 'XX' (unknown).      

    observer : str, optional
        Reference CIE *observer* in degree ('10deg' or '2deg'). by default 'default'.
        When 'default', it fetches the observer value recorded in the db_config.json file of the package. If no value has been recorded, then it sets the observer value to '10deg'.     

    illuminant : (str, optional)  
        Reference CIE *illuminant*. It can be any value of the following list: ['A', 'B', 'C', 'D50', 'D55', 'D60', 'D65', 'D75', 'E', 'FL1', 'FL2', 'FL3', 'FL4', 'FL5', 'FL6', 'FL7', 'FL8', 'FL9', 'FL10', 'FL11', 'FL12', 'FL3.1', 'FL3.2', 'FL3.3', 'FL3.4', 'FL3.5', 'FL3.6', 'FL3.7', 'FL3.8', 'FL3.9', 'FL3.10', 'FL3.11', 'FL3.12', 'FL3.13', 'FL3.14', 'FL3.15', 'HP1', 'HP2', 'HP3', 'HP4', 'HP5', 'LED-B1', 'LED-B2', 'LED-B3', 'LED-B4', 'LED-B5', 'LED-BH1', 'LED-RGB1', 'LED-V1', 'LED-V2', 'ID65', 'ID50']. by default 'default'.
        When 'default', it fetches the illuminant value recorded in the db_config.json file of the package. If no value has been recorded, then it sets the illuminant value to 'D65'.      

    output_format : str, optional
        Output file format for the interim files, by default 'xlsx'.
        By default, the interim files will be saved as '.xlxs' (excel). When 'ods', it will save the output files as '.ods', which is the open source version of excel.
      
    delete_files : bool, optional
        Whether to delete the raw files

    return_data : bool, optional
        Whether to return the processed data

    Returns
    -------
    Excel file
        It returns an excel or an opendocument file composed of three tabs (info, CIELAB, spectra).
    """

    # Load the databases function and config file    
    config_info = config.get_config_info()
    
    
    # Set the db value
    if db == 'default':
        if len(config_info['databases']) == 0:
            db = False            
        else:
            db = config_info['databases']['usage']

    
    # Set the observer value
    if observer == 'default':        
        if len(config_info['colorimetry']) == 0:
            observer = '10deg'
        else:
            observer = config.get_colorimetry_info().loc['observer'].values[0]

    
    # Set the illuminant value
    if illuminant == 'default':
        if len(config_info['colorimetry']) == 0:
            illuminant = 'D65'
        else:
            illuminant = config.get_colorimetry_info().loc['illuminant'].values[0]

    
    # Set the white reference value
    white_standard = utils.get_white_standard(white_standard, db, device_id=device)

    
    # Set the authors names
    authors = utils.get_authors(authors, db)

    
    # Set the organization info
    organization = utils.get_institution(organization, db)
        

    # Set the wavelengths interpolation behaviour
    if interpolation_wl == 'default' and db == False:
        interpolation_wl = 'standard'

    elif interpolation_wl == 'default' and db == True:
        interpolation_wl = config.get_config_info()['devices'][device]['interpolation']
        
    
    # Retrieve the defined process function
    if device.lower() in ['asd', 'avt', 'avantes', 'km', 'oo', 'tidas']:
        process_functions = {'asd':'RS_ASD', 'avt':'RS_Avt', 'avantes':'RS_Avt', 'km':'RS_KM', 'oo':'RS_OO', 'tidas':'RS_Tidas'}
        process_function = process_functions[device.lower()]

    elif device in config_info['devices'].keys():                     
        process_function = config_info['devices'][device]['process_function']

    else:
        print(f'Processed aborted ! The device parameter you entered ("{device}") has not been registered in the configuration file.)')
        print('To register info in the configuration file, see the documentation: https://g-patin.github.io/reflectance/')
        return
    
    
    # Run the process_rawfiles function according to the microfading device
    if process_function == 'RS_Tidas':        
        
        return process_rawfiles.RS_Tidas(
            files=files, 
            filenaming=filenaming, 
            folder=folder, 
            db=db, 
            comment=comment,
            device_ID=device, 
            interpolation_wl=interpolation_wl, 
            rounding=rounding, 
            authors=authors, 
            white_standard=white_standard, 
            observer=observer, 
            illuminant=illuminant, 
            delete_files=delete_files, 
            return_filename=return_data)
    

    elif process_function == 'RS_Avt':        
        
        return process_rawfiles.RS_Avt(
            files=files, 
            device_ID=device,
            db=db,
            preconfig=preconfig,
            interpolation_wl=interpolation_wl, 
            filenaming=filenaming, 
            folder=folder,            
            comment=comment,         
            rounding=rounding, 
            authors=authors, 
            organization=organization,
            white_standard=white_standard,
            background=background,
            spot_size=spot_size,
            observer=observer, 
            illuminant=illuminant, 
            delete_files=delete_files, 
            return_data=return_data)
    
    elif process_function == 'RS_ASD':

        return process_rawfiles.RS_ASD(
            raw_files=files, 
            filenaming=filenaming, 
            folder=folder, 
            db=db, 
            comment=comment, 
            splice_correction=splice_correction,
            device_ID=device, 
            interpolation_wl=interpolation_wl, 
            average=average,
            rounding=rounding, 
            authors=authors,
            background=background, 
            white_standard=white_standard, 
            observer=observer, 
            illuminant=illuminant, 
            delete_files=delete_files, 
            return_filename=return_data)


def remove_comments_info():
    """Remove the comments information of a desired device from the config_info.json file.  
    """
    return config.remove_comments_info()


def remove_devices_info():
    """Remove the information of a desired device from the config_info.json file.  
    """
    return config.remove_devices_info()
    

def remove_institution_info():
    """Remove the institution information from the config_info.json file.  
    """
    return config.remove_institution_info()


def remove_systems_info():
    """Remove the information of a desired system from the config_info.json file.  
    """
    return config.remove_systems_info()


def reset_config():
    """Reset the content of config_info.json to its initial state, i.e. all empty dictionaries.  
    """
    return config.reset_config()


def set_colorimetry_info():
    """Record the colorimetric information (observer and illuminant) in the config_info.json file of the reflectance package.
    """
    return config.set_colorimetry_info()   


def set_config_info():
    """Add new information inside config_info.json file

    Returns
    -------
    It returns an ipywidgets where you can update the content of the config_info.json file
    """
    
    return config.set_config_info()


def set_comments_info():
    """Record the comments order in the db_config.json file of the reflectance package. 
    It is only relevant if the software of the device has a "comment" entry where you can insert information.
    """
    return config.set_comments_info()


def set_DB(folder_path:Optional[str] = '', use:Optional[bool] = True):
    """Record the databases info in the db_config.json file of the reflectance package.
    """    
    return config.set_db(folder_path=folder_path, use=use)    


def set_devices_info():
    """Record the information related to the measurement device in the config_info.json file.
    """
    return config.set_devices_info()


def set_filenaming_interim():
    """Set the filenaming of interim files in the config_info.json file.
    """
    return config.set_filenaming_interim()


def set_filenaming_raw():
    """Set the filenaming of raw files in the config_info.json file.
    """
    return config.set_filenaming_raw()


def set_institution_info():
    """Record the institution information in the config_info.json file.
    """
    return config.set_institution_info()


def set_systems_info():
    """Record the information about measurement systems used to perform the reflectance measurements.
    """
    return config.set_systems_info()



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
       
    
    def get_spectra(self, wl_range:Union[int, float, list, tuple] = 'all', spectral_mode:Optional[str] = 'R', smoothing:Optional[tuple] = (1,0), derivation:Optional[bool] = False):
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
            When 'R' or 'r', it returns the reflectance spectra
            When 'DR' or 'dr, it returns the density reflection using the following equation: DR = -log(R)

        smoothing : tuple of two integers, optional
            Whether to smooth the reflectance data using the Savitzky-Golay filter from the Scipy package, by default (1,0)
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
            if spectral_mode.upper() == 'A':
                df_sp = np.log(df_sp) * (-1)
                                       

            # set the wavelengths
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

            
            # smooth the data            
            df_sp = pd.DataFrame(savgol_filter(df_sp.T.values, window_length=smoothing[0], polyorder=smoothing[1]).T, columns=df_sp.columns, index=wavelengths)
            
            
            # append the spectral data
            data_sp.append(df_sp) 


        # concat the list of spectra into a dataframe
        data_sp = pd.concat(data_sp, axis=1)

        
        # whether to compute the first derivation values
        if derivation:
            data_sp = pd.DataFrame(np.gradient(data_sp, axis=0), index=data_sp.index, columns=data_sp.columns)


        # rename the columns and index
        data_sp.columns.names = ['meas_ids', None]
        data_sp.index.names = ['wavelength_nm']

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
        concat_data = pd.concat(wanted_data, axis=1, ignore_index=False)
        

        if index == False:
            concat_data = concat_data.reset_index()
        
        else:
            concat_data.index.names = ['coordinates']

        return concat_data       
           
   
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
          
  
    def get_metadata(self, labels:Optional[list] = 'all', section:Optional[str] = 'all'):
        """Retrieve the metadata.

        Parameters
        ----------
        labels : Optional[list], optional
            A list of strings corresponding to the wanted metadata labels, by default 'all'
            The metadata labels can be found in the 'info' sheet of the excel files.
            When 'all', it returns all the metadata

        section : Optional[str], optional
            Retrieve metadata from one of the following sections: 'project', 'object', 'device', 'analysis', 'system', 'colorimetric'.
            For example, if you want to retrieve all the information about the objects, you can enter 'object' as a value.

        Returns
        -------
        pandas dataframe
            It returns the metadata inside a pandas dataframe where each column corresponds to a single file.
        """
        
        '''
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
        '''


        
        
        df = self.read_files()
        metadata = [x[0] for x in df]

        sections = ['project', 'object', 'system', 'device', 'analysis', 'colorimetric']
        df_metadata = pd.DataFrame(index = metadata[0].set_index('parameter').index)

        for m in metadata:
            m = m.set_index('parameter')
            Id = m.loc['meas_id']['value']
            
            df_metadata[Id] = m['value']

        
        if section in sections:

            if section == 'colorimetric':                
                df_metadata = df_metadata.loc[f'[COLORIMETRIC INFO]':]

            else:
                end_label = f'[{sections[sections.index(section) + 1].upper()} INFO]'
                df_metadata = df_metadata.loc[f'[{section.upper()} INFO]':end_label].iloc[:-1,:]

            return df_metadata
        
        elif labels != 'all' and isinstance(labels, (str,list)):
            return df_metadata.loc[labels] 
                
        else: 
            return df_metadata 
       

    def compute_Lab(self, illuminant:Optional[str] = 'default', observer:Optional[str] = 'default'):
        """
        Compute the CIE L*a*b* values.

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
        

        if observer == 'default':
            if len(config.get_colorimetry_info()) == 0:
                observer = '10deg'
            else:                
                observer = config.get_colorimetry_info().loc['observer']['value']

        else:
            observer = f'{str(observer)}deg'


        if illuminant == 'default':
            if len(config.get_colorimetry_info()) == 0:
                illuminant = 'D65'
            else:
                illuminant = config.get_colorimetry_info().loc['illuminant']['value']

        
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

        cols_to_keep = df_sp.columns[df_sp.columns.get_level_values(1).isin(['nominal', 'mean'])] 
        df_sp_nominal = df_sp[cols_to_keep]

        df_Lab = []

        df_Lab = pd.DataFrame(index=['L*','a*','b*']) 
        wl = df_sp_nominal.index

        for sp in df_sp_nominal.T.values:

            sd = colour.SpectralDistribution(sp,wl)
            XYZ = colour.sd_to_XYZ(sd,cmfs_observers[observer], illuminant=colour.SDS_ILLUMINANTS[illuminant])        
            Lab = np.round(colour.XYZ_to_Lab(XYZ/100,ccs_ill),3)
            df_Lab = pd.concat([df_Lab, pd.DataFrame(Lab, index=['L*','a*','b*'])], axis=1)

        df_Lab.columns = df_sp_nominal.columns
        df_Lab.index.names = ['coordinates']
        
        return df_Lab         
              
     
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


    def compute_delta(self, coordinates:Optional[list] = ['dE00'], reference:Optional[str] = 'first'):

        df_cl = self.get_cielab()
        cols_to_keep = df_cl.columns[df_cl.columns.get_level_values(1).isin(['value', 'mean'])] 
        df_cl_nominal = df_cl[cols_to_keep]

        len_cl = df_cl_nominal.shape[1]
        
        if len_cl == 1:
            print('Not enough spectra. There has to be at least two reflectance spectra.')
            return
        
        # define the reference data
        if reference == 'first':
            reference_data = df_cl_nominal.iloc[:,0]
        elif reference == 'last':
            reference_data = df_cl_nominal.iloc[:,-1]
        elif reference == 'mean':
            reference_data = df_cl_nominal.mean(axis=1)
        else:
            print(f'The value "{reference}" you entered is invalid. Please enter one of the following possibilities: "first", "last", "mean".')
            return
        
        
        # compute the delta values
        df_deltas = pd.DataFrame(df_cl_nominal.T.values - reference_data.values, index=df_cl_nominal.columns, columns=[f'd{x}' for x in df_cl_nominal.index]).T

        if 'dE00' in coordinates:
            dE00_values = [colour.delta_E(reference_data[['L*','a*','b*']].values, x) for x in df_cl_nominal.loc[['L*','a*','b*']].T.values]
            df_dE00 = pd.DataFrame(dE00_values, index=df_deltas.columns, columns=['dE00']).T
            df_deltas = pd.concat([df_deltas, df_dE00], axis=0)

        if 'dE76' in coordinates:
            dE76_values = [colour.delta_E(reference_data[['L*','a*','b*']].values, x, method='CIE 1976') for x in df_cl_nominal.loc[['L*','a*','b*']].T.values]
            df_dE76 = pd.DataFrame(dE76_values, index=df_deltas.columns, columns=['dE76']).T
            df_deltas = pd.concat([df_deltas, df_dE76], axis=0)

        if 'dE94' in coordinates:
            dE94_values = [colour.delta_E(reference_data[['L*','a*','b*']].values, x, method='CIE 1994') for x in df_cl_nominal.loc[['L*','a*','b*']].T.values]
            df_dE94 = pd.DataFrame(dE94_values, index=df_deltas.columns, columns=['dE94']).T
            df_deltas = pd.concat([df_deltas, df_dE94], axis=0)

        if 'dE94T' in coordinates:
            dE94T_values = [colour.delta_E(reference_data[['L*','a*','b*']].values, x, method='CIE 1994', textiles=True) for x in df_cl_nominal.loc[['L*','a*','b*']].T.values]
            df_dE94T = pd.DataFrame(dE94T_values, index=df_deltas.columns, columns=['dE94T']).T
            df_deltas = pd.concat([df_deltas, df_dE94T], axis=0)

        if 'CAM02' in coordinates:
            CAM02_values = [colour.delta_E(reference_data[['L*','a*','b*']].values, x, method='CAM02-UCS') for x in df_cl_nominal.loc[['L*','a*','b*']].T.values]
            df_CAM02 = pd.DataFrame(CAM02_values, index=df_deltas.columns, columns=['CAM02']).T
            df_deltas = pd.concat([df_deltas, df_CAM02], axis=0)

        if 'CAM16' in coordinates:
            CAM16_values = [colour.delta_E(reference_data[['L*','a*','b*']].values, x, method='CAM16-LCD') for x in df_cl_nominal.loc[['L*','a*','b*']].T.values]
            df_CAM16 = pd.DataFrame(CAM16_values, index=df_cl_nominal.columns, columns=['CAM16']).T
            df_deltas = pd.concat([df_cl_nominal, df_CAM16], axis=0)
        
        
        # return wanted values
        df_wanted = df_deltas.loc[coordinates]
        return df_wanted

    
    def compute_mcdm(self, rounding:Optional[int] = 3):

        df_cl = self.get_cielab(coordinates=['L*','a*','b*'])
        cols_to_keep = df_cl.columns[df_cl.columns.get_level_values(1).isin(['value', 'mean'])] 
        df_cl_nominal = df_cl[cols_to_keep]

        len_cl = df_cl_nominal.shape[1]    

        if len_cl == 1:
            print('Not enough spectra. There has to be at least two reflectance spectra.')
            return
        
        Lab_n = df_cl_nominal.mean(axis=1).values   
        dEs = []

        for meas in df_cl_nominal.columns:
            Lab = df_cl_nominal[meas].values
            dE = colour.delta_E(Lab, Lab_n)
            dEs.append(dE)

        mcdm = ufloat(np.round(np.mean(dEs),rounding), np.round(np.std(dEs, ddof=1),rounding))

        return mcdm

    
    def compute_mean(self, return_data:Optional[bool] = True, criterion:Optional[str] = 'spot_group', save:Optional[bool] = False, folder:Optional[str] = '.', filename:Optional[str] = 'default', rounding:Optional[tuple] = (4,4)):
        """Compute mean and standard deviation values of several microfading measurements.

        Parameters
        ----------
        return_data : Optional[bool], optional
            Whether to return the data, by default True        

        criterion : Optional[str], optional
            _description_, by default 'spot_group'            

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

        rounding : Optional[tuple], optional
            Number of decimal digits (colorimetric values, spectral values)

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
            mean_array = np.nanmean(np.stack(nan_arrays), axis=1)

            # Calculate std
            std_array = np.nanstd(np.stack(nan_arrays), axis=1)
                    
            return mean_array, std_array
        
        
        def to_float(x):
            try:
                return float(x)
            except ValueError:
                return x

        
        data_info = self.get_metadata().fillna(' ')        

        # Select the first column as a template
        df_info = data_info.iloc[:,0]


        criterion_value = df_info.loc[criterion]
        object_id = df_info.loc['object_id']

        if criterion == 'spot_group':
            meas_id = f'RS.{object_id}.{criterion_value}'            
            df_info.loc['meas_id'] = meas_id
        elif criterion == 'object' or criterion == 'project':
            meas_id = f'MF.{criterion_value}'
            df_info.loc['meas_id'] = meas_id
        else:
            print('Choose one of the following options for the criterion parameter: ["spot_group", "object", "project"]')


        ###### SPECTRAL DATA #######

        data_sp = self.get_spectra().T.values        

        # Average the spectral data

        sp = mean_std_with_nan([data_sp])
        sp_mean = np.round(sp[0][0], rounding[1])
        sp_std = np.round(sp[1][0], rounding[1])
              
        
        # Retrieve the wavelength range
        wl = self.get_wavelength.iloc[:,0].values
        
        
        # Create a multi-index pandas DataFrame
        header_tuples = [(meas_id, 'mean'),(meas_id,'std')]
        multiindex_cols = pd.MultiIndex.from_tuples(header_tuples, names=['meas_id', 'data_type'])
        
        data_df_sp = np.empty((len(wl), 2))       
        data_df_sp[:, 0::2] = np.array([sp_mean]).T
        data_df_sp[:, 1::2] = np.array([sp_std]).T
        df_sp_final = pd.DataFrame(data_df_sp,columns=multiindex_cols, index=wl)
        df_sp_final.index.name = 'wavelength_nm'
            
        
           
        ###### COLORIMETRIC DATA #######

        data_cl = self.get_cielab()        
        index_cl = data_cl.index

        # Average the colorimetric data    
        cl = mean_std_with_nan([data_cl.T.values])
        cl_mean = np.round(cl[0][0],rounding[0])
        cl_std = np.round(cl[1][0],rounding[0])

        # Create a multi-index pandas DataFrame
        cl_tuples = [(meas_id, 'mean'),(meas_id,'std')]
        multiindex_cols = pd.MultiIndex.from_tuples(cl_tuples, names=['meas_id', 'data_type'])
        
        data_df_cl = np.empty((cl_mean.shape[0], 2))       
        data_df_cl[:, 0::2] = np.array([cl_mean]).T
        data_df_cl[:, 1::2] = np.array([cl_std]).T
        df_cl_final = pd.DataFrame(data_df_cl,columns=multiindex_cols, index=index_cl)
        df_cl_final.index.name = 'coordinates'
                
        
        ###### INFO #######

                

        # Rename measurement type info
        df_info.loc['measurement_type'] = '[MEAN REFLECTANCE MEASUREMENT]'
        

        # Date time
        most_recent_dt = max(data_info.loc['datetime_analysis'])
        df_info.loc['datetime_analysis'] = most_recent_dt

        most_recent_dt_processing = max(data_info.loc['datetime_processing'])
        df_info.loc['datetime_processing'] = most_recent_dt_processing
        
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
        df_info.loc['object_owner'] = '_'.join(sorted(set(data_info.loc['object_owner'].values)))
        df_info.loc['object_material'] = '_'.join(sorted(set(data_info.loc['object_material'].values)))
        df_info.loc['support'] = '_'.join(sorted(set(data_info.loc['support'].values)))
        df_info.loc['color'] = '_'.join(sorted(set(data_info.loc['color'].values)))
        df_info.loc['colorants_id'] = '_'.join(sorted(set(data_info.loc['colorants_id'].values)))
        df_info.loc['colorants_name'] = '_'.join(sorted(set(data_info.loc['colorants_name'].values)))
        df_info.loc['binding'] = '_'.join(sorted(set(data_info.loc['binding'].values)))
        df_info.loc['ratio'] = '_'.join(sorted(set(data_info.loc['ratio'].values)))
        df_info.loc['thickness_um'] = '_'.join(sorted(set(data_info.loc['thickness_um'].values)))
        df_info.loc['status'] = '_'.join(sorted(set(data_info.loc['status'].values)))
        df_info.loc['object_comment'] = '_'.join(sorted(set(data_info.loc['object_comment'].values)))

        # Device data info
        if len(set(data_info.loc['device_id'].values)) > 1:
            df_info.loc['device_id'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['device_id'].values])))
        
        df_info.loc['device_type'] = '_'.join(sorted(set(data_info.loc['device_type'].values)))
        df_info.loc['model'] = '_'.join(sorted(set(data_info.loc['model'].values)))
        df_info.loc['brand'] = '_'.join(sorted(set(data_info.loc['brand'].values)))
        df_info.loc['software_version'] = '_'.join(sorted(set(data_info.loc['software_version'].values)))
        #df_info.loc['measurement_mode'] = '_'.join(sorted(set(data_info.loc['measurement_mode'].values)))
        #df_info.loc['zoom'] = '_'.join(sorted(set(data_info.loc['zoom'].values)))
        #df_info.loc['iris'] = '_'.join(sorted(set(str(data_info.loc['iris'].values))))
        df_info.loc['geometry'] = '_'.join(sorted(set(data_info.loc['geometry'].values)))
        df_info.loc['device_params'] = '_'.join(sorted(set(data_info.loc['device_params'].values)))
        #df_info.loc['distance_ill_mm'] = '_'.join(sorted(set(str(data_info.loc['distance_ill_mm'].values))))
        #df_info.loc['distance_coll_mm'] = '_'.join(sorted(set(str(data_info.loc['distance_coll_mm'].values))))       

        
        if len(set(data_info.loc['fiber_ill'].values)) > 1:
            df_info.loc['fiber_ill'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['fiber_ill'].values])))

        if len(set(data_info.loc['fiber_coll'].values)) > 1:
            df_info.loc['fiber_coll'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['fiber_coll'].values])))

        
        if len(set(data_info.loc['lamp'].values)) > 1:
            df_info.loc['lamp'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['lamp'].values])))
        

        if len(set(data_info.loc['filter_ill'].values)) > 1:
            df_info.loc['filter_ill'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['filter_ill'].values])))

        if len(set(data_info.loc['filter_coll'].values)) > 1:
            df_info.loc['filter_coll'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['filter_coll'].values])))

        if len(set(data_info.loc['white_standard'].values)) > 1:
            df_info.loc['white_standard'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['white_standard'].values])))
        

        # Analysis data info
        
        
        meas_nbs = '-'.join([x.split('.')[-1] for x in self.get_meas_ids])
        df_info.loc['spot_group'] = f'{"-".join(sorted(set(data_info.loc["spot_group"].values)))}_{meas_nbs}'    
        df_info.loc['spot_description'] = '_'.join(sorted(set(data_info.loc['spot_description'].values)))
        df_info.loc['background'] = '_'.join(sorted(set(data_info.loc['background'].values)))  

        if len(set(data_info.loc['specular_component'].values)) > 1:
            df_info.loc['specular_component'] = '_'.join(sorted(set([x.split('_')[0] for x in data_info.loc['specular_component'].values]))) 

        value_intTime = list(self.get_metadata('integration_time_ms').unique())     
        if len(value_intTime) == 1:            
            value_intTime = float(value_intTime[0])
        
        value_avgScans = list(self.get_metadata('average_scans').unique())     
        if len(value_avgScans) == 1:            
            value_avgScans = int(value_avgScans[0])

        
        df_info.loc['integration_time_ms'] = value_intTime
        df_info.loc['average_scans'] = value_avgScans
        df_info.loc['measurements_N'] = len(data_sp)
        df_info.loc['illuminant'] = '_'.join(sorted(set(data_info.loc['illuminant'].values)))
        df_info.loc['observer'] = '_'.join(sorted(set(data_info.loc['observer'].values)))
        df_info.loc['interpolation'] = '_'.join(sorted(set(data_info.loc['interpolation'].values)))
                  
        
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

            elif isinstance(filename, list):
                filename_values = [df_info.loc[x] if x != 'date' else str(df_info.loc['datetime_analysis']).split(' ')[0] for x in filename]

                filename = [x if x !='[MEAN REFLECTANCE MEASUREMENT]' else 'avg' for x in filename_values]
                filename = [x.split('_')[0] if '_' in x else x for x in filename]
                filename = '_'.join(filename)
                filename = f'{filename}.xlsx'

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
    

    def compute_sp_derivate(self):
        """Compute the first derivative values of reflectance spectra.

        Returns
        -------
        a list of pandas dataframes
            It returns the first derivative values of the reflectance spectra inside dataframes where each column corresponds to a single spectra.
        """

        sp = self.get_spectra(derivation=True)
        return sp
    
    
    def plot_CIELAB(self, std:Optional[bool] = True, colors:Union[str,list] = None, title:Optional[str] = None, fontsize:Optional[int] = 20, legend_labels:Union[str,list] = 'default', legend_position:Optional[str] = 'in', legend_fontsize:Optional[int] = 20, legend_title:Optional[str] = None, obs_ill:Optional[bool] = True, save:Optional[bool] = False, path_fig:Optional[str] = 'cwd'):
        """Plot the Lab values related to the microfading analyses.

        Parameters
        ----------
        std : bool, optional
            A list of standard variation values respective to each element given in the data parameter, by default []
          
        title : str, optional
            Whether to add a title to the plot, by default None

        fontsize : int, optional
            Fontsize of the plot (title, ticks, and labels), by default 24

        legend_labels : Union[str, list], optional
            A list of labels respective to each element given in the data parameter that will be shown in the legend. When the list is empty there is no legend displayed, by default 'default'
            When 'default', each label will composed of the Id number of the number followed by a short description

        legend_position : str, optional
            Position of the legend, by default 'in'
            The legend can either be inside the figure ('in') or outside ('out')

        legend_fontsize : int, optional
            Fontsize of the legend, by default 24

        legend_title : str, optional
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

        # Retrieve the data and std

        data_Lab = self.get_cielab(coordinates=['L*', 'a*', 'b*'])

        data_mean = []
        data_std = []

        for col in data_Lab.columns:
            data = data_Lab[col]
            
            if data.name[1] == 'mean':
                data_mean.append(data.values)
                
            if data.name[1] == 'std':
                if std:
                    data_std.append(data.values)
                else:
                    data_std.append(np.zeros(len(data.values)))
                
            if data.name[1] == 'nominal':
                data_mean.append(data.values)
                data_std.append(np.zeros(len(data.values))) 

        
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
            legend_labels = self.get_metadata('meas_id').values

            for col in data_Lab.columns:
                data = data_Lab[col]
                
                if data.name[1] == 'mean':
                    legend_labels.append(data.name[0])

                if data.name[1] == 'value':
                    legend_labels.append(data.name[0])
            
            legend_title = 'Measurement $n^o$'
                
        
        # Whether to plot the observer and illuminant info
        if obs_ill:
            
            if len(config.get_colorimetry_info()) == 0:
                observer = '10deg'
                illuminant = 'D65'
            else:
                observer = config.get_colorimetry_info().loc['observer']['value']
                illuminant = config.get_colorimetry_info().loc['illuminant']['value']

            dic_obs = {'10deg':'$\mathrm{10^o}$', '2deg':'$\mathrm{2^o}$'}            
            obs_ill = f'{dic_obs[observer]}-{illuminant}'
        
        else:
            obs_ill = None

        return plotting.CIELAB(data=data_mean, stds=data_std, legend_labels=legend_labels, colors=colors, title=title, fontsize=fontsize, legend_fontsize=legend_fontsize, legend_position=legend_position, legend_title=legend_title, obs_ill=obs_ill, save=save, path_fig=path_fig)
   

    def plot_sp(self, std:Optional[bool] = True, spectra:Optional[str] = 'i', spectral_mode:Optional[str] = 'R', figsize:Optional[tuple] = (15,8), legend_labels:Union[str,list] = 'default', title:Optional[str] = None, fontsize:Optional[int] = 24, fontsize_legend:Optional[int] = 24, legend_title:Optional[str] = 'default', wl_range:Optional[tuple] = None, colors:Union[str,list] = None, lw:Union[int, list] = 2, ls:Union[str, list] = '-', text:Optional[str] = None, save=False, path_fig='cwd', derivation=False, smoothing=(1,0), report:Optional[bool] = False):
        """Plot the reflectance spectra corresponding to the associated microfading analyses.

        Parameters
        ----------
        std : bool, optional
            Whether to show the standard deviation values, by default True

        spectra : Optional[str], optional
            Define which spectra to display, by default 'i'
            'i' for initial spectral, 
            'f' for final spectra,
            'i+f' for initial and final spectra, 
            'all' for all the spectra, 
            'doses' for spectra at different dose values indicated by the dose_unit and dose_values parameters
        
        spectral_mode : string, optional
            When 'R', it returns the reflectance spectra            
            When 'DR', it returns the density reflection using the following equation: DR = -log(R)

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

        # retrieve the data
        data_sp = self.get_spectra(wl_range=wl_range, spectral_mode=spectral_mode, smoothing=smoothing, derivation=derivation)

        wavelengths = data_sp.index
        data_n = []   # wavelengths + nominal data        
        data_s = []   # standard deviation data

        for col in data_sp.columns:
            data = data_sp[col]
            
            if data.name[1] == 'mean':
                data_n.append(np.array([wavelengths,data.values]))
                
            if data.name[1] == 'std':
                if std:
                    data_s.append(data.values)
                else:
                    data_s.append(np.zeros(len(data.values)))
                
            if data.name[1] == 'nominal':
                data_n.append(np.array([wavelengths,data.values]))
                data_s.append(np.zeros(len(data.values)))         
        

        # define the labels of the legend
        if legend_labels == 'default':
            legend_labels = self.get_metadata(labels='meas_id').values
            """
            for col in data_sp.columns:
                data = data_sp[col]
                
                if data.name[1] == 'mean':
                    legend_labels.append(data.name[0])

                if data.name[1] == 'nominal':
                    legend_labels.append(data.name[0])

            """

        elif legend_labels == None or legend_labels == 'off':
            legend_labels = ''
                    
        
        # define the title of the legend
        if legend_title == 'default':
            legend_title = 'Measurement $n^o$'

        
        # define the colors of the curves
        if colors == 'sample':
            colors = self.compute_sRGB().T.values

        elif isinstance(colors, str):
            colors = [colors] * len(data_n)

        elif colors == None:
            colors = [None] * len(data_n)



        return plotting.spectra(data=data_n, stds=data_s, spectral_mode=spectral_mode, figsize=figsize, legend_labels=legend_labels, title=title, fontsize=fontsize, fontsize_legend=fontsize_legend, legend_title=legend_title, x_range=wl_range, colors=colors, lw=lw, ls=ls, text=text, save=save, path_fig=path_fig, derivation=derivation)
        



        # Retrieve the metadata
        info = self.get_metadata()

        if 'group_description' in info.index:                
            group_descriptions = info.loc['group_description'].values

        else:
            group_descriptions = [''] * len(self.files)


        # Define the colour of the curves
        if colors == 'sample':
            colors = self.compute_sRGB().iloc[0,:].values.clip(0,1).reshape(len(self.files),-1)

        elif isinstance(colors, str):
            colors = [colors] * len(self.files)

        elif colors == None:
            colors = [None] * len(self.files)
            
        # Define the labels
        if legend_labels == 'default':
            legend_labels = [f'{x}-{y}' for x,y in zip(self.get_meas_ids,group_descriptions)]
            legend_title = 'Measurement $n^o$'

        '''
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
        '''                        
        
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
       

    def plot_sp_delta(self,spectra:Optional[tuple] = ('i','f'), legend_labels:Union[str,list] = 'default', title:Optional[str] = None, fontsize:Optional[int] = 24, legend_fontsize:Optional[int] = 24, legend_title='', wl_range:Union[int,float,list,tuple] = None, colors:Union[str,list] = None, ls:Union[str,list] = None, lw:Union[str,list] = None, spectral_mode:Optional[str] = 'dR', derivation=False, smoothing=(1,0)):

        df_sp = self.get_spectra(wl_range=wl_range, spectral_mode=spectral_mode,smoothing=smoothing)
        cols_to_keep = df_sp.columns[df_sp.columns.get_level_values(1).isin(['value', 'mean'])] 
        df_sp_nominal = df_sp[cols_to_keep]

        len_sp = df_sp_nominal.shape[1]

        def get_spectrum(x):
            if len(x.columns) == 1:
                sp = unumpy.uarray(x.values.flatten(), np.zeros(len(x)))
                            
            else:
                sp = unumpy.uarray(x.iloc[:,0].values.flatten(), x.iloc[:,1].values.flatten())

            return sp


        if len_sp == 1:
            print('Not enough spectra. There has to be at least two reflectance spectra.')
            return
        
        elif len_sp == 2:
            cols = df_sp.columns.get_level_values(0)
            sp1 = get_spectrum(df_sp[cols[0]])
            sp2 = get_spectrum(df_sp[cols[1]])

            wanted_data = [(df_sp.index,unumpy.nominal_values(sp2-sp1))]  
            wanted_std = [unumpy.std_devs(x) for x in wanted_data]

        else:
            cols = df_sp.columns.get_level_values(0)
            sp_ref = get_spectrum(df_sp[cols[0]])

            sp_deltas = df_sp_nominal.T.values - sp_ref
            wanted_data = [(df_sp_nominal.index,unumpy.nominal_values(x)) for x in sp_deltas]
            wanted_std = [unumpy.std_devs(x) for x in sp_deltas]         
                  
                 
        # Retrieve the metadata
        info = self.get_metadata()

        if 'group_description' in info.index:                
            group_descriptions = info.loc['group_description'].values

        else:
            group_descriptions = [''] * len(self.files)        
        
        
        # Define the colour of the curves
        if colors == 'sample':
            colors = self.compute_sRGB().iloc[0,:].values.clip(0,1).reshape(len(self.files),-1)

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
         
        # Define the line styles
        ls = ['-'] * len(wanted_data) * 2

        # Define the line thickness
        lw = [2] * len(wanted_data) * 2
        
        
        plotting.spectra(data=wanted_data, stds=wanted_std, spectral_mode=spectral_mode, x_range=wl_range, colors=colors, lw=lw, ls=ls, fontsize_legend=legend_fontsize, legend_labels=legend_labels, legend_title=legend_title, title=title, fontsize=fontsize, derivation=derivation)


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
    
    
    def compute_sRGB(self, illuminant='default', observer='default', clip:Optional[bool] = True):
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
            Whether to constraint the srgb values between 0 and 1. by default True.

        Returns
        -------
        pandas dataframe
            It returns the sRGB values inside a dataframe where each column corresponds to a single file.
        """
                

        if observer == 'default':
            if len(config.get_colorimetry_info()) == 0:
                observer = '10deg'
            else:
                observer = config.get_colorimetry_info().loc['observer']['value']

        else:
            observer = f'{str(observer)}deg'


        if illuminant == 'default':
            if len(config.get_colorimetry_info()) == 0:
                illuminant = 'D65'
            else:
                illuminant = config.get_colorimetry_info().loc['illuminant']['value']
        
        
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

        cols_to_keep = df_sp.columns[df_sp.columns.get_level_values(1).isin(['nominal', 'mean'])] 
        df_sp_nominal = df_sp[cols_to_keep]

        
        df_srgb = pd.DataFrame(index=['R','G','B']) 
        wl = df_sp_nominal.index

        for sp in df_sp_nominal.T.values:

            sd = colour.SpectralDistribution(sp,wl)
            XYZ = colour.sd_to_XYZ(sd,cmfs_observers[observer], illuminant=colour.SDS_ILLUMINANTS[illuminant])        
            srgb = np.round(colour.XYZ_to_sRGB(XYZ / 100, illuminant=ccs_ill), 4)   
            if clip:
                srgb = srgb.clip(0,1)
            df_srgb = pd.concat([df_srgb, pd.DataFrame(srgb, index=['R','G','B'])], axis=1)

        df_srgb.columns = df_sp_nominal.columns
        df_srgb.index.names = ['coordinates']
        
        return df_srgb


    @property
    def get_wavelength(self):
        """Return the wavelength range of the microfading measurements.
        """
        df_sp = self.get_spectra()

        cols_to_keep = df_sp.columns[df_sp.columns.get_level_values(1).isin(['nominal', 'mean'])] 
        df_sp_nominal = df_sp[cols_to_keep]

        wls = {}
        for col in df_sp_nominal.columns:
            sp_data = df_sp_nominal[col].dropna(axis=0)            
            wls[col] = sp_data.index

        wavelengths = pd.DataFrame.from_dict(wls,orient='index').T
        
        return wavelengths


    def compute_XYZ(self, illuminant:Optional[str] = 'default', observer:Union[str,int] = 'default'):
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

        
        if observer == 'default':
            if len(config.get_colorimetry_info()) == 0:
                observer = '10deg'
            else:
                observer = config.get_colorimetry_info().loc['observer']['value']

        else:
            observer = f'{str(observer)}deg'


        if illuminant == 'default':
            if len(config.get_colorimetry_info()) == 0:
                illuminant = 'D65'
            else:
                illuminant = config.get_colorimetry_info().loc['illuminant']['value']
        
        
        cmfs_observers = {
            '10deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1964 10 Degree Standard Observer"],
            '2deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1931 2 Degree Standard Observer"] 
            }
        
                     
        df_sp = self.get_spectra() 

        cols_to_keep = df_sp.columns[df_sp.columns.get_level_values(1).isin(['nominal', 'mean'])] 
        df_sp_nominal = df_sp[cols_to_keep]

        
        df_XYZ = pd.DataFrame(index=['X','Y','Z']) 
        wl = df_sp_nominal.index

        for sp in df_sp_nominal.T.values:

            sd = colour.SpectralDistribution(sp,wl)
            XYZ = np.round(colour.sd_to_XYZ(sd,cmfs_observers[observer], illuminant=colour.SDS_ILLUMINANTS[illuminant]),4)                    
            df_XYZ = pd.concat([df_XYZ, pd.DataFrame(XYZ, index=['X','Y','Z'])], axis=1)

        df_XYZ.columns = df_sp_nominal.columns
        df_XYZ.index.names = ['coordinates']
        
        return df_XYZ
          

    def compute_xy(self, illuminant:Optional[str] = 'default', observer:Union[str, int] = 'default'):
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
        

        if observer == 'default':
            if len(config.get_colorimetry_info()) == 0:
                observer = '10deg'
            else:
                observer = config.get_colorimetry_info().loc['observer']['value']

        else:
            observer = f'{str(observer)}deg'


        if illuminant == 'default':
            if len(config.get_colorimetry_info()) == 0:
                illuminant = 'D65'
            else:
                illuminant = config.get_colorimetry_info().loc['illuminant']['value']
     
        
        cmfs_observers = {
            '10deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1964 10 Degree Standard Observer"],
            '2deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1931 2 Degree Standard Observer"] 
            }
        
                       
        df_sp = self.get_spectra() 

        cols_to_keep = df_sp.columns[df_sp.columns.get_level_values(1).isin(['nominal', 'mean'])] 
        df_sp_nominal = df_sp[cols_to_keep]

        
        df_xy = pd.DataFrame(index=['x','y']) 
        wl = df_sp_nominal.index

        for sp in df_sp_nominal.T.values:

            sd = colour.SpectralDistribution(sp,wl)
            XYZ = colour.sd_to_XYZ(sd,cmfs_observers[observer], illuminant=colour.SDS_ILLUMINANTS[illuminant]) 
            xy = np.round(colour.XYZ_to_xy(XYZ),4)           
            df_xy = pd.concat([df_xy, pd.DataFrame(xy, index=['x','y'])], axis=1)

        df_xy.columns = df_sp_nominal.columns
        df_xy.index.names = ['coordinates']
        
        return df_xy






    