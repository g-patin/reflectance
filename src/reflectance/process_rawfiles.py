import os
import pandas as pd
import numpy as np
import colour
from math import pi
from pathlib import Path
from typing import Optional, Union
import scipy.interpolate as sip
from scipy.interpolate import RegularGridInterpolator
from io import StringIO


from . import databases
from . import RS_info_templates


def RS_Tidas(files: list, device_ID:Optional[str] = 'default', db:Optional[bool] = 'default', filenaming:Optional[str] = 'none', folder:Optional[str] = '.',  comment:Optional[str] = '', interpolation_wl:Optional[tuple] ='default', rounding_sp:Optional[int] = 'none',  authors:Optional[str] = 'XX', white_standard:Optional[bool] = 'default', observer:Optional[str] = 'default', illuminant:Optional[str] = 'default', background:Optional[str] = 'black', delete_files:Optional[bool] = True, return_filename:Optional[bool] = True):

    # check whether the objects and projects databases have been created    
    if db:    

        DB = databases.DB()
        if DB.folder_db is None or DB.folder_db == 'folder_path':
            return 'Databases have not been created. Please, create databases by running the function "create_DB" from the reflectance package.'
        
        else:     
            db_projects, db_objects = DB.get_db()

            # remove the column 'project_id'
            if 'project_id' in db_objects.columns:
                db_objects = db_objects.drop('project_id', axis=1)
    
    else:
        filenaming = 'none'
     

    # define the illuminant value
    if illuminant == 'default' and db == True:
        if len(DB.get_colorimetry_info()) == 0:
            illuminant = 'D65'
        else:
            illuminant = DB.get_colorimetry_info().loc['illuminant']['value']

    elif illuminant == 'default' and db == False:
        illuminant = 'D65'

    
    # define the observer
    if observer == 'default' and db == True:
        if len(DB.get_colorimetry_info()) == 0:
            observer = '10deg'
        else:
            observer = DB.get_colorimetry_info().loc['observer']['value']

    elif observer == 'default' and db == False:
        observer = '10deg'


    # define dictionaries for colorimetric calculations
    observers = {        
        '10deg': 'cie_10_1964',
        '2deg' : 'cie_2_1931',
    }
    
    cmfs_observers = {
        '10deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1964 10 Degree Standard Observer"],
        '2deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1931 2 Degree Standard Observer"] 
    }

    
    # get the colorimetric data for illuminant and observer
    illuminant_SDS = colour.SDS_ILLUMINANTS[illuminant]
    illuminant_CCS = colour.CCS_ILLUMINANTS[observers[observer]][illuminant]
    cmfs = cmfs_observers[observer]


    # define the authors names
    if authors == 'XX':
        authors_names = 'unknown' 

    elif db:
        df_authors = DB.get_persons()
        if '-' in authors or ' - ' in authors:                     
            list_authors = []
            for x in authors.split('-'):
                x = x.strip()
                df_author = df_authors[df_authors['initials'] == x]
                list_authors.append(f"{df_author['surname'].values[0]}, {df_author['name'].values[0]}")                    
            authors_names = '_'.join(list_authors)
                    
        else:            
            if authors in df_authors['initials'].values:
                df_author = df_authors[df_authors['initials'] == authors]
                authors_names = f"{df_author['surname'].values[0]}, {df_author['name'].values[0]}"
            
            else:
                print(f'The author name "{authors}" has not been registered in the databases. Use the function add_new_person() to register the person.')
                authors_names = authors

    else:
        authors_names = authors


    # retrieve the white standard info
    if white_standard == 'default' and db == True:
        if len(DB.get_colorimetry_info()) == 0:
            white_standard = 'undefined'
        else:
            white_standard_ID = DB.get_colorimetry_info().loc['white_standard']['value']

    elif white_standard == 'default' and db == False:
        white_standard = 'undefined'
    

    # retrieve the general device info

    if db:
        config_devices = DB.get_db_config()['devices']
        config_device = config_devices[device_ID]

        device_model = config_device['model']
        device_brand = config_device['brand']
        geometry = config_device['geometry']
        fiber_ill = config_device['fiber_ill']
        fiber_coll = config_device['fiber_coll']
        specular_component = config_device['specular_component']

        
    
    # get the raw files
    raw_txt_files = [Path(file) for file in files if '.txt' in Path(file).name]    
    
    

    # process each raw file
    for raw_file in raw_txt_files:

        # get the raw files (to be deleted at the end)
        raw_files = [x for x in files if str(raw_file.stem) in Path(x).name]
    
        # define filenames
        file_path = Path(raw_file) 
        stemName = file_path.stem
        
        # open the raw file    
        f = open(file_path, encoding="ISO-8859-1").read()  
        
          
        ####### RETRIEVE THE INFO ########    
          
        lookfor_params = '[LOGIN]'                
            
        string_header = '\n'.join([ x.strip()[:] for x in f[:f.index(lookfor_params)].splitlines()[1:]])
        string_params = '\n'.join([ x.strip()[:] for x in f[f.index(lookfor_params)+len(lookfor_params):f.index('[DIO]')].splitlines()[1:]])
                
        fake_file_header = StringIO(string_header)
        fake_file_params = StringIO(string_params)
            
        df_header = pd.read_csv(fake_file_header, sep = '\t')    
        df_params = pd.read_csv(fake_file_params, sep = '=', header = None, names = ['parameter', 'value']).set_index('parameter')   
       
        comments_keys = df_header.set_index('Format').loc['Comment'].values[0].split('_')
        date_time = df_header.set_index('Format').loc['Date'].values[0]
                
        
        # define the general info parameters and values
        parameters_general_info = [
            "[SINGLE REFLECTANCE MEASUREMENT]",
            "authors",
            "date_time",
            "comment",
            ]
        
        values_general_info = [
            ' ',
            authors_names,
            date_time,
            comment,
        ]
        

        # define the device parameters and values
        if db:
            parameters_device = RS_info_templates.device_info
            comments_parameters = DB.get_db_config()['comments'][device_ID]

            db_config_keys = [x for x in DB.get_db_config().keys() if x not in ['colorimetry','comments','databases','devices']]
            db_config_dicts = [DB.get_db_config()[x] for x in db_config_keys]
            all_dicts = {}

            for db_config_dict in db_config_dicts:
                all_dicts.update(db_config_dict)

            
            comments_values = [all_dicts[x] if x in all_dicts.keys() else x for x in comments_keys]
            
                    
            values_device = [
                " ",
                device_ID,
                device_brand,
                device_model,
                "measurement_mode",
                "zoom",
                "iris",
                geometry,
                "lamp",
                "filter_ill",
                fiber_ill,
                fiber_coll,
                "distance_ill_mm",
                "distance_coll_mm",
                specular_component,             
                white_standard,
            ]

        else:
            parameters_device = list(df_params.index)[1:]
            values_device = list(df_params.values.flatten())[1:]


        # define the colorimetric parameters and values
        parameters_colorimetry = ['[COLORIMETRIC INFO]','illuminant','observer']
        values_colorimetry = [' ', illuminant, observer]


        # create df_info to be saved

        if db == False:   
                       
            if "_" in stemName:
                meas_id = stemName.split('_')[0]
            else:
                meas_id = stemName

            info_parameters = parameters_general_info + ['meas_id'] + parameters_device + parameters_colorimetry
            info_values = values_general_info + [meas_id] + values_device + values_colorimetry

        else:

            # retrieve info from filename
            info = (file_path.name).split('_')
            project_id = info[0]
            object_id = info[1]
            meas_nb = info[2]
            group = info[3]
            group_description = info[4]

            meas_id = f'RS.{object_id}.{meas_nb}'

            db_projects = db_projects.set_index('project_id')

            if project_id in db_projects.index:                
                values_project = [' ', project_id] + list(db_projects.loc[project_id].values)

            else:
                print(f'Project ID "{project_id}" not registered ! If you want to use the databases, please first register the projects and objects in the databases using the function add_project() and add_object().')
                return None
            
            db_objects = db_objects.set_index('object_id')
            if object_id in db_objects.index:
                values_object = [' ', object_id] + list(db_objects.loc[object_id].values)

            else:
                print(f'Object ID "{object_id}" not registered. If you want to use the databases, please first register the projects and objects in the databases using the function add_project() and add_object().')
                return None

            integration_time = int(float((df_params.loc['It']['value']).replace(',','.')))
            average = int(float(df_params.loc['Aver']['value']))            
            measurements_N = ''  # is defined at the end of the function      
            spot_size = ''      

            values_analyses = [
                " ",
                meas_id,
                group,
                group_description,
                spot_size,
                background,                
                integration_time,
                average,                
                measurements_N,              
            ]            
            

            info_parameters = parameters_general_info + ["[PROJECT INFO]", "project_id"] + list(db_projects.columns) + ["[OBJECT INFO]", "object_id"] + list(db_objects.columns) + parameters_device + RS_info_templates.analysis_info + parameters_colorimetry
            
            info_values = values_general_info + values_project + values_object + values_device + values_analyses + values_colorimetry


        dict_info = dict(zip(info_parameters,info_values))
        df_info = pd.DataFrame.from_dict(dict_info,orient='index', columns=['value'])
        df_info.index.name = 'parameter'

        # Fill in some of the info values
        
        if db:
            for param in parameters_device[2:] + ['spot_size_mm', 'background']:
                print(param)
                if param in DB.get_db_config()['comments'][device_ID]:
                    pass
                elif param in DB.get_db_config()['devices'][device_ID].keys():
                    df_info.loc[param] = DB.get_db_config()['devices'][device_ID][param]
                elif param in DB.get_db_config()['colorimetry'].keys():
                    df_info.loc[param] = DB.get_db_config()['colorimetry'][param]
                else:
                    df_info.loc[param] = 'undefined'
            
            for parameter,value in zip(comments_parameters, comments_values):
                df_info.loc[parameter] = value
        
        
        ####### PROCESS THE SPECTRAL DATA ########

        # retrieve the spectral data     
        lookfor_data = '[DATA]'
        string_rawdata = '\n'.join([ x.strip()[:-1] for x in f[f.index(lookfor_data)+len(lookfor_data):].splitlines()[1:]])   
            
        fake_file_rawdata = StringIO(string_rawdata)        
        df_rawdata = pd.read_csv(fake_file_rawdata, sep = '\t', skipfooter = 2, engine = 'python') 
           
        
        if df_rawdata.shape[1] > 2:
            df_rawdata.index.name = 'wavelength_nm'

        else:
            df_rawdata.columns = ['wavelength_nm',meas_id]
            df_rawdata = df_rawdata.set_index('wavelength_nm')
            

        
        # whether to interpolate the spectral data
        if interpolation_wl == 'none':
            wanted_wl = df_rawdata.index

        elif isinstance(interpolation_wl, tuple):
            wanted_wl = pd.Index(np.arange(interpolation_wl[0],interpolation_wl[1],interpolation_wl[2]), name='wavelength_nm')
        
        else:
            print(f"The '{interpolation_wl}' value that you entered is not valid. Enter either 'none' if you don't want any interpolation or a tuple of three values (start_wl, end_wl, step).")
            return

        df_sp = pd.DataFrame(data = sip.interp1d(df_rawdata.index, df_rawdata, axis = 0)(wanted_wl),
                            index = wanted_wl,
                            columns = df_rawdata.columns)
        
        # rounding the spectral data
        if rounding_sp == 'none':
            df_sp = df_sp/100
        elif isinstance(rounding_sp,int):
            df_sp = np.round(df_sp/100,rounding_sp)  
        else:
            print(f"The value '{rounding_sp}' you entered is not valid. Please enter a positive integer.")
            return  

        print(df_sp)  


        ####### CONVERT THE REFLECTANCE VALUES TO COLORIMETRIC VALUES ########
        
        sd = [colour.SpectralDistribution(x,df_sp.index) for x in df_sp.T.values]                 

        XYZ = [colour.sd_to_XYZ(x,cmfs,illuminant=illuminant_SDS) for x in sd]        
        xy = [np.round(colour.XYZ_to_xy(x),4) for x in XYZ]        
        Lab = [np.round(colour.XYZ_to_Lab(x/100, illuminant_CCS),3) for x in XYZ]        
        LCh = [np.round(colour.Lab_to_LCHab(x),3) for x in Lab]        
        values_cielab = [[list(x)+list(y)+list(z[1:])][0] for x,y,z in zip(xy,Lab,LCh)]

        dict_cielab = dict(zip(df_sp.columns,values_cielab))
        df_cielab = pd.DataFrame.from_dict(dict_cielab,orient='index', columns=['x','y','L*','a*','b*','C*','h']).T
        df_cielab.index.name = 'coordinates'


        # add a new row 'value' at the top
        df_value = pd.DataFrame(df_sp.shape[1] * ['value'], columns=['value']).T
        df_value.index.name = 'wavelength_nm'            
        df_value.columns = df_sp.columns
        df_sp = pd.concat([df_value, df_sp])

        df_value.index.name = 'coordinates'  
        df_cielab = pd.concat([df_value, df_cielab])


        # define the number of measurements
        df_info.loc['measurements_N'] = len(df_cielab.columns)

        # reset the index columns for db_projects and db_objects
        if db:
            db_objects = db_objects.reset_index()
            db_projects = db_projects.reset_index()
    
        # define the output filename
        if filenaming == 'none':
            filename = stemName

        elif filenaming == 'auto':
            group = stemName.split('_')[2]
            group_description = stemName.split('_')[3]
            object_type = df_info.loc['object_type']['value']
            date = pd.to_datetime(date_time).date()
            filename = f'{project_id}_{meas_id}_{group}_{group_description}_{object_type}_{date}'

        elif isinstance(filenaming, list):

            if 'date' in filenaming:
                new_df_info = df_info.copy()
                new_df_info.loc['date'] = str(df_info.loc['date_time']['value'].date())                    

                filename = "_".join([new_df_info.loc[x]['value'].split("_")[0] if "_" in new_df_info.loc[x]['value'] else new_df_info.loc[x]['value'] for x in filenaming])                    

            else:                                  
                filename = "_".join([df_info.loc[x]['value'].split("_")[0] if "_" in df_info.loc[x]['value'] else df_info.loc[x]['value'] for x in filenaming])
               
               
        # export the dataframes to an excel file
        with pd.ExcelWriter(Path(folder) / f'{filename}.xlsx') as writer:

            df_info.to_excel(writer, sheet_name='info', index=True)
            df_cielab.to_excel(writer, sheet_name="CIELAB", index=True)            
            df_sp.to_excel(writer, sheet_name="spectra", index=True)

                    
        ###### DELETE FILE #######        
            
        if delete_files:                      
            [os.remove(file) for file in raw_files]
            
        print(f'{raw_file} has been successfully processed !')

        





