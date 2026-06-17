import os
import sys
import pandas as pd
import numpy as np
import colour
from math import pi
from pathlib import Path
from typing import Optional, Union
import scipy.interpolate as sip
from scipy.interpolate import RegularGridInterpolator
from io import StringIO
from datetime import datetime
import msdb
import specdal as spd

from . import RS_info_templates
from . import config
from . import utils


def RS_Avt(
    files: list, 
    device_ID:Optional[str] = 'default', 
    db:Optional[bool] = 'default',   
    preconfig:Optional[bool] = 'default',  
    interpolation_wl:Optional[tuple] ='default', 
    observer:Optional[str] = '10deg', 
    illuminant:Optional[str] = 'D65',
    rounding:Union[tuple,int] = 'none',    
    authors:Optional[str] = 'XX', 
    organization:Optional[str] = 'XX', 
    comment:Optional[str] = '',
    sep_rawfilename_info:Optional[str] = '_',
    sep_comment_info:Optional[str] = '_',
    white_standard:Optional[bool] = 'undefined', 
    background:Optional[str] = 'black',
    spot_size:Optional[float] = 'unknown',
    save:Optional[bool] = True,
    folder:Optional[str] = '.',
    filenaming:Optional[str] = 'none',
    delete_files:Optional[bool] = False, 
    return_data:Optional[bool] = False, 
    ):
    
    # define the illuminant value
    illuminant_SDS, illuminant_CCS = utils.get_illuminant(illuminant, observer, db)
    
    # define the color matching functions
    cmfs = utils.get_cmfs(observer, db)

    # define the rounding values
    rounding_cl, rounding_sp = utils.define_rounding(rounding)  

    # retrieve the configuration info
    
    config_info = config.get_config_info()
    device_info = []
    filenaming_keys = []
    comment_keys = []
    
    if db:        
        devices_info = config_info['devices'] 
        comments_info = config_info['comments'] 
        filenamings_info = config_info['filenaming']

        # check whether database files have been created
        if len(config_info['databases']) == 0:
            return 'Databases have not been created. Please, create databases by running the function "create_DB" from the reflectance package.'
            
        else:             
            db_name = config_info['databases']['db_name'] 
            db = msdb.DB(db_name)
            db_projects = db.get_projects()
            db_objects = db.get_objects()
            db_lamps = db.get_lamps().set_index('ID')
            db_fibers = db.get_fibers().set_index('ID')
            db_filters = db.get_filters().set_index('ID')
              

        if device_ID in devices_info.keys():
            device_info = devices_info[device_ID]
        else:            
            print(f'The device info related to the device you entered ({device_ID}) cannot be found in the config_info.json file.')

        
        if device_ID in comments_info.keys():
            comment_keys = comments_info[device_ID]
        else:            
            print(f'The comment info related to the device you entered ({device_ID}) cannot be found in the config_info.json file.')

        
        if device_ID in filenamings_info.keys():
            filenaming_keys = filenamings_info[device_ID]
            filenaming_keys_raw = filenaming_keys['raw']
            filenaming_keys_interim = filenaming_keys['interim']
        else:            
            print(f'The filenaming info related to the device you entered ({device_ID}) cannot be found in the config_info.json file.')

    
    # get the raw files
    raw_files = [Path(file) for file in files if '.txt' in Path(file).name]

    
    
    ####### PROCESS RAW FILES ########

    for raw_file in raw_files:

        ####### DEFINE FILENAME ########
        
        file_path = Path(raw_file) 
        stemName = file_path.stem   

        if sep_rawfilename_info in stemName:     
            info_raw_filename = stemName.split(sep_rawfilename_info)
        
        else:            
            info_raw_filename = stemName


        ####### OPEN RAW FILE ########

        file_content = open(raw_file).read()
        file_content = file_content.replace(',', '.')
        lookfor = 'Wave'        
        parameters = file_content[:file_content.index(lookfor)].splitlines() 
        
        
        ####### RETRIEVE COMMENT VALUES ########
        
        comment_values = parameters[0]

        if sep_comment_info in comment_values:        
            comment_values = comment_values.split(sep_comment_info)        
        
        
        ####### RETRIEVE ANALYTICAL PARAMETERS PROVIDED BY THE DEVICE ########

        dic_parameters = {}

        for i in parameters[1:]:
            dic_parameters[i.split(':')[0]]=[i.split(':')[1]]
                
        df_parameters = pd.DataFrame.from_dict(dic_parameters).T    

        df_parameters.columns = ['value']            
        df_parameters.index.names = ['parameter'] 

        df_parameters = df_parameters.rename(index={'Date':'date_time'}) 
        
        
        ####### PROCESS SPECTRAL DATA ########

        # retrieve wavelengths and spectral values
        wavelengths = [float(x.split(';')[0]) for x in file_content[file_content.index(lookfor):].splitlines()[3:-1]]
        sp_raw = [float(x.split(';')[-1]) / 100 for x in file_content[file_content.index(lookfor):].splitlines()[3:-1]]

        # interpolate spectral values for colorimetric calculation
        wl_visible = np.arange(400,800,1)
        sp_color = sip.interp1d(wavelengths, sp_raw)(wl_visible)

        # interpolate spectral values
        if isinstance(interpolation_wl,(list, tuple)) :
            wanted_wl = np.arange(*interpolation_wl)
            wanted_sp = sip.interp1d(wavelengths, sp_raw)(wanted_wl)

        elif interpolation_wl == 'standard':
            wanted_wl = np.arange(int(wavelengths[0])+1, int(wavelengths[-1]))
            wanted_sp = sip.interp1d(wavelengths, sp_raw)(wanted_wl)

        else:
            wanted_wl = wavelengths
            wanted_sp = sp_raw

        # create spectral dataframe
        df_sp = pd.DataFrame({'wavelength_nm':wanted_wl, 'reflectance':wanted_sp})
        df_sp = df_sp.set_index('wavelength_nm')   

        
        # round the spectral values
        if isinstance(rounding_sp, int):
            if rounding_sp == 0:
                df_sp = df_sp.astype('int64')

            else:
                df_sp = np.round(df_sp, rounding_sp)

        # add a new row 'value' at the top
        df_value = pd.DataFrame(df_sp.shape[1] * ['nominal'], columns=['data_type']).T
        df_value.index.name = 'wavelength_nm'            
        df_value.columns = df_sp.columns
        df_sp = pd.concat([df_value, df_sp])


        ####### PROCESS THE COLORIMETRIC DATA ########

        
        df_cielab = utils.compute_cie_coordinates(
            wavelengths=wl_visible,
            reflectance=sp_color,
            coordinates=['x','y','L*','a*','b*','C*','h'],
            illuminant=illuminant,
            observer=observer,
            db=db,   
            rounding=rounding_cl         
            )
        

        # add a new row 'value' at the top
        df_value = pd.DataFrame(df_cielab.shape[1] * ['nominal'], columns=['data_type']).T
        df_value.index.name = 'coordinates'            
        df_value.columns = df_cielab.columns
        df_cielab = pd.concat([df_value, df_cielab])
        
                  
        
        ####### RETRIEVE GENERAL INFO ########

        if db:

            if 'date' in filenaming_keys_raw:
                date_time_analysis = info_raw_filename[filenaming_keys_raw.index('date')]

            elif 'date' in comment_keys:
                date_time_analysis = comment_values[comment_keys.index('date')]

            else:
                date_time_analysis = 'undefined'
                  
        else:
            date_time_analysis = 'undefined'
        
        date_time_processing = datetime.now().replace(microsecond=0)        
        
        values_general_info = [
            'SINGLE REFLECTANCE MEASUREMENT',
            authors,
            organization,
            date_time_analysis,
            date_time_processing,
            comment,
        ]      


        ####### RETRIEVE OBJECT INFO ########

        if not db:
            object_id = 'unknown'
        
        elif len(comment_keys) == 0 and len(filenaming_keys) == 0:
            object_id = 'unknown'
            
        elif 'object_id' in comment_keys:
            object_id = comment_values[comment_keys.index('object_id')]

        elif 'raw' in filenaming_keys and 'object_id' in filenaming_keys['raw']:     
            object_id = info_raw_filename[filenaming_keys['raw'].index('object_id')]

        else:
            print('The object_id number cannot be found in the rawfilename or the comment values. The object ID is set to "unknown".')
            object_id = 'unknown'

    
        if object_id == 'unknown':   
            params_object = RS_info_templates.object_info
            values_object = (len(params_object)-1) * ['unknown'] 
        
        elif object_id in db_objects['object_id'].values:

            object_info = db_objects.query(f'object_id == "{object_id}"')

            params_object = list(object_info.columns)
            values_object = list(object_info.values[0])                  
          
        if '[OBJECT INFO]' not in params_object:
            params_object = ['[OBJECT INFO]'] + params_object

        
        
        ####### RETRIEVE ANALYSIS INFO ########

        average = int(float(df_parameters.loc['Averaging Nr. [scans]','value']))
        integration_time = float(df_parameters.loc['Integration time [ms]','value'])
        smoothing = int(float(df_parameters.loc['Smoothing Nr. [pixels]','value']))

        if db:        

            if 'spot_group' in filenaming_keys_raw:
                spot_group = info_raw_filename[filenaming_keys_raw.index('spot_group')]

            elif 'spot_group' in comment_keys:
                spot_group = comment_values[comment_keys.index('spot_group')]

            else:
                spot_group = 'undefined'


            if 'spot_description' in filenaming_keys_raw:
                spot_description = info_raw_filename[filenaming_keys_raw.index('spot_description')]

            elif 'spot_description' in comment_keys:
                spot_description = comment_values[comment_keys.index('spot_description')]

            else:
                spot_description = 'undefined'  


            if 'white_standard' in filenaming_keys_raw:
                white_standard = info_raw_filename[filenaming_keys_raw.index('white_standard')]

            elif 'white_standard' in comment_keys:
                white_standard = comment_values[comment_keys.index('white_standard')]

            else:
                white_standard = 'undefined'  

        else:
            spot_group = 'undefined'
            spot_description = 'undefined'
            white_standard = 'undefined'


        white_standard = utils.get_white_standard(white_standard, db, device_id=device_ID)    
                       

        values_analysis = [            
            " ",    # meas_id , value assigned below
            spot_group,
            spot_description,            
            background,
            integration_time,
            int(average),
            smoothing,
            1,   # measurements_N
            white_standard,
            interpolation_wl
        ]

        ####### RETRIEVE DEVICE INFO ########  


        if len(device_info) > 0:            

            device_id = device_ID
            device_type = device_info['device_type']
            brand = device_info['brand']
            model = device_info['model']
            software_version = device_info['software_version']
            device_params = device_info['device_params']

        else:
            device_id = df_parameters.loc['Data measured with spectrometer [name]','value']
            device_type = 'spectrometer'
            brand = 'Avantes'
            model = 'unknown'
            software_version = 'unknown'            
            device_params = 'none'


        values_device = [            
            device_id,
            device_type,
            brand,
            model,
            software_version,
            device_params
        ]
        
        
        ####### RETRIEVE SYSTEM INFO ########
        
        
        if db:
            if len(comment_keys) == 0 and len(filenaming_keys) == 0:
                system_id = 'unknown'
                
            elif 'system_id' in  comment_keys:
                system_id = comment_values[comment_keys.index('system_id')]

            elif 'system_id' in filenaming_keys['raw']:
                system_id = info_raw_filename[filenaming_keys['raw'].index('system_id')]

            else:
                print('The system id number cannot be found in the rawfilename or the comment values. System values defined to "unknown".')
                system_id = 'unknown'
                        
            
            if system_id in config_info['systems'].keys():

                system_info = config_info['systems'][system_id]

                system_name = system_info['system_name']
                constructor = system_info['constructor']
                geometry = system_info['geometry']
                filter_ill = system_info['filter_ill']
                filter_coll = system_info['filter_coll']
                fiber_ill = system_info['fiber_ill']
                fiber_coll = system_info['fiber_coll']
                lamp = system_info['lamp']
                specular_component = system_info['specular_component']
                spot_size_config = system_info['spot_size_mm']
                system_params = system_info['system_params']

            else:
                system_name = constructor = 'unknown'
                geometry = lamp = filter_ill = filter_coll = fiber_ill = fiber_coll = specular_component = spot_size = spot_size_config = 'unknown'
                system_params = 'none'  

            if 'spot_size_mm' in filenaming_keys_raw:
                spot_size = info_raw_filename[filenaming_keys_raw.index('spot_size_mm')]

            elif 'spot_size_mm' in comment_keys:
                spot_size = comment_values[comment_keys.index('spot_size_mm')]

            elif spot_size != spot_size_config:            
                pass

            else:
                spot_size = spot_size_config  


            if 'lamp' in filenaming_keys_raw:
                lamp = info_raw_filename[filenaming_keys_raw.index('lamp')]

            elif 'lamp' in comment_keys:
                lamp = comment_values[comment_keys.index('lamp')]

            elif lamp != 'unknown':
                pass
                
            else:
                lamp = 'undefined' 

            if db:
                if lamp in db_lamps.index:
                    lamp_description = db_lamps.loc[lamp,'description']
                    lamp = f'{lamp}_{lamp_description}'



            if 'filter_ill' in filenaming_keys_raw:
                fiber_ill = info_raw_filename[filenaming_keys_raw.index('filter_ill')]

            elif 'filter_ill' in comment_keys:
                filter_ill = comment_values[comment_keys.index('filter_ill')]

            elif filter_ill != ' ':
                pass
                
            else:
                filter_ill = 'undefined' 

            if db:
                if filter_ill in db_filters.index:
                    filter_ill_description = db_filters.loc[filter_ill,'description']
                    filter_ill = f'{filter_ill}_{filter_ill_description}'
            
            
            
            if 'fiber_ill' in filenaming_keys_raw:
                fiber_ill = info_raw_filename[filenaming_keys_raw.index('fiber_ill')]

            elif 'fiber_ill' in comment_keys:
                fiber_ill = comment_values[comment_keys.index('fiber_ill')]

            elif fiber_ill != ' ':
                pass
                
            else:
                fiber_ill = 'undefined' 


            if db:
                if fiber_ill in db_fibers.index:
                    fiber_ill_description = db_fibers.loc[fiber_ill,'description']
                    fiber_ill = f'{fiber_ill}_{fiber_ill_description}'


            if 'fiber_coll' in filenaming_keys_raw:
                fiber_coll = info_raw_filename[filenaming_keys_raw.index('fiber_coll')]

            elif 'fiber_coll' in comment_keys:
                fiber_coll = comment_values[comment_keys.index('fiber_coll')]

            elif fiber_coll != ' ':
                pass
                
            else:
                fiber_coll = 'undefined'

            if db:
                if fiber_coll in db_fibers.index:
                    fiber_coll_description = db_fibers.loc[fiber_coll,'description']
                    fiber_coll = f'{fiber_coll}_{fiber_coll_description}'

        else:
            system_id = system_name = constructor = 'undefined'            
            geometry = lamp = spot_size = filter_ill = filter_coll = fiber_ill = fiber_coll = specular_component = 'undefined'
            system_params = {}


        values_system = [            
            system_id,
            system_name,
            constructor,
            geometry,
            lamp,
            spot_size,
            filter_ill,
            filter_coll,
            fiber_ill,
            fiber_coll,
            specular_component,
            system_params
        ]

        
        ####### RETRIEVE PROJECT INFO ########

        if not db:
            project_id = 'unknown'
        
        elif len(comment_keys) == 0 and len(filenaming_keys) == 0:
            project_id = 'unknown'
            
        elif 'project_id' in comment_keys:
            project_id = comment_values[comment_keys.index('project_id')]

        elif 'raw' in filenaming_keys and 'project_id' in filenaming_keys['raw']:     
            project_id = info_raw_filename[filenaming_keys['raw'].index('project_id')]

        else:
            print('The project_id number cannot be found in the rawfilename or the comment values. The project ID is set to "unknown".')
            project_id = 'unknown'
            
                
        if project_id == 'unknown':
            params_project = RS_info_templates.project_info
            values_project = (len(params_project)-1) * ['unknown']  
        
        elif project_id in db_projects['project_id'].values:

            project_info = db_projects.query(f'project_id == "{project_id}"')

            params_project = list(project_info.columns)
            values_project = list(project_info.values[0])      

        if '[PROJECT INFO]' not in params_project:
            params_project = ['[PROJECT INFO]'] + params_project
            
                  
        ####### ASSIGN MEAS_ID ########

        if db:
            if 'measurement_Nb' in filenaming_keys_raw:
                measurement_Nb = info_raw_filename[filenaming_keys_raw.index('measurement_Nb')]

            elif 'measurement_Nb' in comment_keys:
                measurement_Nb = comment_values[comment_keys.index('measurement_Nb')]

            else:
                measurement_Nb = 'XX'

            meas_id = f'RS.{object_id}.{measurement_Nb}'

        else:
            if '_' in stemName:
                meas_id = stemName.split('_')[0]

            else:
                meas_id = stemName 
                
        
        values_analysis[0] = meas_id
        
        
        ####### RETRIEVE COLORIMETRIC INFO ########

        values_colorimetry = [illuminant, observer]
        

        ####### CREATE INFO DATAFRAME ########         

        """
        params_general_info = RS_info_templates.general_info
        params_analysis = RS_info_templates.analysis_info
        params_colorimetry = RS_info_templates.colorimetric_info
        params_system = RS_info_templates.system_info
        params_device = RS_info_templates.device_info
        
        

        info_parameters = params_general_info + params_project + params_object + params_system + params_device + params_analysis + params_colorimetry        
        
        info_values = values_general_info + values_project + values_object + values_system + values_device + values_analysis + values_colorimetry

        dict_info = dict(zip(info_parameters,info_values))
        df_info = pd.DataFrame.from_dict(dict_info,orient='index', columns=['value'])
        df_info.index.name = 'parameter'
        """
        
        df_info = utils.create_df_info(
            db=db,
            info_general=values_general_info,
            info_project=values_project,
            info_object=values_object,
            info_system=values_system,
            info_device=values_device,
            info_analysis=values_analysis,
            info_colorimetry=values_colorimetry
            )


        ####### RENAME COLUMNS DATAFRAMES ######## 
      
        df_cielab.columns = [meas_id]
        df_sp.columns = [meas_id]
                
        ####### SAVE DATA ########

        if save:

            if db:
                # define the output filename
                if filenaming == 'none':
                    filename = stemName

                elif filenaming == 'auto':                
                    object_type = df_info.loc['object_type']['value']
                    date = date_time_analysis 
                    filename = f'{project_id}_{meas_id}_{spot_group}_{spot_description}_{object_type}_{date}_{device_ID}'

                elif isinstance(filenaming, list):

                    if 'date' in filenaming:
                        new_df_info = df_info.copy()
                        new_df_info.loc['date'] = date_time_analysis #str(df_info.loc['date_time']['value'].date())                    

                        filename = "_".join([new_df_info.loc[x]['value'].split("_")[0] if "_" in new_df_info.loc[x]['value'] else new_df_info.loc[x]['value'] for x in filenaming])                    

                    else:                                  
                        filename = "_".join([df_info.loc[x]['value'].split("_")[0] if "_" in df_info.loc[x]['value'] else df_info.loc[x]['value'] for x in filenaming])

            else:
                filename = stemName
            

            # export the dataframes to an excel file
            with pd.ExcelWriter(Path(folder) / f'{filename}.xlsx') as writer:

                df_info.to_excel(writer, sheet_name='info', index=True)
                df_cielab.to_excel(writer, sheet_name="CIELAB", index=True)            
                df_sp.to_excel(writer, sheet_name="spectra", index=True)

                    
        ###### DELETE FILE #######        
            
        if delete_files:                      
            os.remove(raw_file)

        
        ####### RETURN DATA ########

        if return_data:
            return [df_cielab,df_sp]
            
        print(f'{raw_file} has been successfully processed !')


def RS_Tidas(
    files: list, 
    device_ID:Optional[str] = 'default', 
    db:Optional[bool] = 'default', 
    filenaming:Optional[str] = 'default', 
    folder:Optional[str] = '.',  
    comment:Optional[str] = '', 
    interpolation_wl:Optional[tuple] ='default', 
    rounding:Optional[int] = 'none',  
    authors:Optional[str] = 'XX', 
    white_standard:Optional[bool] = 'default', 
    observer:Optional[str] = 'default', 
    illuminant:Optional[str] = 'default', 
    background:Optional[str] = 'black', 
    delete_files:Optional[bool] = True, 
    return_filename:Optional[bool] = True):

    # check whether the objects and projects databases have been created    
    config_info = config.get_config_info()
    
    if db:    
        
        if len(config_info['databases']) == 0:
            return 'Databases have not been created. Please, create databases by running the function "create_DB" from the reflectance package.'
        
        else:             
            db_name = config_info['databases']['db_name'] 
            db_rs = msdb.DB(db_name)
            db_projects = db_rs.get_projects()
            db_objects = db_rs.get_objects()
            db_white_standards = db_rs.get_white_standards().set_index('ID')
            db_devices = db_rs.get_devices().set_index('ID')

            # remove the column 'project_id'
            if 'project_id' in db_objects.columns:
                db_objects = db_objects.drop('project_id', axis=1)
    
    else:
        filenaming = 'none' # override whatever input value for filenaming
     

    # define the illuminant value
    if illuminant == 'default' and db == True:
        if len(config.get_colorimetry_info()) == 0:
            illuminant = 'D65'
        else:
            illuminant = config.get_colorimetry_info().loc['illuminant']['value']

    elif illuminant == 'default' and db == False:
        illuminant = 'D65'

    
    # define the observer
    if observer == 'default' and db == True:
        if len(config.get_colorimetry_info()) == 0:
            observer = '10deg'
        else:
            observer = config.get_colorimetry_info().loc['observer']['value']

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
        

    # retrieve the white standard info
    if white_standard == 'default' and db == True:
        if len(config.get_colorimetry_info()) == 0:
            white_standard = 'undefined'
        else:
            white_standard_ID = config.get_config_info()['devices'][device_ID]['white_standard']
            white_standard_description = db_white_standards.loc[white_standard_ID,'description']
            white_standard = f'{white_standard_ID}_{white_standard_description}'

    elif white_standard == 'default' and db == False:
        white_standard = 'undefined'
    
    
    # define the host organization
    if len(config_info['institution']) == 0:
        host_organization = 'undefined'
            
    else:
        host_organization = config_info['institution']['name']

    
    # retrieve the general device info

    if db:
        config_devices = config_info['devices']
        config_device = config_devices[device_ID]       

        device_type = config_device['device_type']
        device_model = config_device['model']
        device_brand = config_device['brand']

        if len(config_device['system_ID']) > 0 and config_device['system_ID'] in config_info['systems'].keys():
            geometry = config_info['systems'][config_device['system_ID']]['geometry']
            fiber_ill = config_info['systems'][config_device['system_ID']]['fiber_ill']
            fiber_coll = config_info['systems'][config_device['system_ID']]['fiber_coll']
            specular_component = config_info['systems'][config_device['system_ID']]['specular_component']

        else:
            geometry = 'unknown'
            fiber_ill = 'unknown'
            fiber_coll = 'unknown'
            specular_component = 'unknown'


    # define the rounding values
    rounding_cl, rounding_sp = utils.define_rounding(rounding)
    

    
    # get the raw files with the measurement data (c01)
    #print(files)
    raw_txt_files = [Path(file) for file in files if '.txt' in Path(file).name and 'c01_' in Path(file).name]    
    #print(raw_txt_files)
        
    #### PROCESS EACH RAW FILE ####
    
    for raw_file in raw_txt_files:
        
        # get the raw files (to be deleted at the end)
        raw_files = [x for x in files if str(raw_file.stem) in Path(x).name]
    
        # define filenames
        file_path = Path(raw_file) 
        stemName = file_path.stem
        
        # open the raw file    
        f = open(file_path, encoding="ISO-8859-1").read()  
        
          
        ####### RETRIEVE THE INFO ########    
          
        lookfor_value = '[LOGIN]' 

        # retrieve header info (before the lookfor_value)           
        string_header = '\n'.join([ x.strip()[:] for x in f[:f.index(lookfor_value)].splitlines()[1:]])
        fake_file_header = StringIO(string_header)
        df_header = pd.read_csv(fake_file_header, sep = '\t')

        # retrieve analytical info (after the lookfor_value)
        string_params = '\n'.join([ x.strip()[:] for x in f[f.index(lookfor_value)+len(lookfor_value):f.index('[DIO]')].splitlines()[1:]])
        fake_file_params = StringIO(string_params)            
        df_params = pd.read_csv(fake_file_params, sep = '=', header = None, names = ['parameter', 'value']).set_index('parameter')   
        
        # retrieve the comment line info
        comment_info = df_header.set_index('Format').loc['Comment'].values[0].split('_')
        if db:
            comment_keys = config_info['comments'][device_ID]
                
        # retrieve the datetime info
        date_time_analysis = df_header.set_index('Format').loc['Date'].values[0]
        date_time_analysis = datetime.strptime(date_time_analysis, "%d/%m/%Y %H:%M:%S")        
        date_time_processing = datetime.now()

        # retrieve analytical info
        integration_time = int(float((df_params.loc['It']['value']).replace(',','.')))
        average = int(float(df_params.loc['Aver']['value'])) 
        
        
        # retrieve the filename info
        if db == False:   
                       
            if "_" in stemName:
                meas_id = stemName.split('_')[0]
            else:
                meas_id = stemName

            group = 'undefined'
            group_description = 'undefined'
            
        else:            
            info = (file_path.name).split('_')
            project_id = info[0]
            object_id = info[1]
            meas_nb = info[2]
            group = info[3]
            group_description = info[4]

            meas_id = f'RS.{object_id}.{meas_nb}'
        

        # create an empty df_info
        
        parameters_general_info = RS_info_templates.general_info  
        parameters_device_info = RS_info_templates.device_info 
        parameters_analysis_info = RS_info_templates.analysis_info
        parameters_colorimetry_info = RS_info_templates.colorimetric_info     
        
        if db == False:

            parameters_project_info = RS_info_templates.project_info
            parameters_object_info = RS_info_templates.object_info                     
            
        else:
            parameters_project_info =  ["[PROJECT INFO]"] + list(db_projects.columns)
            parameters_object_info = ["[OBJECT INFO]"] + list(db_objects.columns)
            
            
        info_parameters = parameters_general_info + parameters_project_info + parameters_object_info + parameters_device_info + parameters_analysis_info + parameters_colorimetry_info

        df_info_empty = pd.DataFrame(index=info_parameters, columns=['value'])
        df_info_empty.index.name = 'parameter'
        df_info = df_info_empty.copy()
        
        # fill in general info
        values_general_info = [
            'SINGLE REFLECTANCE MEASUREMENT',
            authors,
            host_organization,
            date_time_analysis,
            date_time_processing,
            comment,
        ]

        df_info.loc[parameters_general_info,'value'] = values_general_info
                
        
        # fill in project info
        if db and project_id in db_projects['project_id'].values: 
            db_projects = db_projects.set_index('project_id')
            values_project_info = [' ',project_id] + list(db_projects.loc[project_id].values)
            
            df_info.loc[parameters_project_info,'value'] = values_project_info


        # fill in object info
        if db and object_id in db_objects['object_id'].values: 
            db_objects = db_objects.set_index('object_id')
            values_object_info = [' ',object_id] + list(db_objects.loc[object_id].values)
            
            df_info.loc[parameters_object_info,'value'] = values_object_info

        
        # fill in device info
                
        if db and device_ID in config_info['devices'].keys():
            
            device_description = db_devices.loc[device_ID]['description']
            df_info.loc['device_id','value'] = f'{device_ID}_{device_description}'
            
            device_info = config_info['devices'][device_ID]
            

            device_info_keys = (list(device_info.keys()))
            device_info_keys.remove('process_function')
        
            for device_info_key in device_info_keys:
                
                device_value = device_info[device_info_key]
                
                if isinstance(device_value, dict):                    
                    device_value = [device_value]

                df_info.loc[device_info_key,'value'] = device_value

        else:
            
            df_info.loc['device_type','value'] = 'Assembly'
            df_info.loc['brand','value'] = 'J&M Analytik AG'
            df_info.loc['model','value'] = f"{df_params.loc['Name']['value'].values[1]}_Serial#:{df_params.loc['Serial#','value']}"
            df_info.loc['software_version','value'] = f'TIDASDAQ3-{df_params.loc["SWVersion","value"]}'

                
        # fill in the analysis info       
        values_analysis_info = [
            "",
            meas_id, 
            group,
            group_description,
            "",                  # spot_size,
            background,
            integration_time,
            average,
            "unknown",           # smoothing_pixels
            "",                  # N_measurements, it is defined later
            white_standard,            
        ]
        

        df_info.loc[parameters_analysis_info,'value'] = values_analysis_info


        # fill in the colorimetric info
        values_colorimetry_info = [
            "",
            illuminant,
            observer
        ]
        df_info.loc[parameters_colorimetry_info,'value'] = values_colorimetry_info
        
        
        # fill in the comment info
        if db:
            for comment_key in comment_keys:                
                
                if comment_key in df_info.index:

                    if comment_key == 'white_standard':

                        white_standard_ID = comment_info[comment_keys.index(comment_key)]
                        white_standard_description = db_white_standards.loc[white_standard_ID,'description']
                        white_standard = f'{white_standard_ID}_{white_standard_description}'

                        df_info.loc[comment_key,'value'] = white_standard

                    else:

                        df_info.loc[comment_key,'value'] = comment_info[comment_keys.index(comment_key)]

                elif comment_key in df_info.loc['device_params','value'][0].keys():
                    current_dict = df_info.loc['device_params','value'][0]
                    current_dict[comment_key] = comment_info[comment_keys.index(comment_key)]

                    df_info.loc['device_params','value'] = [current_dict]

        
        # fill in info about software version
        df_info.loc['software_version', 'value'] = f'TIDASDAQ3-{df_params.loc["SWVersion","value"]}'
        
        # fill in info about lamp
        lamp_info = df_info.loc['lamp', 'value']
        if lamp_info in db_rs.get_lamps()['ID'].values:
            lamp_description = db_rs.get_lamps().query(f'ID == "{lamp_info}"')['description'].values[0]
            lamp_info = f'{lamp_info}_{lamp_description}'
            df_info.loc['lamp','value'] = lamp_info


        # fill in info about filter
        filter_info = df_info.loc['filter', 'value']
        if filter_info in db_rs.get_filters()['ID'].values:
            filter_description = db_rs.get_filters().query(f'ID == "{filter_info}"')['description'].values[0]
            filter_info = f'{filter_info}_{filter_description}'
            df_info.loc['filter','value'] = filter_info


        # fill in info about fiber
        fiber_ill_info = df_info.loc['fiber_ill', 'value']
        if fiber_ill_info in db_rs.get_fibers()['ID'].values:
            fiber_ill_description = db_rs.get_fibers().query(f'ID == "{fiber_ill_info}"')['description'].values[0]
            fiber_ill_info = f'{fiber_ill_info}_{fiber_ill_description}'
            df_info.loc['fiber_ill','value'] = fiber_ill_info


        fiber_coll_info = df_info.loc['fiber_coll', 'value']
        if fiber_coll_info in db_rs.get_fibers()['ID'].values:
            fiber_coll_description = db_rs.get_fibers().query(f'ID == "{fiber_coll_info}"')['description'].values[0]
            fiber_coll_info = f'{fiber_coll_info}_{fiber_coll_description}'
            df_info.loc['fiber_coll','value'] = fiber_coll_info
        

        
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
            interpolated_wl = pd.Index(np.arange(np.int32(wanted_wl[0])+1,np.int32(wanted_wl[-1])-1,1), name='wavelength_nm')

            df_sp = pd.DataFrame(data=sip.interp1d(df_rawdata.index, df_rawdata, axis=0)(wanted_wl),
                            index=wanted_wl,
                            columns=df_rawdata.columns).dropna(axis=0)
            
            df_sp_interpolated = pd.DataFrame(data=sip.interp1d(df_rawdata.index, df_rawdata, axis=0)(interpolated_wl),
                            index=interpolated_wl,
                            columns=df_rawdata.columns)
            
            df_sp_interpolated = df_sp_interpolated / 100

        elif isinstance(interpolation_wl, (tuple,list)):
            wanted_wl = pd.Index(np.arange(interpolation_wl[0],interpolation_wl[1],interpolation_wl[2]), name='wavelength_nm')

            df_sp = pd.DataFrame(data=sip.interp1d(df_rawdata.index, df_rawdata, axis=0)(wanted_wl),
                            index=wanted_wl,
                            columns=df_rawdata.columns)
            
            df_sp_interpolated = df_sp / 100
        
        else:
            print(f"The '{interpolation_wl}' value that you entered is not valid. Enter either 'none' if you don't want any interpolation or a tuple of three values (start_wl, end_wl, step).")
            return

        
        # rounding the spectral data
        if rounding_sp == 'none':
            df_sp = df_sp/100
        elif isinstance(rounding_sp,int):
            df_sp = np.round(df_sp/100,rounding_sp)  
        else:
            print(f"The value '{rounding_sp}' you entered is not valid. Please enter a positive integer.")
            return         


        ####### CONVERT THE REFLECTANCE VALUES TO COLORIMETRIC VALUES ########
        
        sd = [colour.SpectralDistribution(x,df_sp_interpolated.index) for x in df_sp_interpolated.T.values]
        
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
            date = pd.to_datetime(date_time_analysis).date()
            filename = f'{project_id}_{meas_id}_{group}_{group_description}_{object_type}_{date}'

        elif filenaming == 'default' and db == True:
            
            filename_parameters = config_info['filenaming'][device_ID]['interim']

            if 'date' in filename_parameters:
                new_df_info = df_info.copy()
                new_df_info.loc['date'] = str(df_info.loc['datetime_analysis']['value'].date())               
                
                filename = "_".join([new_df_info.loc[x]['value'].split("_")[0] if "_" in new_df_info.loc[x]['value'] else new_df_info.loc[x]['value'] for x in filename_parameters])

            else:
                filename = "_".join([df_info.loc[x]['value'].split("_")[0] if "_" in df_info.loc[x]['value'] else df_info.loc[x]['value'] for x in filename_parameters])

        elif filenaming == 'default' and db == False:
            filename = stemName

        elif isinstance(filenaming, list):

            if 'date' in filenaming:
                new_df_info = df_info.copy()               
                
                new_df_info.loc['date'] = str(df_info.loc['datetime_analysis']['value'].date())                    

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


def RS_ASD(
    raw_files: list,
    device_ID:Optional[str] = 'default', 
    db:Optional[bool] = 'default', 
    filenaming:Optional[str] = 'default',     
    folder:Optional[str] = '.',  
    comment:Optional[str] = '',
    splice_correction:Union[tuple, str] = ([1000,1800], 10), 
    interpolation_wl:Optional[tuple] = 'default', 
    average:Optional[int] = 50,
    rounding:Union[int, tuple, str] = (4,4),
    authors:Optional[str] = 'XX', 
    white_standard:Optional[bool] = 'default', 
    observer:Optional[str] = 'default', 
    illuminant:Optional[str] = 'default', 
    background:Optional[str] = 'black', 
    delete_files:Optional[bool] = True, 
    return_filename:Optional[bool] = True): 

    # check whether the objects and projects databases have been created    
    config_info = config.get_config_info()
    
    if db:    
        
        # check if the database files have been registered or created
        if len(config_info['databases']) == 0:
            return 'Databases have not been created. Please, create databases by running the function "create_DB" from the reflectance package.'
        
        else:             
            db_name = config_info['databases']['db_name'] 
            db_rs = msdb.DB(db_name)
            db_projects = db_rs.get_projects()
            db_objects = db_rs.get_objects()
            db_white_standards = db_rs.get_white_standards().set_index('ID')
            db_devices = db_rs.get_devices().set_index('ID')

            # remove the column 'project_id'
            if 'project_id' in db_objects.columns:
                db_objects = db_objects.drop('project_id', axis=1)

        # check if the device has been registered
        if device_ID.lower() == 'asd':
            device_type = 'Assembly'
            device_brand = 'Malvern Panalytica'
            device_model = 'ASD'
            geometry = '45:0'
            fiber_coll = 'undefined'
            fiber_ill = 'undefined'
            lamp = 'undefined'
            specular_component = 'SCE_excluded'
            

        elif device_ID in config_info['devices'].keys():
            device_info = config_info['devices'][device_ID]
            device_type = device_info['device_type']
            device_brand = device_info['brand']
            device_model = device_info['model']            
            geometry = device_info['geometry']
            fiber_coll = device_info['fiber_coll']
            fiber_ill = device_info['fiber_ill']
            lamp = device_info['lamp']
            specular_component = device_info['specular_component']
            white_standard = device_info['white_standard']

        # check if filenaming registered
        if device_ID in config_info['filenaming'].keys():
            filenaming_raw_keys = config_info['filenaming'][device_ID]['raw']
            filenaming_interim_keys = config_info['filenaming'][device_ID]['interim']



    
    else:
        filenaming = 'none' # override whatever input value for filenaming



    # retrieve the illuminant value
    if illuminant == 'default' and db == True:
        if len(config.get_colorimetry_info()) == 0:
            illuminant = 'D65'
        else:
            illuminant = config.get_colorimetry_info().loc['illuminant']['value']

    elif illuminant == 'default' and db == False:
        illuminant = 'D65'

    
    # retrieve the observer value
    if observer == 'default' and db == True:
        if len(config.get_colorimetry_info()) == 0:
            observer = '10deg'
        else:
            observer = config.get_colorimetry_info().loc['observer']['value']

    elif observer == 'default' and db == False:
        observer = '10deg'

    
    # define the rounding values    
    rounding_cl, rounding_sp = utils.define_rounding(rounding)


    # retrieve the white standard info
    if white_standard == 'default' and db == True:
        if len(config.get_colorimetry_info()) == 0:
            white_standard = 'undefined'
        else:
            white_standard_ID = config.get_config_info()['devices'][device_ID]['white_standard']
            white_standard_description = db_white_standards.loc[white_standard_ID,'description']
            white_standard = f'{white_standard_ID}_{white_standard_description}'

    elif white_standard == 'default' and db == False:
        white_standard = 'undefined'
    
    
    # define the host organization
    if len(config_info['institution']) == 0:
        host_organization = 'undefined'
            
    else:
        host_organization = config_info['institution']['name']
               
    
    # process each raw files
    for raw_file in raw_files:

        # read raw file
        file_path = Path(raw_file)
        stemName = file_path.stem
        file_suffix = file_path.suffix

        
        if file_suffix == '.asd':

            df_data = spd.read(file_path)        
            df_sp_raw = df_data[0]
            df_parameters = df_data[1]

        elif file_suffix == '.txt':
            try:
                df_data = pd.read_csv(file_path, sep='\t')            
                df_sp_raw = df_data['Wavelength':].iloc[1:,:]

            except TypeError:
                df_data = pd.read_csv(file_path, sep=',')            
                df_sp_raw = df_data['Wavelength':].iloc[1:,:]
               
        

        ####### RETRIEVE THE INFO AND METADATA ########


        # retrieve the filename info
        if db == False:   
                       
            if "_" in stemName:
                meas_id = stemName.split('_')[0]
            else:
                meas_id = stemName

            project_id = 'undefined'
            object_id = 'undefined'
            group = 'undefined'
            group_description = 'undefined'
            
        else:                        
            info_raw_filename = stemName.split('_')
            dic_raw_filename = dict(zip(filenaming_raw_keys, info_raw_filename))

            if 'object_id' and 'measurement_Nb' in filenaming_raw_keys:
                               
                object_id = dic_raw_filename['object_id']
                meas_nb = dic_raw_filename['measurement_Nb']
                meas_id = f'RS.{object_id}.{meas_nb}'          

            else:
                pass  # think about something
                #meas_id = f'RS.{}'

            if 'project_id' in filenaming_raw_keys:
                project_id = dic_raw_filename['project_id']

            else:
                project_id = 'undefined'

                
        # retrieve parameters inside the raw file
        if file_suffix == '.asd':
            integration_time = df_parameters['integration_time']
            
            gps_time_analysis = df_parameters['gps_time_tgt']
            date_time_analysis = datetime.fromtimestamp(gps_time_analysis)
            #date_time_analysis = datetime.strftime(date_time_analysis, "%Y-%m-%d %H:%M:%S")

        elif file_suffix == '.txt':
            if "Integration Time" in df_data.index:                
                integration_time = int(df_data.loc['Integration Time'].values[0])

            if "Sample Count" in df_data.index:                
                average = int(df_data.loc['Sample Count'].values[0])

            if "Date Taken" in df_data.index:                
                date_time_analysis = (df_data.loc['Date Taken'].values[0]) 	
                try:
                    date_time_analysis = datetime.strptime(date_time_analysis, "%m/%d/%Y %I:%M:%S %p")
                except ValueError:
                    pass
                
         
        # retrieve datetime data processing        
        date_time_processing = datetime.now()      

                       
        # get the general info
        values_general_info = [
            'SINGLE REFLECTANCE MEASUREMENT',
            authors,
            host_organization,
            date_time_analysis,
            date_time_processing,
            comment,
        ]
 
        
        # get the project info
        if db and project_id in db_projects['project_id'].values: 
            db_projects = db_projects.set_index('project_id')
            values_project_info = [project_id] + list(db_projects.loc[project_id].values)

        else:
            values_project_info = []
            

        # get the object info
        if db and object_id in db_objects['object_id'].values: 
            db_objects = db_objects.set_index('object_id')
            values_object_info = [object_id] + list(db_objects.loc[object_id].values)
  
                
        # fill in device info
                
        if db and device_ID in config_info['devices'].keys():
            
            device_description = db_devices.loc[device_ID]['description']
            df_info.loc['device_id','value'] = f'{device_ID}_{device_description}'
            
            device_info = config_info['devices'][device_ID]
            

            device_info_keys = (list(device_info.keys()))
            device_info_keys.remove('process_function')
        
            for device_info_key in device_info_keys:
                
                device_value = device_info[device_info_key]
                
                if isinstance(device_value, dict):                    
                    device_value = [device_value]

                df_info.loc[device_info_key,'value'] = device_value

        else:
            
            df_info.loc['device_type','value'] = 'Assembly'
            df_info.loc['brand','value'] = 'Malvern Panalytical'
            df_info.loc['model','value'] = 'unknown'
            df_info.loc['software_version','value'] = 'unknown'

                
        # fill in the analysis info   

        if not db:
            group = 'undefined'

        else:
            if 'group' in filenaming_raw_keys:
                group = dic_raw_filename['group']

            else:
                group = 'undefined'

        
        if not db:
            group_description = 'undefined'

        else:
            if 'group_description' in filenaming_raw_keys:
                group_description = dic_raw_filename['group_description']

            else:
                group_description = 'undefined'

        
        values_analysis_info = [            
            meas_id, 
            group,
            group_description,
            "",                  # spot_size,
            background,
            integration_time,
            average,
            "unknown",           # smoothing_pixels
            "",                  # N_measurements, it is defined later
            white_standard,            
        ]        

        # get the colorimetric info
        values_colorimetry_info = [illuminant, observer]
        
        
       
        
        ####### PROCESS THE SPECTRAL DATA ########       

        if file_suffix == '.asd':

            wavelengths = df_sp_raw.index
            reference = df_sp_raw['ref_count']
            target = df_sp_raw['tgt_count']
            rs = target / reference

        elif file_suffix == '.txt':
            wavelengths = df_sp_raw.index.astype(int)
            rs = df_sp_raw.astype(float).iloc[:,0].values
        

        if interpolation_wl == 'none':

            df_sp = pd.DataFrame({'wavelength_nm':wavelengths, meas_id:rs})
            df_sp = df_sp.set_index('wavelength_nm')

            df_sp_interpolated = df_sp
        
        
        elif isinstance(interpolation_wl, (tuple,list)):
            wanted_wl = pd.Index(np.arange(interpolation_wl[0],interpolation_wl[1],interpolation_wl[2]), name='wavelength_nm')

            df_sp = pd.DataFrame(
                data=sip.interp1d(wavelengths, rs, axis=0)(wanted_wl),
                index=wanted_wl,
                columns=[meas_id])
            
            df_sp_interpolated = df_sp
        
        else:
            print(f"The '{interpolation_wl}' value that you entered is not valid. Enter either 'none' if you don't want any interpolation or a tuple of three values (start_wl, end_wl, step).")
            return        
        
        
        if splice_correction == 'none' or splice_correction == False:
            pass        
        
        else:
            corrected_sp = utils.compute_slice_correction(df_sp.index, df_sp[meas_id].values, splice_correction[0], splice_correction[1])


            df_sp = pd.DataFrame(
                data = corrected_sp,
                index = df_sp.index,
                columns=[meas_id]
            )


        if isinstance(rounding_sp, int):
            if rounding_sp == 0:
                df_sp = df_sp.astype('int64')

            else:
                df_sp = np.round(df_sp, rounding_sp)

        
        ####### PROCESS THE COLORIMETRIC DATA ########

        df_cielab = utils.compute_cie_coordinates(
            wavelengths=df_sp_interpolated.index,
            reflectance=df_sp_interpolated.iloc[:,0].values,
            coordinates=['x','y','L*','a*','b*','C*','h'],
            illuminant=illuminant,
            observer=observer,
            db=db,   
            rounding=rounding_cl         
            )
        
        # add a new row 'value' at the top
        df_value = pd.DataFrame(df_sp.shape[1] * ['nominal'], columns=['nominal']).T
        df_value.index.name = 'wavelength_nm'            
        df_value.columns = df_sp.columns
        df_sp = pd.concat([df_value, df_sp])


        ####### CONVERT THE REFLECTANCE VALUES TO COLORIMETRIC VALUES ########
                 
        df_cielab.columns = [meas_id]
        df_cielab = pd.concat([df_value, df_cielab])
        df_cielab.index.name = 'coordinates'

        


        # define the number of measurements
        df_info.loc['measurements_N'] = len(df_cielab.columns)


        df_info = utils.create_df_info(
            db=db,
            info_general=values_general_info,
            info_project=values_project_info,
            info_object=values_object_info,
            info_system=values_system,
            info_device=values_device,
            info_analyses=values_analysis_info,
            info_colorimetry=values_colorimetry_info
            )

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
            date = pd.to_datetime(date_time_analysis).date()
            filename = f'{project_id}_{meas_id}_{group}_{group_description}_{object_type}_{date}'

        elif filenaming == 'default' and db == True:
            
            filename_parameters = config_info['filenaming'][device_ID]['interim']

            if 'date' in filename_parameters:
                new_df_info = df_info.copy()
                new_df_info.loc['date'] = str(df_info.loc['datetime_analysis']['value'].date())               
                
                filename = "_".join([new_df_info.loc[x]['value'].split("_")[0] if "_" in new_df_info.loc[x]['value'] else new_df_info.loc[x]['value'] for x in filename_parameters])

            else:
                filename = "_".join([df_info.loc[x]['value'].split("_")[0] if "_" in df_info.loc[x]['value'] else df_info.loc[x]['value'] for x in filename_parameters])

        elif filenaming == 'default' and db == False:
            filename = stemName

        elif isinstance(filenaming, list):

            if 'date' in filenaming:
                new_df_info = df_info.copy()
                new_df_info.loc['date'] = str(df_info.loc['datetime_analysis']['value'].date())                    

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