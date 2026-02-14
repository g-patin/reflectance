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

from . import RS_info_templates
from . import config
from . import utils


def RS_Avt(files: list, device_ID:Optional[str] = 'default', db:Optional[bool] = 'default', filenaming:Optional[str] = 'none', folder:Optional[str] = '.',  comment:Optional[str] = '', wl_range:Optional[tuple] = (180,1100,1), interpolation_wl:Optional[tuple] ='default', rounding_sp:Optional[int] = 'none',  authors:Optional[str] = 'XX', white_standard:Optional[bool] = 'undefined', observer:Optional[str] = '10deg', illuminant:Optional[str] = 'D65', background:Optional[str] = 'black', delete_files:Optional[bool] = True, return_data:Optional[bool] = False, save:Optional[bool] = True):

    
    # define the illuminant value
    illuminant_SDS, illuminant_CCS = utils.get_illuminant(illuminant, observer, db)
    
    # define the color matching functions
    cmfs = utils.get_cmfs(observer, db)    
    
    # get the raw files
    raw_files = [Path(file) for file in files if '.txt' in Path(file).name]    
    
    ####### PROCESS RAW FILES ########

    for raw_file in raw_files:

        ####### DEFINE FILENAME ########
        
        file_path = Path(raw_file) 
        stemName = file_path.stem


        ####### OPEN RAW FILE ########

        file_content = open(raw_file).read()
        lookfor = 'Wave'        
        parameters = file_content[:file_content.index(lookfor)].splitlines() 

        
        ####### RETRIEVE ANALYTICAL PARAMETERS ########

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

        
        # interpolate spectral values
        if interpolation_wl:
            wanted_wl = np.arange(*wl_range)
            wanted_sp = sip.interp1d(wavelengths, sp_raw)(wanted_wl)
                        
        else:
            wanted_wl = wavelengths
            wanted_sp = sp_raw


        # create spectral dataframe
        df_sp = pd.DataFrame({'wavelength_nm':wanted_wl, 'reflectance':wanted_sp})
        df_sp = df_sp.set_index('wavelength_nm')       

        
        # rounding the spectral data
        if rounding_sp == 'none':
            df_sp = df_sp
        elif isinstance(rounding_sp,int):
            df_sp = np.round(df_sp,rounding_sp)  
        else:
            print(f"The value '{rounding_sp}' you entered is not valid. Please enter a positive integer.")
            return
        
        
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
       
        
        ####### RETRIEVE INFO ########

        date_time_analysis = ''
        date_time_processing = datetime.now()

        average = int(float(df_parameters.loc['Averaging Nr. [scans]','value']))
        integration_time = int(float(df_parameters.loc['Integration time [ms]','value']))
        smoothing = int(float(df_parameters.loc['Smoothing Nr. [pixels]','value']))

        meas_id = ''
        group = 'unknown'
        group_description = 'undefined'
        spot_size = 'unknown'

        values_analysis = [
            " ",
            meas_id,
            group,
            group_description,
            spot_size,
            background,
            integration_time,
            average,
            smoothing,
            1,   # measurements_N
            white_standard
        ]

        print(values_analysis)
        return

        ####### CREATE INFO DATAFRAME ########

        if db:

            # retrieve the configuration info
            config_info = config.get_config_info()
            
            # check whether database files have been created
            if len(config_info['databases']) == 0:
                return 'Databases have not been created. Please, create databases by running the function "create_DB" from the reflectance package.'
            
            else:             
                db_name = config_info['databases']['db_name'] 
                db = msdb.DB(db_name)
                db_projects = db.get_projects()
                db_objects = db.get_objects()

                # remove the column 'project_id'
                if 'project_id' in db_objects.columns:
                    db_objects = db_objects.drop('project_id', axis=1)

            params_project = db_projects.columns
            params_object = db_objects.columns


        else:
            params_project = RS_info_templates.project_info
            params_object = RS_info_templates.object_info

            values_project = len(params_project) * ['']
            values_object = len(params_object) * ['']

        
        
        values_general_info = [
            'SINGLE REFLECTANCE MEASUREMENT',
            authors,
            date_time_analysis,
            date_time_processing,
            comment,
        ]

        values_colorimetry = [
            illuminant,
            observer,
        ]

        return values_general_info, values_colorimetry


        params_general_info = RS_info_templates.general_info
        params_analyses = RS_info_templates.analysis_info
        params_colorimetry = RS_info_templates.colorimetric_info
        params_system = RS_info_templates.system_info
        params_device = RS_info_templates.device_info

        info_parameters = params_general_info + params_project + params_object + params_system + params_device + params_analyses + params_colorimetry

        info_values = values_general_info + values_project + values_object + values_system + values_device + values_analyses + values_colorimetry

        dict_info = dict(zip(info_parameters,info_values))
        df_info = pd.DataFrame.from_dict(dict_info,orient='index', columns=['value'])
        df_info.index.name = 'parameter'





        ####### RETURN DATA ########

        if return_data:
            return [df_cielab,df_sp]
        

        ####### SAVE DATA ########

        if save:

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

      


def RS_Tidas(files: list, device_ID:Optional[str] = 'default', db:Optional[bool] = 'default', filenaming:Optional[str] = 'default', folder:Optional[str] = '.',  comment:Optional[str] = '', interpolation_wl:Optional[tuple] ='default', rounding_sp:Optional[int] = 'none',  authors:Optional[str] = 'XX', white_standard:Optional[bool] = 'default', observer:Optional[str] = 'default', illuminant:Optional[str] = 'default', background:Optional[str] = 'black', delete_files:Optional[bool] = True, return_filename:Optional[bool] = True):

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


    '''
    # define the authors names
    if authors == 'XX':
        authors_names = 'unknown' 

    elif db:
        df_authors = db.get_users()
        if '-' in authors or ' - ' in authors:                     
            list_authors = []
            for x in authors.split('-'):
                x = x.strip()
                df_author = df_authors[df_authors['initials'] == x]
                list_authors.append(f"{df_author['surname'].values[0]}, {df_author['name'].values[0]}")                    
            authors_names = '_'.join(list_authors)
                    
        else:            
            print(authors)
            if authors in df_authors['initials'].values:
                df_author = df_authors[df_authors['initials'] == authors]
                authors_names = f"{df_author['surname'].values[0]}, {df_author['name'].values[0]}"
            
            else:
                print(f'The author name "{authors}" has not been registered in the databases. Use the function add_new_person() to register the person.')
                authors_names = authors

    else:
        authors_names = authors
    '''

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
        geometry = config_device['geometry']
        fiber_ill = config_device['fiber_ill']
        fiber_coll = config_device['fiber_coll']
        specular_component = config_device['specular_component']

        
    
    # get the raw files with the measurement data (c01)
    raw_txt_files = [Path(file) for file in files if '.txt' in Path(file).name and 'c01_' in Path(file).name]    

        
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





