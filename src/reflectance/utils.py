from typing import Optional, Union
import pandas as pd
import numpy as np
import colour
import msdb
from scipy.stats import linregress

from . import config
from . import RS_info_templates


# define dictionaries for colorimetric calculations
observers = {        
    '10deg': 'cie_10_1964',
    '2deg' : 'cie_2_1931',
}
    
cmfs_observers = {
    '10deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1964 10 Degree Standard Observer"],
    '2deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1931 2 Degree Standard Observer"] 
}


def compute_cie_coordinates(
    wavelengths, 
    reflectance, 
    coordinates:Union[list, str] = ['L*','a*', 'b*'], 
    illuminant:Optional[str] = 'default', 
    observer:Optional[str] = 'default', 
    db:Optional[bool] = False,
    rounding:Union[int, str] = 2):
   

    # get the illuminant
    illuminant_SDS, illuminant_CCS = get_illuminant(illuminant, observer, db)
    
    # get the cmfs values
    cmfs = get_cmfs(observer, db)

    # make sure that the wavelengths and reflectance have the same length
    if not len(wavelengths) == len(reflectance):

        print('The amount of wavelength values is different than the amount of reflectance values. Please make sure that both have the same length.')
        return

    

    sd = colour.SpectralDistribution(reflectance,wavelengths)        
    XYZ = colour.sd_to_XYZ(sd,cmfs,illuminant=illuminant_SDS)      
    xy = colour.XYZ_to_xy(XYZ)       
    Lab = colour.XYZ_to_Lab(XYZ/100, illuminant_CCS)     
    LCh = colour.Lab_to_LCHab(Lab) 

    all_coordinates = list(XYZ) + list(xy) + list(Lab) + list(LCh)[1:]

    df_coordinates_all = pd.DataFrame(all_coordinates, columns=['values'], index=['X','Y','Z','x','y','L*','a*','b*','C*','h'])
    df_coordinates_wanted = df_coordinates_all.loc[coordinates]
    df_coordinates_wanted.index.name = 'coordinates'

    if isinstance(rounding, int):
        
        if rounding == 0:
            df_coordinates_wanted = df_coordinates_wanted.astype('int64')
        else:
            df_coordinates_wanted = np.round(df_coordinates_wanted, rounding)

        
    return df_coordinates_wanted



def compute_slice_correction(wavelengths: [np.ndarray, pd.DataFrame, list], reflectance: [np.ndarray, list], splice: list[int], interpolation_bands:Optional[int] = 10):

    if isinstance(reflectance, list):
        reflectance = np.array(reflectance).reshape(1,-1)
    
    elif isinstance(reflectance, np.ndarray):
        reflectance = reflectance.reshape(1,-1)

    elif isinstance(reflectance, pd.DataFrame):
        reflectance = reflectance.values

    else:
        print('Processed aborted. The reflectance values should be given as a list, a pandas dataframe, or a numpy array.')
        return

    def extrapfun(x, y, xout):
        slope, intercept, _, _, _ = linregress(x, y)
        return intercept + slope * xout

    if len(splice) not in (1, 2):
        raise ValueError("splice must be a list of length 1 or 2")

    if len(splice) == 1:
        index_b = len(wav)
    else:
        index_b = np.where(wavelengths == splice[1])[0][0]

    has_three_regions = len(splice) == 2

    

    if len(wavelengths) != reflectance.shape[1]:
        raise ValueError("length(wavelengths) must be equal to ncol(reflectance)")

    index = [np.where(wavelengths == s)[0][0] for s in splice]
    
    Xa = reflectance[:, :index[0]+1]
    Xb = reflectance[:, index[0]+1:index_b+1]
    
    tmp_first = Xb[:, :interpolation_bands]
    w_first = wavelengths[index[0]+1:index[0]+1+interpolation_bands]
    
    pred_Xa = np.array([extrapfun(w_first, y, splice[0]) for y in tmp_first])
    offset_a = Xa[:, -1] - pred_Xa

    if has_three_regions:
        Xc = reflectance[:, index[1]+1:]
        tmp_second = Xb[:, -interpolation_bands:]
        w_second = wavelengths[index[1]-interpolation_bands+1:index[1]+1]
        pred_Xb = np.array([extrapfun(w_second, y, splice[1]) for y in tmp_second])
        offset_b = Xc[:, 0] - pred_Xb
        output = np.hstack((Xa - offset_a[:, None], Xb, Xc - offset_b[:, None]))
    else:
        output = np.hstack((Xa - offset_a[:, None], Xb))

    return output[0]


def create_df_info(db:Optional[bool] = False, info_general:Optional[list] = [], info_project:Optional[list] = [], info_object:Optional[list] = [], info_system:Optional[list] = [], info_device:Optional[list] = [], info_analysis:Optional[list] = [], info_colorimetry:Optional[list] = []):

    # retrieve the parameter names
    params_general_info = RS_info_templates.general_info
    params_project = RS_info_templates.project_info
    params_object = RS_info_templates.object_info
    params_analysis = RS_info_templates.analysis_info
    params_colorimetry = RS_info_templates.colorimetric_info
    params_system = RS_info_templates.system_info
    params_device = RS_info_templates.device_info

    if db == False:
        params_object = params_object[:2]
        info_object = ['undefined']

        params_project = params_project[:2]        
        info_project = ['undefined']

    else:
        # check if the database files have been registered or created
        config_info = config.get_config_info()
        if len(config_info['databases']) == 0:
            return 'Databases have not been created. Please, create databases by running the function "create_DB" from the reflectance package.'
        
        else:             
            db_name = config_info['databases']['db_name'] 
            db_rs = msdb.DB(db_name)
            db_projects = db_rs.get_projects()
            db_objects = db_rs.get_objects()

            params_project = ["[PROJECT INFO]"] + list(db_projects.columns)
            params_object = ["[OBJECT INFO]"] + list(db_objects.columns)

            
    info_project = [' '] + info_project
    info_object = [' '] + info_object
    info_system = [' '] + info_system
    info_device = [' '] + info_device
    info_analysis = [' '] + info_analysis
    info_colorimetry = [' '] + info_colorimetry


    if info_project == []:
        info_project = [' '] * len(param)


    if len(info_general) != len(params_general_info):
        print(f'The number of values ({len(info_general)}) that you entered for the info_general should be equal to {len(params_general_info)-1}. Please provide a list of string values for the following parameters: {params_general_info.values}. If you do not have a value for one of the parameter, insert an empty string.')        
        return 

    if len(info_project) != len(params_project):
        print(f'The number of values ({len(info_project)}) that you entered for the info_project should be equal to {len(params_project)-1}. Please provide a list of string values for the following parameters: {list(params_project)}. If you do not have a value for one of the parameter, insert an empty string.')        
        return  

    if len(info_object) != len(params_object):
        print(f'The number of values ({len(info_project)}) that you entered for the info_object should be equal to {len(params_object)-1}. Please provide a list of string values for the following parameters: {params_object}. If you do not have a value for one of the parameter, insert an empty string.')        
        return  

    if len(info_analysis) != len(params_analysis):
        print(f'The number of values ({len(info_analyses)}) that you entered for the info_analyses should be equal to {len(params_analysis)-1}. Please provide a list of string values for the following parameters: {params_analysis.values}. If you do not have a value for one of the parameter, insert an empty string.')        
        return  
          
    # concatenate the parameter names
    info_parameters = params_general_info + params_project + params_object + params_system + params_device + params_analysis + params_colorimetry        
        
    # contactenate the info values
    info_values = info_general + info_project + info_object + info_system + info_device + info_analysis + info_colorimetry

    dict_info = dict(zip(info_parameters,info_values))
    df_info = pd.DataFrame.from_dict(dict_info,orient='index', columns=['value'])
    df_info.index.name = 'parameter'

    return df_info


def define_rounding(rounding:Union[str,int,tuple] = 'none'):
    """Define the rounding parameters for the reflectance data

    Parameters
    ----------
    rounding : Union[str,int,tuple], optional
        _description_, by default 'none'

    Returns
    -------
    tuple
        It returns a tuple of two value. The first value defines the rounding behaviour for the colorimetric coordinates. The second values defines the rounding behaviour for the spectra values.
    """

    if isinstance(rounding, int):
        rounding_cl = rounding
        rounding_sp = rounding
     
    elif isinstance(rounding, (tuple, list)):
        rounding_cl = rounding[0]
        rounding_sp = rounding[1]
 
    else:
        rounding_cl = 'none'
        rounding_sp = 'none'

    return rounding_cl,rounding_sp


def get_authors(authors, db:bool):        

    if authors == 'XX':
        authors_names = 'unknown'

    else:
        if db:
            db_name = config.get_config_info()['databases']['db_name']
            db = msdb.DB(db_name=db_name)
            df_authors = db.get_users()

            if '-' in authors or ' - ' in authors:                     
                list_authors = []
                
                for x in authors.split('-'):
                    x = x.strip()
                    if x in df_authors['initials'].values:                        
                        df_author = df_authors[df_authors['initials'] == x]                    
                        list_authors.append(f"{df_author['surname'].values[0]}, {df_author['name'].values[0]}")   
                    else:
                        print(f'The user ("{x}") has not been registered in the database file.')
                        list_authors.append(x)            
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

       
    return authors_names
  

def get_illuminant(illuminant:str, observer:str, db:bool):

    # define the illuminant value
    if illuminant == 'default' and db == True:
        if len(config.get_colorimetry_info()) == 0:
            illuminant = 'D65'
        else:
            illuminant = config.get_colorimetry_info().loc['illuminant']['value']

    elif illuminant == 'default' and db == False:
        illuminant = 'D65'

    valid_illuminants = ['A', 'B', 'C', 'D50', 'D55', 'D60', 'D65', 'D75', 'E', 'FL1', 'FL2', 'FL3', 'FL4', 'FL5', 'FL6', 'FL7', 'FL8', 'FL9', 'FL10', 'FL11', 'FL12', 'FL3.1', 'FL3.2', 'FL3.3', 'FL3.4', 'FL3.5', 'FL3.6', 'FL3.7', 'FL3.8', 'FL3.9', 'FL3.10', 'FL3.11', 'FL3.12', 'FL3.13', 'FL3.14', 'FL3.15', 'HP1', 'HP2', 'HP3', 'HP4', 'HP5', 'LED-B1', 'LED-B2', 'LED-B3', 'LED-B4', 'LED-B5', 'LED-BH1', 'LED-RGB1', 'LED-V1', 'LED-V2', 'ID65', 'ID50', 'ISO 7589 Photographic Daylight', 'ISO 7589 Sensitometric Daylight', 'ISO 7589 Studio Tungsten', 'ISO 7589 Sensitometric Studio Tungsten', 'ISO 7589 Photoflood', 'ISO 7589 Sensitometric Photoflood', 'ISO 7589 Sensitometric Printer']

    if illuminant not in valid_illuminants:
            print(f"The illuminant value you entered ({illuminant}) is not valid. Please choose a value from the following options: {valid_illuminants}")
            return
    
    if observer not in ['2deg', '10deg']:
            print(f'The observer value you entered ({observer}) is not valid. Please choose a value from the following options: ["2deg","10deg"]')
            return
    
    # get the colorimetric data for illuminant and observer
    illuminant_SDS = colour.SDS_ILLUMINANTS[illuminant]
    illuminant_CCS = colour.CCS_ILLUMINANTS[observers[observer]][illuminant]

    return illuminant_SDS, illuminant_CCS


def get_institution(institution:Optional[str] == 'XX', db:Optional[bool] = False):

    if db:

        config_info_institution = config.get_config_info()['institution']

        if len(config_info_institution) == 0:

            if institution == 'XX':
                institution_info = 'undefined'

            else:
                institution_info = institution

        else:
            institution_acronym = config_info_institution['acronym']
            institution_name = config_info_institution['name']
                    
            if institution_acronym == '':
                institution_info = institution_name
            else:
                institution_info = f'{institution_acronym}_{institution_name}'

    else:
        if institution == 'XX':
            institution_info = 'undefined'

        else:
            institution_info = institution
               
    return institution_info


def get_cmfs(observer:str, db:bool):
    # define the observer
    if observer == 'default' and db == True:
        if len(config.get_colorimetry_info()) == 0:
            observer = '10deg'
        else:
            observer = config.get_colorimetry_info().loc['observer']['value']

    elif observer == 'default' and db == False:
        observer = '10deg'


    if observer not in ['2deg', '10deg']:
            print(f'The observer value you entered ({observer}) is not valid. Please choose a value from the following options: ["2deg","10deg"]')
            return

    cmfs = cmfs_observers[observer]

    return cmfs


def get_white_standard(white_standard:str, db:bool, device_id:Optional[str] = None):
    
    if db:

        db_name = config.get_config_info()['databases']['db_name']
        db = msdb.DB(db_name=db_name)
        db_white_standards = db.get_white_standards().set_index('ID')

        if white_standard == 'default':

            if device_id not in config.get_config_info()['devices'].keys():
                print(f'The device ID you enter ("{device_id}") has not been registered in the database files.')
                print('To register info in the database files, see the documentation : https://g-patin.github.io/reflectance/')
                print('The white standard has been set to "undefined".')
                white_standard_ID = 'undefined'

            elif len(config.get_config_info()['devices'][device_id]['white_standard']) == 0:
                print(f'The device ID has been registered but the white standard ID value is empty.')
                print('To register info in the database files, see the documentation : https://g-patin.github.io/reflectance/')
                print('The white standard has been set to "undefined".')
                white_standard_ID = 'undefined'          
            
            else:
                white_standard_ID = config.get_config_info()['devices'][device_id]['white_standard']
                
        else:
            white_standard_ID = white_standard
        
        if white_standard_ID in db_white_standards.index:
            white_standard_info = db_white_standards.loc[white_standard_ID,'description']
            white_standard_info = f'{white_standard_ID}_{white_standard_info}'

        elif white_standard_ID == 'undefined':
            white_standard_info = 'undefined'
        
        else:
            print(f'The white standard value you entered ("{white_standard}") has not been registered in the database files.')
            print('To register info in the database files, see the documentation : https://g-patin.github.io/reflectance/')
            print('The white standard has been set to "undefined".')
            white_standard_info = 'undefined'
    
    else:
        white_standard_info = white_standard

    return white_standard_info
   

