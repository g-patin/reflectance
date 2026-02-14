import colour
import msdb
from typing import Optional

from . import config



# define dictionaries for colorimetric calculations
observers = {        
    '10deg': 'cie_10_1964',
    '2deg' : 'cie_2_1931',
}
    
cmfs_observers = {
    '10deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1964 10 Degree Standard Observer"],
    '2deg': colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1931 2 Degree Standard Observer"] 
}



# define the authors names
     
"""
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
            if authors in df_authors['initials'].values:
                df_author = df_authors[df_authors['initials'] == authors]
                authors_names = f"{df_author['surname'].values[0]}, {df_author['name'].values[0]}"
            
            else:
                print(f'The author name "{authors}" has not been registered in the databases. Use the function add_new_person() to register the person.')
                authors_names = authors

    else:
        authors_names = authors
"""




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
    
    # get the colorimetric data for illuminant and observer
    illuminant_SDS = colour.SDS_ILLUMINANTS[illuminant]
    illuminant_CCS = colour.CCS_ILLUMINANTS[observers[observer]][illuminant]

    return illuminant_SDS, illuminant_CCS


def get_cmfs(observer:str, db:bool):
    # define the observer
    if observer == 'default' and db == True:
        if len(config.get_colorimetry_info()) == 0:
            observer = '10deg'
        else:
            observer = config.get_colorimetry_info().loc['observer']['value']

    elif observer == 'default' and db == False:
        observer = '10deg'

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
   

