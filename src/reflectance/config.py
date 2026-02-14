import pandas as pd
from pathlib import Path
from typing import Optional
import json
import os
from ipywidgets import Layout, Label
import ipywidgets as ipw
from IPython.display import display, clear_output
import ast
import msdb


from . import RS_info_templates

style = {"description_width": "initial"}
config_file = Path(__file__).parent / 'config_info.json'



def get_colorimetry_info():

    if not config_file.exists():
        print("The configuration file does not exist. Please ensure 'db_config.json' is created.")
        return None
    with open(config_file, "r") as f:
        config = json.load(f)

    # Check if the 'lighting' key exists in the config
    if "colorimetry" in config:
        colorimetry_info = config["colorimetry"]
        
        # Convert user info to a DataFrame
        df = pd.DataFrame.from_dict(colorimetry_info, orient="index", columns=["value"])
        return df
    else:
        print("The colorimetric conditions have not been registered. Please register using the 'set_colorimetry_info' function.")
        return None
    

def get_config_info():

    # Load folder path from JSON file if it exists
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            config = json.load(file)
            return config
        
    else:
        print('The config_info.json has been deleted ! Please re-install the reflectance package.')
        return None 
    
    
def get_config_path():

    config_info = get_config_info()
    config_info_path = config_info['databases']['path_folder']

    return config_info_path


def get_db(name_db:Optional[str] = 'RS'):
    
    return msdb.DB(db_name=name_db)
    

def get_institution_info():

    # Retrieve the config info
    config_info = get_config_info() 
    
    # Check if the 'institution' key exists in the config
    if "institution" in config_info:
        institution_info = config_info["institution"]

        # Return nothing if no institution info registered
        if len(institution_info) == 0:
            print("The institution info have not been registered. Please register using the 'set_institution_info' function.")
            return None
        
        # Convert the institution info to a DataFrame and return them
        df = pd.DataFrame.from_dict(institution_info, orient="index", columns=["value"])
        return df
        
    else:
        print("The dictionary named 'institution' has been removed from the config_info.json file. Re-insert it as an empty dictionary or re-install the package.")
        return None


def reset_config():
    """Reset the config_info.json to its initial state, i.e. all empty dictionaries.
    """

    config_info = get_config_info()

    for key in config_info.keys():
        config_info[key] = {}

    # Save the updated config back to the JSON file
    with open(config_file, "w") as f:
        json.dump(config_info, f, indent=4)

        print(f'The {config_file.name} file has been successfully reset.')
    

def set_colorimetry_info():   

    # import the databases
    db = get_db()

    # define some widgets
    wg_observer = ipw.Dropdown(
        description = 'Observer (deg)',
        value = '10',
        options = ['2', '10'],
        style = style
    )

    wg_illuminant = ipw.Dropdown(
        description = 'Illuminant',
        value = 'D65',
        options = ['A', 'B', 'C', 'D50', 'D55', 'D60', 'D65', 'D75', 'E', 'FL1', 'FL2', 'FL3', 'FL4', 'FL5', 'FL6', 'FL7', 'FL8', 'FL9', 'FL10', 'FL11', 'FL12', 'FL3.1', 'FL3.2', 'FL3.3', 'FL3.4', 'FL3.5', 'FL3.6', 'FL3.7', 'FL3.8', 'FL3.9', 'FL3.10', 'FL3.11', 'FL3.12', 'FL3.13', 'FL3.14', 'FL3.15', 'HP1', 'HP2', 'HP3', 'HP4', 'HP5', 'LED-B1', 'LED-B2', 'LED-B3', 'LED-B4', 'LED-B5', 'LED-BH1', 'LED-RGB1', 'LED-V1', 'LED-V2', 'ID65', 'ID50'],
        style = style
    )
    wg_white_standard = ipw.Dropdown(
        description = 'White standard',            
        options = db.get_white_standards()['ID'].values,
        style = style
    )
    recording = ipw.Button(
        description='Save',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',            
    )
    button_record_output = ipw.Output()
    
    # define the function to record the widgets values
    def button_record_pressed(b):
        """
        Save the colorimetry info in the config_info file.
        """
        button_record_output.clear_output(wait=True)
        with open(config_file, "r") as f:
            config = json.load(f)
        # Update config with user data
        config["colorimetry"] = {
            "observer": f'{wg_observer.value}deg',
            "illuminant": wg_illuminant.value, 
            "white_standard": wg_white_standard.value,                               
        }
        # Save the updated config back to the JSON file
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        
        with button_record_output:
            print(f'Colorimetric conditions info recorded in the {config_file.name} file.')
    
    # link the button to the aforementioned function
    recording.on_click(button_record_pressed)

    # display the widgets
    display(ipw.VBox([wg_observer, wg_illuminant, wg_white_standard]))
    display(ipw.HBox([recording, button_record_output]))


def set_comment_info():

    config_info = get_config_info()
    parameters = RS_info_templates.general_info[1:] + RS_info_templates.project_info[1:] + RS_info_templates.object_info[1:] + RS_info_templates.device_info[1:] + RS_info_templates.analysis_info[1:] + ['measurement_Nb']
    parameters = sorted(parameters)  
    devices = list(config_info['devices'].keys())
    

    wg_device_ID = ipw.Dropdown(
        description='Device ID',
        value=devices[0],
        options=devices,
        style=style,
    )
        
    wg_comment1 = ipw.Dropdown(
        description='Comment 1',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_comment2 = ipw.Dropdown(
        description='Comment 2',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_comment3 = ipw.Dropdown(
        description='Comment 3',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_comment4 = ipw.Dropdown(
        description='Comment 4',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_comment5 = ipw.Dropdown(
        description='Comment 5',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_comment6 = ipw.Dropdown(
        description='Comment 6',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_comment7 = ipw.Dropdown(
        description='Comment 7',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_comment8 = ipw.Dropdown(
        description='Comment 8',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_comment9 = ipw.Dropdown(
        description='Comment 9',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    recording = ipw.Button(
        description='Save',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',            
    )

    button_record_output = ipw.Output()

        
    def change_parameters(change):        

        selected_device_ID = wg_device_ID.value
        device_parameters_flex = config_info['devices'][selected_device_ID]['device_params']
               
        if len(device_parameters_flex) > 0:
            
            device_parameters_flex = list(device_parameters_flex.keys())        
            wg_comment1.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_comment1.value = 'none'

            wg_comment2.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_comment2.value = 'none'

            wg_comment3.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_comment3.value = 'none'

            wg_comment4.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_comment4.value = 'none'

            wg_comment5.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_comment5.value = 'none'

            wg_comment6.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_comment6.value = 'none'

            wg_comment7.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_comment7.value = 'none'

            wg_comment8.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_comment8.value = 'none'

            wg_comment9.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_comment9.value = 'none'





    def button_record_pressed(b):
        """
        Save the comments info in the config_info.json file.
        """

        button_record_output.clear_output(wait=True)
        comments = [wg_comment1.value, wg_comment2.value, wg_comment3.value, wg_comment4.value, wg_comment5.value, wg_comment6.value, wg_comment7.value, wg_comment8.value, wg_comment9.value]

        # remove the 'none' values from the list of comments
        comments = [x for x in comments if x != 'none']

        with open(config_file, "r") as f:
            config = json.load(f)
            existing_comments = config['comments']

        # Update config with user data
        existing_comments[wg_device_ID.value] = comments
        config['comments'] = existing_comments


        # Save the updated config back to the JSON file
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)

            
        with button_record_output:
            print('The comment information have been recorded in the config_info.json file.')

        
    wg_device_ID.observe(change_parameters, names='value')
    recording.on_click(button_record_pressed)

    display(ipw.VBox([wg_device_ID,wg_comment1, wg_comment2, wg_comment3, wg_comment4, wg_comment5, wg_comment6, wg_comment7, wg_comment8, wg_comment9]))
    display(ipw.HBox([recording, button_record_output]))


def set_config_info():

    config_info = get_config_info()
    keys = [x for x in config_info.keys() if x not in ['colorimetry','comments','databases','devices']]


    wg_keys = ipw.Dropdown(
        description='Keys',
        placeholder='Select a key',
        options=keys,
        style=style
    )

    wg_ID = ipw.Text(
        description='ID',
        placeholder='Enter an ID number',
        style=style
    )

    wg_description = ipw.Text(
        description='Description',
        placeholder='Item information',
        style=style
    )

    recording = ipw.Button(
        description='Save',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',            
    )

    button_record_output = ipw.Output()


    def button_record_pressed(b):
        """
        Save the info in the config_info.json file.
        """

        button_record_output.clear_output(wait=True)

        with open(config_file, "r") as f:
            config = json.load(f)
            existing_config = config[wg_keys.value]

            
        existing_config[wg_ID.value] = wg_description.value                    
        config[wg_keys.value] = existing_config                 
            
        # Save the updated config back to the JSON file
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)

            
        with button_record_output:
            print(f'The info have been saved in the config_info.json file.')

        
    recording.on_click(button_record_pressed)

    display(ipw.VBox([wg_keys,wg_ID, wg_description]))
    display(ipw.HBox([recording, button_record_output]))


def set_db(folder_path:Optional[str] = '', use:Optional[bool] = True, msdb_config:Optional[bool] = True):

    # retrieve the databases names
    existing_db_names = msdb.get_db_names()
    
    # define some widgets
    wg_name = ipw.Combobox(
        description = 'db_name',
        placeholder='Enter a new database name or select a database registered in the msdb package',
        value = '',
        options = existing_db_names,
        style = style,
        layout=Layout(width="50%", height="30px"),
    ) 

    wg_folder = ipw.Text(
        description = 'db_path',
        placeholder = 'Location of the database folder on your computer',
        value = folder_path,
        style = style, 
        layout=Layout(width="50%", height="30px"),
    )

    wg_use = ipw.Dropdown(
        description = 'Use',
        value = use,
        options = [True, False],
        style = style,
        layout=Layout(width="10%", height="30px"),
    )  

    recording = ipw.Button(
        description='Save',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',            
    )

    button_record_output = ipw.Output()
    
    # define the function to record the widgets values
    def button_record_pressed(b):
        """
        Save the databases info in the db_config.json file.
        """
        button_record_output.clear_output(wait=True)
        with open(config_file, "r") as f:
            config = json.load(f)
        # update config with user data
        config["databases"] = {
            "db_name": wg_name.value,
            "path_folder": wg_folder.value,
            "usage": wg_use.value,
                          
        }
        # save the updated config back to the JSON file
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)

        with button_record_output:
            print(f'Database ({wg_name.value}) info recorded in the config_info.json file of the reflectance package.')

        # save the database info inside the db_config.json fo the msdb package
        if wg_name.value not in existing_db_names:
            msdb.set_db(db_name=wg_name.value,path_folder=wg_folder.value, widgets=False)
            with button_record_output:
                
                print(f'Database ({wg_name.value}) recorded in the db_config.json file of the msdb package.')
        
    
    # define function when the database is already existing
    def change_db_name(change):
        if change.new in existing_db_names:
            wg_folder.value = msdb.get_config_file()['databases'][change.new]['path_folder']

    
    # link the button to the aforementioned function
    recording.on_click(button_record_pressed)
    wg_name.observe(change_db_name)

    # display the widgets
    display(ipw.VBox([wg_name,wg_folder, wg_use]))
    display(ipw.HBox([recording, button_record_output]))


def set_devices_info():

    # retrieve the content of the config_info file
    config_info = get_config_info()

    
    # instantiate a DB class object
    name_db = 'RS'
    db = get_db(name_db=name_db)


    # retrieve registered devices ID
    devices_ID = tuple(db.get_devices()['ID'].values)


    # retrieve the registered fibers ID
    fibers_ID = ['']

    # retrieve the registered white standards
    white_standards = tuple(db.get_white_standards()['ID'].values)


    # create widgets (standard fixed info)

    wg_device_type = ipw.Dropdown(
        description='Device Type',
        placeholder='Select a type',
        options=['Single-unit', 'Assembly'],
        style=style
    )

    wg_device_ID = ipw.Combobox(
        description='Device ID',
        placeholder='Select or enter a device ID',
        options=devices_ID,
        value='',
        style=style, 
    )

    wg_function = ipw.Dropdown(
        description='Process function',
        options=sorted(RS_info_templates.process_rawdata_functions),
        style=style, 
    )

    wg_brand = ipw.Text(
        description='Brand',
        placeholder='Company/Person who made or sold the device',
        style=style, 
    )

    wg_model = ipw.Text(
        description='Device model',
        placeholder='Enter the device model',
        style=style, 
    )

    wg_lamp = ipw.Combobox(
        description='Lamp',
        placeholder='Select or enter a lamp',
        style=style
    )

    wg_geometry = ipw.Dropdown(
        description='Geometry (ill:coll)',            
        options=["0:45", "45:0", "0:0"],
        style=style
    )

    wg_fiber_ill = ipw.Dropdown(
        description='Fiber illumination',                        
        options=["none"] + fibers_ID,
        style=style
    )

    wg_fiber_coll = ipw.Dropdown(
        description='Fiber collection',                        
        options=["none"] + fibers_ID,
        style=style
    )

    wg_specular_component = ipw.Dropdown(
        description='Specular component',                        
        options=["SCE_excluded", "SCI_included", "partly-included", "unknown"],
        value='unknown',
        style=style
    )

    wg_white_standard = ipw.Dropdown(
        description='White standard',                        
        options=white_standards,        
        style=style
    )

    wg_if_interpolation = ipw.Checkbox(
        value=False,
        description='Interpolation',
        disabled=False,
        indent=False,
        #layout=Layout(width="10%", height="30px")
    )
        
    wg_wavelength_range = ipw.IntRangeSlider(
        value=[100, 3000],
        min=100,
        max=3000,
        step=1,
        description='Wavelength range (nm)',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style=style,
        layout=Layout(width="98%", height="30px")
    )

    wg_wavelength_step = ipw.BoundedIntText(
        min=1,
        max=50,
        step=1,
        description='Wavelength step (nm)',
        style=style
    )

    wavelength_range_output = ipw.Output(
        style=style,
        #layout=Layout(width="500px", height="30px")
    )
    
    wavelength_step_output = ipw.Output(
        style=style
    )


    # create widgets (flexible specific info)

    wg_flex_param_label = ipw.Text(
        placeholder='Enter a parameter label',
        style=style
    )

    wg_flex_param_value = ipw.Combobox(
        placeholder='Enter or select a value',
        options=['Unknown'],
        style=style
    )

    wg_flex_param_added = ipw.Textarea(
        style=style, 
        layout=Layout(width="98%", height="320px")
    )


    # create widgets (recording)

    recording = ipw.Button(
        description='Save',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',            
    )

    button_record_output = ipw.Output()


    # Create a button to remove selected materials
    add_flex_param_button = ipw.Button(
        description='Add parameter',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        icon='',
        layout=Layout(width="50%", height="30x"),
        style=style,
    ) 

    remove_flex_param_button = ipw.Button(
        description='Remove parameter',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        icon='',
        layout=Layout(width="50%", height="30x"),
        style=style,
    ) 

    # define function to set wavelength range
    def change_if_interpolation(change):
        if change.new == True:
            with wavelength_range_output:
                wavelength_range_output.clear_output(wait=True)                    
                display(wg_wavelength_range)
            
            with wavelength_step_output:
                wavelength_step_output.clear_output(wait=True)
                display(wg_wavelength_step)

        else:
            with wavelength_range_output:
                wavelength_range_output.clear_output(wait=True) 

            with wavelength_step_output:
                wavelength_step_output.clear_output(wait=True)

    
    # Function to add flex specific info
    def add_flex_info_click(change):

        if wg_flex_param_label.value != '':
            current_flex_param_info = wg_flex_param_added.value

            if current_flex_param_info == '':
                wg_flex_param_added.value = f'{wg_flex_param_label.value}:{wg_flex_param_value.value}'

            else:
                wg_flex_param_added.value = f'{current_flex_param_info}\n{wg_flex_param_label.value}:{wg_flex_param_value.value}'

    
    # Function to remove flex specific info
    def remove_flex_info_click(change):

        if wg_flex_param_added.value != '':

            new_value = '\n'.join(wg_flex_param_added.value.splitlines()[:-1])
            wg_flex_param_added.value = new_value
        


    # Function to save device info
    def button_record_pressed(b):
        """
        Save the exposure conditions info in the db_config.json file.
        """

        button_record_output.clear_output(wait=True)

        with open(config_file, "r") as f:
            config = json.load(f)
            existing_devices_info = config['devices']

        device_params = wg_flex_param_added.value.splitlines()

        device_params_keys = [x.split(':')[0] for x in device_params]
        device_params_values = [x.split(':')[1] for x in device_params]

        device_params_dic = dict(zip(device_params_keys,device_params_values))

        if wg_if_interpolation.value == False:
            interpolation = 'none'

        else:
            interpolation = (wg_wavelength_range.value[0],wg_wavelength_range.value[1],wg_wavelength_step.value)

        new_info =  {
                'device_type': wg_device_type.value,
                'process_function': wg_function.value,
                'brand': wg_brand.value,                
                'model': wg_model.value,
                'geometry': wg_geometry.value,
                'fiber_ill': wg_fiber_ill.value,
                'fiber_coll': wg_fiber_coll.value,
                'lamp': wg_lamp.value,
                'specular_component': wg_specular_component.value,
                'white_standard':wg_white_standard.value,
                'interpolation':interpolation,
                'device_params':device_params_dic
            }
                               
        with button_record_output:
            print(new_info)
        existing_devices_info[wg_device_ID.value] = new_info
        config['devices'] = existing_devices_info

        # Save the updated config back to the JSON file
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)

            
        with button_record_output:
            print(f'The info of device {wg_device_ID.value} have been saved in the db_config.json file.')
        

     # Add some titles
    title_standard_info = Label("Standard info", layout=Layout(width="auto"))
    title_standard_info.style = {"font_weight": "bold", "font_size": "20px", "font_family": "serif"}

    title_specific_info = Label("Specific info", layout=Layout(width="auto"))
    title_specific_info.style = {"font_weight": "bold", "font_size": "20px", "font_family": "serif"}

    
    
    # Set the button click event handler
    add_flex_param_button.on_click(add_flex_info_click)
    remove_flex_param_button.on_click(remove_flex_info_click)
    recording.on_click(button_record_pressed)

    wg_if_interpolation.observe(change_if_interpolation, names='value')


    # Display the widgets

    display(ipw.HBox([ipw.VBox([title_standard_info, wg_device_ID, wg_device_type, wg_brand, wg_model, wg_function, wg_lamp, wg_specular_component, wg_geometry, wg_fiber_ill, wg_fiber_coll, wg_if_interpolation, wavelength_range_output, wavelength_step_output]), ipw.VBox([title_specific_info, wg_flex_param_label, wg_flex_param_value, ipw.HBox([add_flex_param_button, remove_flex_param_button]), wg_flex_param_added])]))
    #display(ipw.VBox([wg_ID, wg_function, wg_brand, wg_model, wg_geometry, wg_fiber_ill, wg_fiber_coll, wg_specular_component]))
    #display(ipw.VBox(list(text_widgets.values())))
    display(ipw.HBox([recording, button_record_output]))


    return

    fibers_ID = list(config_info['fibers'].keys())
    device_keys =sorted(set(config_info['devices'][device_ID]) - set(['process_function','brand','model', 'geometry', 'fiber_ill', 'fiber_coll', 'specular_component']))
    
    wg_specific_parameters = ipw.Text()         
    text_widgets = {item: ipw.Text(description=item, style=style) for item in device_keys}


def set_devices_keys():
        
    flexible_keys = sorted(set(RS_info_templates.device_info + ['background', 'spot_size_mm']) - set(['brand','device_ID', 'model', 'geometry', 'fiber_ill', 'fiber_coll', 'specular_component', '[DEVICE INFO]']))
            
    # define some widgets
    wg_ID = ipw.Text(
        description='Device ID',
        placeholder='Enter the device ID',
        style=style, 
    )

    wg_keys = ipw.SelectMultiple(
        descriptions='Keys',
        options=flexible_keys,
        rows=10,
        style=style,
    )

    recording = ipw.Button(
        description='Save',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',            
    )

    button_record_output = ipw.Output()
    
    # define the function to record the widgets values
    def button_record_pressed(b):
        """
        Save the device keys in the config_info.json file.
        """

        button_record_output.clear_output(wait=True)

        with open(config_file, "r") as f:
            config = json.load(f)
            existing_device_dict = config['devices']            

        permanent_keys = ['process_function','brand', 'device_model', 'geometry', 'fiber_ill', 'fiber_coll', 'specular_component']
        all_keys = permanent_keys + list(wg_keys.value)            

        device_dict = {}
        for key in all_keys:
            device_dict[key] = ''
            
        existing_device_dict[wg_ID.value] = device_dict            
        config["devices"] = existing_device_dict
                    
        # Save the updated config back to the JSON file
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)

            
        with button_record_output:
            print(f'The keys (parameters) for device {wg_ID.value} have been saved in the config_info.json file.')

        
    # link the button to the aforementioned function
    recording.on_click(button_record_pressed)

    # display the widgets
    display(ipw.VBox([wg_ID, wg_keys]))
    display(ipw.HBox([recording, button_record_output]))


def set_fibers_info():

    # define some widgets
    wg_ID = ipw.Text(
        description='ID',
        placeholder='ID number of the fiber',
        style=style, 
        layout=Layout(width="30%", height="30px")
    )
    wg_description = ipw.Text(
        description='Description',
        placeholder='Description of the fiber',
        style=style, 
        layout=Layout(width="30%", height="30px")
    )
    recording = ipw.Button(
        description='Save',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',            
    )
    button_record_output = ipw.Output()
    
    # define the function to record the widgets values
    def button_record_pressed(b):
        """
        Save the fiber info in the config_info.json file.
        """
        button_record_output.clear_output(wait=True)
        with open(config_file, "r") as f:
            config = json.load(f)
            existing_info = config['fibers']

        existing_info[wg_ID.value] = wg_description.value            
        config["fibers"] = existing_info                                      
        
        # Save the updated config back to the JSON file
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        
        with button_record_output:
            print(f'The fiber info have been saved in the {config_file.name} file.')
    
    # link the button to the aforementioned function
    recording.on_click(button_record_pressed)

    # display the widgets
    display(ipw.VBox([wg_ID, wg_description]))
    display(ipw.HBox([recording, button_record_output]))


def set_filters_info():

    # define some widgets
    wg_ID = ipw.Text(
        description='ID',
        placeholder='ID number of the filter',
        style=style, 
        layout=Layout(width="30%", height="30px")
    )
    wg_description = ipw.Text(
        description='Description',
        placeholder='Description of the filter',
        style=style, 
        layout=Layout(width="30%", height="30px")
    )
    recording = ipw.Button(
        description='Save',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',            
    )
    button_record_output = ipw.Output()
    
    # define the function to record the widgets values
    def button_record_pressed(b):
        """
        Save the filter info in the config_info.json file.
        """
        button_record_output.clear_output(wait=True)
        with open(config_file, "r") as f:
            config = json.load(f)
            existing_info = config['filters']

        existing_info[wg_ID.value] = wg_description.value            
        config["filters"] = existing_info                                      
        
        # Save the updated config back to the JSON file
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        
        with button_record_output:
            print(f'The filter info have been saved in the {config_file.name} file.')
    
    # link the button to the aforementioned function
    recording.on_click(button_record_pressed)

    # display the widgets
    display(ipw.VBox([wg_ID, wg_description]))
    display(ipw.HBox([recording, button_record_output]))


def set_lamps_info():

    # define some widgets
    wg_ID = ipw.Text(
        description='ID',
        placeholder='ID number of the lamp',
        style=style, 
        layout=Layout(width="30%", height="30px")
    )
    wg_description = ipw.Text(
        description='Description',
        placeholder='Description of the lamp',
        style=style, 
        layout=Layout(width="30%", height="30px")
    )
    recording = ipw.Button(
        description='Save',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',            
    )
    button_record_output = ipw.Output()
    
    # define the function to record the widgets values
    def button_record_pressed(b):
        """
        Save the lamp info in the config_info.json file.
        """
        button_record_output.clear_output(wait=True)
        with open(config_file, "r") as f:
            config = json.load(f)
            existing_info = config['lamps']

        existing_info[wg_ID.value] = wg_description.value            
        config["lamps"] = existing_info                                      
        
        # Save the updated config back to the JSON file
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        
        with button_record_output:
            print(f'The lamp info have been saved in the {config_file.name} file.')
    
    # link the button to the aforementioned function
    recording.on_click(button_record_pressed)

    # display the widgets
    display(ipw.VBox([wg_ID, wg_description]))
    display(ipw.HBox([recording, button_record_output]))


def set_filenaming_interim():

    config_info = get_config_info()
    parameters = RS_info_templates.general_info[1:] + RS_info_templates.project_info[1:] + RS_info_templates.object_info[1:] + RS_info_templates.device_info[1:] + RS_info_templates.analysis_info[1:] + ['measurement_Nb', 'date']
    parameters = sorted(parameters)
    devices = list(config_info['devices'].keys())
    

    wg_device_ID = ipw.Dropdown(
        description='Device ID',
        value=devices[0],
        options=devices,
        style=style,
    )
        
    wg_name1 = ipw.Dropdown(
        description='Name 1',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name2 = ipw.Dropdown(
        description='Name 2',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name3 = ipw.Dropdown(
        description='Name 3',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name4 = ipw.Dropdown(
        description='Name 4',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name5 = ipw.Dropdown(
        description='Name 5',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name6 = ipw.Dropdown(
        description='Name 6',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name7 = ipw.Dropdown(
        description='Name 7',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name8 = ipw.Dropdown(
        description='Name 8',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name9 = ipw.Dropdown(
        description='Name 9',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    recording = ipw.Button(
        description='Save',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',            
    )

    button_record_output = ipw.Output()


    def change_parameters(change):

        selected_device_ID = wg_device_ID.value       
        device_parameters_flex = config_info['devices'][selected_device_ID]['device_params']
       
        if len(device_parameters_flex) > 0:
            
            device_parameters_flex = list(device_parameters_flex.keys())            
                        
            wg_name1.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name1.value = 'none'

            wg_name2.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name2.value = 'none'

            wg_name3.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name3.value = 'none'

            wg_name4.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name4.value = 'none'

            wg_name5.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name5.value = 'none'

            wg_name6.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name6.value = 'none'

            wg_name7.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name7.value = 'none'

            wg_name8.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name8.value = 'none'

            wg_name9.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name9.value = 'none'        
        

    def button_record_pressed(b):
        """
        Save the names info in the config_info.json file.
        """

        button_record_output.clear_output(wait=True)
        names = [wg_name1.value, wg_name2.value, wg_name3.value, wg_name4.value, wg_name5.value, wg_name6.value, wg_name7.value, wg_name8.value, wg_name9.value]

        # remove the 'none' values from the list of names
        names = [x for x in names if x != 'none']

        with open(config_file, "r") as f:
            config = json.load(f)
            existing_filenamings = config['filenaming']
            if wg_device_ID.value in existing_filenamings.keys():
                
                existing_filenamings_device = existing_filenamings[wg_device_ID.value]

            else:                
                existing_filenamings[wg_device_ID.value] = {}
                existing_filenamings_device = existing_filenamings[wg_device_ID.value]

        
        # Update config with filenaming info
        existing_filenamings_device['interim'] = names
        config['filenaming'][wg_device_ID.value] = existing_filenamings_device

        # Save the updated config back to the JSON file
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)

            
        with button_record_output:
            print(f'The filenaming of interim files for device {wg_device_ID.value} have been recorded in the config_info.json file.')

    wg_device_ID.observe(change_parameters, names='value')
    recording.on_click(button_record_pressed)

    display(ipw.VBox([wg_device_ID,wg_name1, wg_name2, wg_name3, wg_name4, wg_name5, wg_name6, wg_name7, wg_name8, wg_name9]))
    display(ipw.HBox([recording, button_record_output]))


def set_filenaming_raw():

    config_info = get_config_info()
    parameters = RS_info_templates.general_info[1:] + RS_info_templates.project_info[1:] + RS_info_templates.object_info[1:] + RS_info_templates.device_info[1:] + RS_info_templates.analysis_info[1:] + ['measurement_Nb']
    parameters = sorted(parameters)     
    devices = list(config_info['devices'].keys())
    

    wg_device_ID = ipw.Dropdown(
        description='Device ID',
        value='Select a device',
        options=['Select a device'] + devices,
        style=style
    )
        
    wg_name1 = ipw.Dropdown(
        description='Name 1',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name2 = ipw.Dropdown(
        description='Name 2',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name3 = ipw.Dropdown(
        description='Name 3',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name4 = ipw.Dropdown(
        description='Name 4',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name5 = ipw.Dropdown(
        description='Name 5',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name6 = ipw.Dropdown(
        description='Name 6',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name7 = ipw.Dropdown(
        description='Name 7',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name8 = ipw.Dropdown(
        description='Name 8',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    wg_name9 = ipw.Dropdown(
        description='Name 9',
        value='none',
        options=['none'] + parameters,
        style=style
    )

    recording = ipw.Button(
        description='Save',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',            
    )

    button_record_output = ipw.Output()

    
    
    def change_parameters(change):
        
        selected_device_ID = wg_device_ID.value        
        device_parameters_flex = config_info['devices'][selected_device_ID]['device_params']
               
        if len(device_parameters_flex) > 0:
            
            device_parameters_flex = list(device_parameters_flex.keys())        
            wg_name1.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name1.value = 'none'

            wg_name2.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name2.value = 'none'

            wg_name3.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name3.value = 'none'

            wg_name4.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name4.value = 'none'

            wg_name5.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name5.value = 'none'

            wg_name6.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name6.value = 'none'

            wg_name7.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name7.value = 'none'

            wg_name8.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name8.value = 'none'

            wg_name9.options = ['none'] + sorted(parameters + device_parameters_flex)
            wg_name9.value = 'none'
       
    

    def button_record_pressed(b):
        """
        Save the names info in the config_info.json file.
        """

        button_record_output.clear_output(wait=True)
        names = [wg_name1.value, wg_name2.value, wg_name3.value, wg_name4.value, wg_name5.value, wg_name6.value, wg_name7.value, wg_name8.value, wg_name9.value]

        # remove the 'none' values from the list of names
        names = [x for x in names if x != 'none']
        

        with open(config_file, "r") as f:
            config = json.load(f)
            existing_filenamings = config['filenaming']
            if wg_device_ID.value in existing_filenamings.keys():
                
                existing_filenamings_device = existing_filenamings[wg_device_ID.value]

            else:                
                existing_filenamings[wg_device_ID.value] = {}
                existing_filenamings_device = existing_filenamings[wg_device_ID.value]

        
        # Update config with filenaming info
        existing_filenamings_device['raw'] = names
        config['filenaming'][wg_device_ID.value] = existing_filenamings_device

        # Save the updated config back to the JSON file
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)

            
        with button_record_output:            
            print(f'The filenaming of raw files for device {wg_device_ID.value} have been recorded in the config_info.json file.')

    recording.on_click(button_record_pressed)
    wg_device_ID.observe(change_parameters, names='value')
    

    display(ipw.VBox([wg_device_ID,wg_name1, wg_name2, wg_name3, wg_name4, wg_name5, wg_name6, wg_name7, wg_name8, wg_name9]))
    display(ipw.HBox([recording, button_record_output]))


def set_institution_info():    

    config_info = get_config_info()

    # define some widget
    wg_name = ipw.Text(
        description = 'Institution name',
        placeholder = 'Enter a name',            
        style = style,
        layout=Layout(width="50%", height="30px"),
    )

    wg_acronym = ipw.Text(
        description = 'Institution acronym',
        placeholder = 'Enter an acronym (optional)',            
        style = style,
        layout=Layout(width="50%", height="30px"),
    )

    wg_department = ipw.Text(
        description = 'Department',
        placeholder = 'Enter a department (optional)',            
        style = style,
        layout=Layout(width="50%", height="30px"),
    )

    wg_address = ipw.Text(
        description = 'Institution address',
        placeholder = 'Enter an address (optional)',            
        style = style,
        layout=Layout(width="50%", height="30px"),
    )

    recording = ipw.Button(
        description='Save',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',            
    )

    button_record_output = ipw.Output()

    
    # define the function to record the widgets values
    def button_record_pressed(b):
        """
        Save the institution info in the config_info.json file.
        """

        button_record_output.clear_output(wait=True)
        
        # Update config with user data
        config_info["institution"] = {
            "name": wg_name.value,
            "acronym": wg_acronym.value,
            "department": wg_department.value,
            "address": wg_address.value,                                
        }
        # Save the updated config back to the JSON file
        with open(config_file, "w") as f:
            json.dump(config_info, f, indent=4)

            
        with button_record_output:
            print(f'The institution {wg_name.value} info have been recorded in the {config_file.name} file.')


    # link the button widget to the aforementioned function  
    recording.on_click(button_record_pressed)

    # display the widgets
    display(ipw.VBox([wg_name, wg_acronym, wg_department, wg_address]))
    display(ipw.HBox([recording, button_record_output]))


def remove_devices_info():


    # retrieve the content of the config_info file
    config_info = get_config_info()


    # retrieve the devices ID
    devices_ID = config_info['devices'].keys()


    # create ipywidgets
    wg_device_ID = ipw.Dropdown(
        description='Device ID',
        options=devices_ID,
        placeholder='Select a device ID',
        style=style
    )


    # create widgets (recording)
    deleting = ipw.Button(
        description='Delete',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',            
    )

    button_delete_output = ipw.Output()


    # function to remove device_info
    def delete_button_pressed(change):

        button_delete_output.clear_output(wait=True)

        with open(config_file, "r") as f:
            config = json.load(f)
            existing_devices_info = config['devices']


        existing_devices_info.pop(wg_device_ID.value)
        config['devices'] = existing_devices_info

        # Save the updated config back to the JSON file
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)

            
        with button_delete_output:
            print(f'The info of device {wg_device_ID.value} have been deleted from the db_config.json file.')
        


    # set the button clcik event handler
    deleting.on_click(delete_button_pressed)


    # display the widgets
    display(wg_device_ID)
    display(ipw.HBox([deleting, button_delete_output]))
