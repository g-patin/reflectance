## Commands

### Datasets

* `get_datasets [MFT, rawfiles, BWS, stdev]` - Retrieve examples of data files.

### Databases management

* `create_DB [folder]` - Create two empty databases (projects and objects).
* `get_creators` - Retrieve a list of names corresponding to the persons that created the microfaded objects.
* `get_DB [db]` - Retrieve the databases.
* `get_institutions` - Retrieve a list of institutions owning the microfaded objects.
* `get_path_DB` - Retrieve the absolute path of the folder where the databases are stored.
* `get_persons` - Retrieve a list of persons that performed microfading analyses.
* `add_new_institution` - Add a new institution to the database.
* `add_new_person` - Add a new person to the list of persons doing microfading analyses.
* `add_new_project` - Add a new project to the database.
* `add_new_object` - Add a new object to the database.
* `update_DB_project [new, old]` - Modify or add a new parameter to the project database.
* `update_DB_object [new, old]` - Modify or add a new parameter to the object database.

### Data processing

* `process_rawdata [files, device]` - Process rawdata files into interim files.

### MFT class

* `data_points` - Select colourimetric values for one or multiple light dose values.

::: microfading.microfading
    rendering:
      show_root_heading: true
      show_source: true
