In this section, we will show you how to process raw microfading files into our standard *interim* files, that is required to use the `microfading` package.

The current version of the `microfading` package can only process raw files obtained with the Fotonowy microfading device. Don't hesitate to contact us to discuss the possibility to process raw files obtained with other microfading systems.

Processing the rawdata files consists of two main steps:

1. Get a list of your raw files
2. Run the `process_rawdata()` function

## Get raw files

There are many ways in python to retrieve a list of files. I will only show two ways of doing it.

**Option 1** - Retrieve the files manually

In this option you create a list by typing the names of the raw files. To ease process, you can start writing the first letter and then press the `Tab` button to autocomplete the rest of the filename. 

```python
# For example, I created a list of two rawdata files
files = ['Project01_MF.plume01.01_G01_blue_rawdata.txt',
	     'Project12_MF.S0001.01_G01_paperBlock_rawdata.txt']

# If the files are located in a different directory than the notebook you are using, then
# you will need to give the absolute path of the files	     
files = ['/home/john/Documents/data/Project01_MF.plume01.01_G01_blue_rawdata.txt',
	     '/home/john/Documents/data/Project12_MF.S01.01_G01_paperBlock_rawdata.txt']
```

&nbsp;

**Option 2** - Retrieve with `glob`

```python
From glob import glob

# Ask glob to retrieve all the files containing the expression *MF* and *rawdata* in their filenames
# I also asked to sort the files
files = sorted(glob('*MF*rawdata*'))
files
```

<div class="output-area">
<pre>['/home/john/Documents/data/Project01_MF.plume01.01_G01_blue_rawdata.txt',
'/home/john/Documents/data/Project12_MF.S01.01_G01_paperBlock_rawdata.txt']
</pre>
</div>


## Run function

Once you have a list of rawdata files, you will use it inside the `process_rawdata()` function as illustrated below. This function has two compulsory arguments (*files* and *device*) for which you will need to provide values. You will use the list of rawfiles for the *files* argument and you will select the adequate microfading device you used (see sub-sections below).

```python
import microfading as mf

# get the raw files (see step above)
files = ['filename1', 'filename2', etc.]

# run the function to process the rawfiles
mf.process_rawdata(files=files, device='<MFT_device_name>')
```

### Fotonowy device

If you used a Fotonowy microfading device to obtain the raw files, then you will enter 'fotonowy' as value for the *device* argument in the `process_rawdata()` function (see code below).

```python
# run the function to process the rawfiles
mf.process_rawdata(files=files, device='fotonowy')
```

Currently, the Fotonowy device produces 4 raw files for each analysis, thus you will need to enter these 4 files in the list of required raw files. 


## Function parameters

The `process_rawdata()` function contains several parameters. I will let you read the docstrings for detailed information. But I will just focus on two parameters that you might find important to be aware of. 




