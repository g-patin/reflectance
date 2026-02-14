This page describes how to retrieve examples of interim files that can subsequently be used to explore the microfading package. 

First, open a jupyter notebook, import the microfading package and then retrieve the paths using the `get_datasets()` function, as illustrated below. The `get_datasets()` function has four optional parameters that allows you to filter the files:

- `MFT` (string) : select raw files for a given microfading device (by default 'fotonowy'). For now, only raw files from Fotonowy are available.
- `rawfiles` (True or False) : you can decide to retrieve rawdata files or interim files. The former will enable you to run the `process_rawdata()` function, while the latter will allow you to directly create an `MFT` class instance and explore the functionalities of the package.
- `BWS` (True or False) : you can decide to include or exclude measurements performed on blue wool samples.
- `stdev` (True or False) : you can decide to retrieve file with or without standard deviation values. This parameter only applies for the *interim* files.


```python
import microfading as mf
```

```python
# by default rawfiles=False, BWS=True, stdev=False
ds = mf.get_datasets()
ds
```
<div class="output-area">
<pre>
[PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-144_MF.BWS0026.04_G02_BW3_model_2024-08-02_MFT1.xlsx'),
PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-144_MF.BWS0025.04_G02_BW2_model_2024-08-02_MFT1.xlsx'),
PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-144_MF.BWS0024.04_G02_BW1_model_2024-08-02_MFT1.xlsx'),
PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-144_MF.yellowwood.01_G01_yellow_model_2024-08-01_MFT1.xlsx'),
PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-144_MF.vermillon.01_G01_red_model_2024-07-31_MFT1.xlsx')]
</pre>
</div>


&nbsp;


```python
# retrieve all the rawfiles
ds = mf.get_datasets(rawfiles=True)
ds
```
<div class="output-area">
<pre>
[PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-8200 P-001 G01 uncleaned_01-spect_convert.txt'),
PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-8200 P-001 G01 uncleaned_01-spect.txt'),
PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-8200 P-001 G01 uncleaned_01.txt'),
PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-8200 P-001 G01 uncleaned_01.rfc'),
PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-144 BWS0024 G01 BW1_01-spect_convert.txt'),
PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-144 BWS0024 G01 BW1_01-spect.txt'),
PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-144 BWS0024 G01 BW1_01.txt'),
PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-144 BWS0024 G01 BW1_01.rfc')]
</pre>
</div>


&nbsp;


```python
# excluded the rawfiles performed on blue wool samples
ds = mf.get_datasets(rawfiles=True, BWS=False)
ds
```
<div class="output-area">
<pre>
[PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-8200 P-001 G01 uncleaned_01-spect_convert.txt'),
PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-8200 P-001 G01 uncleaned_01-spect.txt'),
PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-8200 P-001 G01 uncleaned_01.txt'),
PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-8200 P-001 G01 uncleaned_01.rfc')]
</pre>
</div>


&nbsp;


```python
# retrieved the interim files with standard devivation values
ds = mf.get_datasets(rawfiles=False, BWS=False, stdev=True)
ds
```
<div class="output-area">
<pre>
[PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-144_MF.dayflower4.G01_avg_0h_model_2024-07-30_MFT2.xlsx'),
PosixPath('/home/gpatin/Documents/test/venv/lib/python3.11/site-packages/microfading/datasets/2024-144_MF.indigo3.G01_avg_0h_model_2024-08-02_MFT2.xlsx')]
</pre>
</div>


