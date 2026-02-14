The functions provided by the reflectance package only work on an instance of a reflectance spectroscopy class (RS). The creation of an instance of a reflectance class is a two-steps procedure and is described below:

## 1. **Get RS data files**

Retrieve a list of data files on which the functions provided by the class will be applied. The data files need to be constrained to a specific file structure (For more information about it, see the [data files section](https://g-patin.github.io/reflectance/datafiles)).

```python
import reflectance as rf
```

```python
# Here we are using the test data, but you should use your own data files
files = rf.get_datasets()
```
	
&nbsp;



## 2. **Create RS class instance**

Create the instance using the files as the main parameter.

```python
rs = rs.RS(files=files)
rs
```
<div class="output-area">
<pre>
Reflectance data class - Number of files = 5
</pre>
</div>

