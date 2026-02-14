Creating databases is an operation that only needs to be performed one time using the `create_DB()` function (see below). Inside a desired folder on your local computer, it will create a few empty files (csv and txt) in which information about reflectance spectroscopy projects and objects can be recorded. 

```python
import reflectance as rf
```


```python
folder = "Enter a desired folder path as a string"
# e.g: "home/john/Documents/RS/databases"

mf.create_DB(folder=folder)
```

&nbsp;

## **Databases created ?**

If you want to know whether databases were created, use the `DB()` function.

```python
import reflectance as rf

# Return True if databases files were created otherwise False
rf.is_DB()

```
<div class="output-area">
<pre>
True
All the databases were created and can be found in the following directory: /home/john/Documents/RS/databases
</pre>
</div>

&nbsp;

## **Databases location ?**

If you want to know where are the databases files located, use the  `get_DB_path()` function. It will return the path of the folder where the databases are located.

```python
import reflectance as rf

# Return the path were the databases files are located if any.
rf.get_DB_path()
```


<div class="output-area">
<pre>
The databases are located in the folder: /home/john/Documents/RS/databases
</pre>
</div>

