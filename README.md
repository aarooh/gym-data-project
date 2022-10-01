# gym-data-project


## Installing virtual environment


Before running this pipeline you should install Python Virtualenv
~~~
cd <repo>

pip install virtualenv

virtualenv env 
~~~

## Activating virtual environment

###Unix / macOS
Confirm activation with
~~~
source env/bin/activate
~~~

~~~
which python
~~~
it should return .../env/bin/python



---
###Windows 
Confirm activation with
~~~
.\env\Scripts\activate
~~~
~~~
where python
~~~
it should return ...\env\Scripts\python.exe

## Installing required packages
~~~
pip install -r requirements.txt
~~~

## Running tests
We are using pytest library to make and run tests

run command:
~~~
pytest test/ in project root
~~~

## Running the pipeline

If you are using vscode, you can use .vscode/launch.json to launch this pipeline script.
Or you can also use command:
~~~
python3 main.py -c configs/config.yml
~~~