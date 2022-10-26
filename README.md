# SmartEDU Automatic Summarization Tool

This tool is intended to concentrate all methods of automatic text summarization used in the SmartEDU project.

## Installing

Two installation methods are supported: `pipenv` and `pip`.

### pipenv

For installing the dependencies with pipenv, simply run

```bash
pipenv install
```

This will create a new virtual environment with the configurations and packages within `Pipfile` and `Pipfile.lock`. This will also create the environment with the correct python version, so there is no need to pay attention to it using this method.

### pip

In order to use pip, make sure to use the correct python version used originally in the project: **python 3.9**. To install the dependencies from the `requirements.txt` file, run

```bash
pip install -r requirements.txt
```

## Running

The methods are implemented as python modules. Therefore, to run any method, please run

```
python -m smartedu-summarizer.methods.<method>
```

`<method>` must be replaced with the desired summarization method. There are current eight methods that can be run:

- `textrank`
- `lexrank`
- `tf_idf`
- `QueSTS`
- `lsa`
- `lexicalchains`
- `Pegasus`
- `Distillbart`

We recommend running each method first with the `--help` argument in order to obtain more information about their arguments.


### Verbose levels

The code is full of logging messages to allow the best visualization of what is happening during execution. There are three levels of verbosity.

- No argument: no message will be printed;
- INFO (`-v`): Messages about the current step being executed will be printed; 
- DEBUG (`-vv`): Not only messages about the current step, but also the actual returning values of each step will also be presented.

## Contributing

If you want to contribute with the project, just keep in mind that the methods must be developed as python modules. They must have their own folders inside `smartedu-summarizer/methods/`. The interface between the user and the method should be implemented in a `__main__.py` file, with meaningful help messages for the arguments.

In order to keep the code organized, we encourage any developer to keep the actual method implementation in a separate file in the method's folder, such as `summarize.py`, and keep `__main__.py` exclusively to gathering the arguments and configuring the logging tool.
