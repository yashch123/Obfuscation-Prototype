# Obfuscation-Prototype
A prototype neural network that runs on obfuscated inputs.

## Installation
Installation support is on Ubuntu. 

### Virtual Machine (on Ubuntu)
To run on a virtual environment, first run the command

```shell
$ make virtual
```

This will install on your LOCAL machine a copy of virtualenv, which can be used for creating a virtual environment on your Ubuntu workstation. To run the activate the virtual machine, run the following command:

```shell
$ source ./venv/bin/activate
```

Now you will be running in a virtual environment and can start running the neural network on the obfuscated inputs.

### Building
Run the command
```shell
$ make build
```

to install the appropriate packages (e.g. tensorflow).
