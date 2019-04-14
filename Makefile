SHELL = /bin/bash

default:
	echo "No default builds. To run on a virtual environment, run 'make virtual', and then run 'source ./venv/bin/activate'. Then run 'make build'."
pip:
	sudo apt update
	sudo apt install python-dev python-pip

build: pip
	pip install --upgrade tensorflow

# Run this if you're intending to run a virtual environment. Then execute the command
# source ./venv/bin/activate
virtual: pip
	# virtualenv gives us the ability to run a VM
	sudo pip install -U virtualenv  # system-wide install
	# Create a new virtual environment by choosing a Python interpreter and making a ./venv directory to hold it:
	virtualenv --system-site-packages -p python2.7 ./venv
