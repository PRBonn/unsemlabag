.PHONY: test

SHELL = /bin/sh

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

GPUS ?= 0
MACHINE ?= default
CONFIG ?= ''
CHECKPOINT ?= 'weights.ckpt'
DATA_PATH := 
FOLDER := .

RUN_IN_CONTAINER = docker run -it --gpus all -e DISPLAY=$DISPLAY -v $(FOLDER):/unsemlabag unsemlab-ag

build:
	docker build . --ssh default --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t unsemlab-ag

download:
	$(RUN_IN_CONTAINER) bash -c "./download_assets.sh"

train:
	$(RUN_IN_CONTAINER) python3 train.py
	
generate:
	$(RUN_IN_CONTAINER) python3 main.py

test:
	$(RUN_IN_CONTAINER) python3 test.py -w $(CHECKPOINT)

map_to_images:
	$(RUN_IN_CONTAINER) python3 map_to_dataset.py

shell:
	$(RUN_IN_CONTAINER) "bash"

freeze_requirements:
	pip-compile requirements.in > requirements.txt

