CUR_DIR := ${CURDIR}
OS := $(shell uname)

.PHONY: clean
clean:
	@echo "cleaning up..."
	@rm -rf ${CUR_DIR}/data/input
	poetry env remove --all
	@rm -rf ${CUR_DIR}/.venv
	@rm -rf ${CUR_DIR}/.snakemake

.PHONY: set_pyenv
set_pyenv:
	@echo "setting up pyenv..."
	@pyenv install 3.11 -s

.PHONY: setup
setup: clean set_pyenv
	@echo "setting up..."
ifeq ($(OS),Darwin)
	@echo "Mac"
else
	@echo "Linux"
	# python3 -m keyring --disable
endif
	@poetry config virtualenvs.create true
	@poetry config virtualenvs.in-project true
	@poetry install
	@poetry run jupyter contrib nbextension install --user
	@poetry run jupyter nbextension enable --py codeium --user
	@poetry run jupyter serverextension enable --py codeium --user
	# @poetry run kaggle competitions download -c pii-detection-removal-from-educational-data -p data/input/
	# @unzip -o data/input/pii-detection-removal-from-educational-data.zip -d data/input/
	# @rm data/input/pii-detection-removal-from-educational-data.zip
	# # poetry run python process_to_quality/generate_features.py

.PHONY: jupyter
jupyter:
	@echo "starting jupyter..."
	@poetry run jupyter notebook --no-browser
