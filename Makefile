CUR_DIR := ${CURDIR}
OS := $(shell uname)
CORES ?= all
DATA_PATH ?= data
SCRIPT_PATH ?= kaggle_template/scripts

.PHONY: clean_snakemake
clean_snakemake:
	@echo "cleaning up snakemake..."
	@rm -rf ${CUR_DIR}/.snakemake
	@rm -rf ${CUR_DIR}/data
	@rm -rf ${CUR_DIR}/dag.pdf

.PHONY: clean
clean: clean_snakemake
	@echo "cleaning up..."
	poetry env remove --all
	@rm -rf ${CUR_DIR}/.venv
	@rm -rf ${CUR_DIR}/core.*

.PHONY: set_pyenv
set_pyenv:
	@echo "setting up pyenv..."
	@pyenv install 3.11 -s

.PHONY: setup
setup: set_pyenv
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

.PHONY: jupyter
jupyter:
	@echo "starting jupyter..."
	@poetry run jupyter notebook --no-browser

.PHONY: format
format:
	@echo "formatting..."
	@poetry install
	# add isort
	@poetry run isort .
	@poetry run black .

.PHONY: cs
cs: clean_snakemake

# to run this on kaggle, run: make snakemake DATA_PATH=/kaggle/input SCRIPT_PATH=scripts
.PHONY: snakemake
snakemake: clean_snakemake
	@make setup
	@echo "running snakemake with $(CORES) cores..."
	@poetry run snakemake all --cores $(CORES) --config base_data_path=$(DATA_PATH) base_script_path=$(SCRIPT_PATH) --nolock --ignore-incomplete

.PHONY: snakemake_kaggle
snakemake_kaggle:
	snakemake --cores all --config base_script_path=kaggle_template/scripts base_data_path=data --nolock --ignore-incomplete
