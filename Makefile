CUR_DIR := ${CURDIR}
OS := $(shell uname)

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
