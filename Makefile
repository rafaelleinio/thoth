# globals
VERSION := $(shell grep __version__ thoth/__metadata__.py | head -1 | cut -d \" -f2 | cut -d \' -f2)

.PHONY: requirements-dev
## install development requirements
requirements-dev:
	@pip install -U -r requirements.dev.txt

.PHONY: requirements-minimum
## install prod requirements
requirements-minimum:
	@pip install -U -r requirements.txt

.PHONY: requirements
## install requirements
requirements: requirements-dev requirements-minimum


.PHONY: tests
## run unit tests with coverage report
tests:
	@echo ""
	@echo "Tests"
	@echo "=========="
	@echo ""
	@pytest -W ignore::DeprecationWarning --cov-report term --cov-report html:tests-cov --cov=thoth --cov-fail-under=100 tests/

.PHONY: unit-tests
## run unit tests with coverage report
unit-tests:
	@echo ""
	@echo "Unit Tests"
	@echo "=========="
	@echo ""
	@pytest -W ignore::DeprecationWarning --cov-config=.coveragerc --cov-report term --cov=thoth tests/unit

.PHONY: integration-tests
## run integration tests with coverage report
integration-tests:
	@echo ""
	@echo "Integration Tests"
	@echo "================="
	@echo ""
	@python -m pytest -W ignore::DeprecationWarning --cov-config=.coveragerc --cov-report term -cov-report html:unit-tests-cov --cov=thoth tests/integration

.PHONY: style-check
## run code style checks with black
style-check:
	@echo ""
	@echo "Code Style"
	@echo "=========="
	@echo ""
	@black --check -t py36 --exclude="build/|buck-out/|dist/|_build/|pip/|\.pip/|\.git/|\.hg/|\.mypy_cache/|\.tox/|\.venv/" . && echo "\n\nSuccess" || (echo "\n\nFailure\n\nRun \"make black\" to apply style formatting to your code"; exit 1)

.PHONY: quality-check
## run code quality checks with flake8
quality-check:
	@echo ""
	@echo "Flake 8"
	@echo "======="
	@echo ""
	@flake8 && echo "Success"
	@echo ""

.PHONY: type-check
## run code type checks with mypy
type-check:
	@echo ""
	@echo "Mypy"
	@echo "======="
	@echo ""
	@mypy --install-types --non-interactive thoth && echo "Success"
	@echo ""

.PHONY: checks
## run all code checks
checks: style-check quality-check type-check

.PHONY: apply-style
## fix stylistic errors with black and isort
apply-style:
	@black -t py36 --exclude="build/|buck-out/|dist/|_build/|pip/|\\.pip/|\.git/|\.hg/|\.mypy_cache/|\.tox/|\.venv/" .
	@isort thoth/ tests/

.PHONY: package
## build thoth package wheel
package:
	@python -m setup sdist bdist_wheel

.PHONY: version
## show version
version:
	@echo "$(VERSION)"

.PHONY: app
## create db infra with docker compose
app:
	@docker compose build
	@docker compose -f docker-compose.yaml up ui

.PHONY: notebook-examples
## run a jupyter notebook service to easily run the examples
notebook-examples:
#	@docker build --target thoth -t thoth .
#	@docker run -p 8888:8888 thoth jupyter notebook --allow-root --ip 0.0.0.0 --NotebookApp.token='' --NotebookApp.password='' --no-browser --notebook-dir=examples/
	@docker compose -f docker-compose.yaml up notebook

.PHONY: teardown
## teardown all infra on docker compose
teardown:
	@docker compose -f docker-compose.yaml down

.PHONY: clean
## clean unused artifacts
clean:
	@find ./ -type d -name 'dist' -exec rm -rf {} +;
	@find ./ -type d -name 'build' -exec rm -rf {} +;
	@find ./ -type d -name 'thoth.egg-info' -exec rm -rf {} +;
	@find ./ -type d -name 'htmlcov' -exec rm -rf {} +;
	@find ./ -type d -name '.pytest_cache' -exec rm -rf {} +;
	@find ./ -type d -name 'spark-warehouse' -exec rm -rf {} +;
	@find ./ -type d -name 'metastore_db' -exec rm -rf {} +;
	@find ./ -type d -name '.ipynb_checkpoints' -exec rm -rf {} +;
	@find ./ -type f -name 'coverage-badge.svg' -exec rm -f {} \;
	@find ./ -type f -name 'coverage.xml' -exec rm -f {} \;
	@find ./ -type f -name '.coverage*' -exec rm -f {} \;
	@find ./ -type f -name '*derby.log' -exec rm -f {} \;
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '*.pyo' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;

.DEFAULT_GOAL := help
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
