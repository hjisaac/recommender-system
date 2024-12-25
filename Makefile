SHELL := /bin/bash
.PHONY: all clean

# Define the Python interpreter
PYTHON = python3

# Define the test commands
TEST_CMD = $(PYTHON) -m unittest discover -p 'test_*.py'
COVERAGE_CMD = poetry run coverage run --source=. -m unittest discover -p 'test_*.py'
COVERAGE_REPORT_CMD = poetry run coverage report -m
COVERAGE_XML_CMD = poetry run coverage xml
COVERAGE_HTML_CMD = poetry run coverage html

help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

compile:
	jupyter nbconvert --to script main.ipynb

# Default target
.PHONY: test
test:
	$(TEST_CMD)

# Target to run tests with verbose output
.PHONY: test_verbose
test_verbose:
	$(TEST_CMD) -v

# Target to run tests with coverage report
.PHONY: test_coverage
test_coverage:
	$(COVERAGE_CMD)
	$(COVERAGE_REPORT_CMD)

# Clean up coverage files
.PHONY: clean
clean:
	find . | grep -E "(__pycache__|\.pyc$)" | xargs rm -rf
	rm -rf .coverage*

# Run all tests and generate HTML coverage report
.PHONY: test_html_report
test_html_report:
	$(COVERAGE_CMD)
	$(COVERAGE_HTML_CMD)

# Run all tests and generate XML coverage report
.PHONY: test_xml_report
test_xml_report:
	$(COVERAGE_CMD)
	$(COVERAGE_XML_CMD)
