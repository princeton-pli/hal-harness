SHELL := /bin/bash
export PYTHONPATH := .
PYTHON ?= python3

# libs we need before running anything
REQ := pytest pytest_cov openai dotenv pymupdf pyreadr pandas numpy docker docx pypdf

STUDY ?= ./data/original/1
CODE_MODE ?= python
MODEL ?= gpt-4o
TIER ?= easy

.PHONY: check-deps install-dev test test-extractor test-generator test-all design-easy execute-easy

check-deps:
	$(PYTHON) core/check_deps.py $(REQ)

install-deps:
	pip install -r requirements-dev.txt

check-docker:
	@command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not in PATH"; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon not reachable (is it running? do you have permissions?)"; exit 1; }
	@echo "Docker: OK"

# extractor
extract-stage1: check-deps
	python -m info_extractor --stage stage_1 --difficulty easy --study-path $(STUDY) --model-name $(MODEL)

web-search: check-deps
	python -m info_extractor --stage web_search --difficulty easy --study-path $(STUDY) --model-name $(MODEL)

# generator module
design-easy: check-deps
	python -m generator --stage design --tier easy --study-path $(STUDY) --templates-dir ./templates --code-mode $(CODE_MODE) --model-name $(MODEL) --tier $(TIER)
execute-easy: check-deps
	python -m generator --stage execute --tier easy --study-path $(STUDY) --code-mode $(CODE_MODE) --model-name $(MODEL)

generate: design-easy execute-easy

# interpreter module
interpret-easy: check-deps
	python -m interpreter --tier easy --study-path $(STUDY) --model-name $(MODEL)

# full pipeline (extract -> design -> execute -> interpret)
pipeline-easy: extract-stage1 design-easy execute-easy interpret-easy

# validator module
evaluate-extract: check-deps
	python -m validator.cli.evaluate_extracted_info --extracted_json_path $(STUDY)/post_registration.json --expected_json_path $(STUDY)/expected_post_registration.json --output_path $(STUDY)
evaluate-design: check-deps	
	python -m validator.cli.evaluate_design_cli --extracted_json_path $(STUDY)/replication_info.json --reference_doc_path $(firstword $(wildcard $(STUDY)/human_preregistration.*)) --output_path $(STUDY)
evaluate-execute: check-deps
	python -m validator.cli.evaluate_execute_cli  --study_path $(STUDY)
evaluate-interpret: check-deps
	python -m validator.cli.evaluate_interpret_cli  --study_path $(STUDY) --reference_report_path $(firstword $(wildcard $(STUDY)/human_report.*))
evaluate-summary: check-deps
	python -m validator.cli.evaluate_summary_cli --study_path $(STUDY)

evaluate-pipeline-easy: evaluate-extract evaluate-design evaluate-execute evaluate-interpret evaluate-summary

# Evaluate all stages
evaluate-pipeline: evaluate-extract evaluate-design evaluate-execute evaluate-interpret

# test suite
test: check-deps
	pytest -q

test-extractor: check-deps
	pytest -q --maxfail=1 --disable-warnings --cov=info_extractor.extractor --cov-report=term-missing

test-generator: check-deps
	pytest -q --maxfail=1 --disable-warnings --cov=generator.design.easy --cov-report=term-missing

test-cov: check-deps
	pytest -q --cov=.

# run the whole suite (extractor + validator + generator) with test coverage
test-all: check-deps
	pytest -q --maxfail=1 --disable-warnings --cov=. --cov-report=term-missing
