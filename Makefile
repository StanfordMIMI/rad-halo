PYTHON ?= python
ANALYSIS = scripts/analysis

.PHONY: analysis statistics tables figures test lint check

# Reproduce every number, table and figure of the paper (CPU, ~1 min).
analysis: statistics tables figures

statistics:
	$(PYTHON) $(ANALYSIS)/dataset_statistics.py
	$(PYTHON) $(ANALYSIS)/agreement.py
	$(PYTHON) $(ANALYSIS)/detector_metrics.py

tables:
	$(PYTHON) $(ANALYSIS)/make_tables.py

figures:
	$(PYTHON) $(ANALYSIS)/make_figures.py

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check . && ruff format --check .

check:
	$(PYTHON) tools/check_leaks.py
