.PHONY: install dev test lint clean freeze help

# Detecta binários do venv (cross-platform friendly — em Windows use WSL)
VENV_BIN := .venv/bin

help:
	@echo "Targets disponíveis:"
	@echo "  install   Cria .venv e instala requirements.txt"
	@echo "  dev       Roda pipeline completo em modo dev (SP, 3 eleições, 100 mun.)"
	@echo "  test      Roda pytest"
	@echo "  lint      Roda ruff em src/ e scripts/"
	@echo "  freeze    Regenera requirements.txt com versões resolvidas"
	@echo "  clean     Remove artefatos de data/interim, data/processed, models/"

install:
	python -m venv .venv
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt

dev:
	$(VENV_BIN)/python scripts/01_ingest.py
	$(VENV_BIN)/python scripts/02_build_panel.py
	$(VENV_BIN)/python scripts/03_features.py
	$(VENV_BIN)/python scripts/04_hypotheses.py
	$(VENV_BIN)/python scripts/05_train.py
	$(VENV_BIN)/python scripts/06_predict.py

test:
	$(VENV_BIN)/python -m pytest tests/ -v

lint:
	$(VENV_BIN)/python -m ruff check src/ scripts/

freeze:
	$(VENV_BIN)/pip freeze > requirements.txt

clean:
	rm -rf data/interim/* data/processed/* models/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
