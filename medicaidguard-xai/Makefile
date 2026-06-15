.PHONY: install data train experiments test lint api dashboard docker clean all

install:
	pip install -r requirements.txt && pip install -e .

data:
	python scripts/generate_data.py

train:
	python scripts/train.py

experiments:
	python scripts/run_experiments.py

test:
	python -m pytest tests -q

lint:
	ruff check src tests scripts dashboard

api:
	uvicorn medicaidguard.api.main:app --app-dir src --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

docker:
	docker compose up --build

clean:
	rm -rf data/synthetic/*.parquet artifacts reports/results/*.csv reports/tables/*.csv
	find . -name __pycache__ -type d -exec rm -rf {} +

all: data train experiments test
