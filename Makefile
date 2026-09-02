.PHONY: install report app test
install:
	python -m pip install -r requirements.txt
report:
	python -m src.pipeline --print-summary
app:
	streamlit run streamlit_app.py
test:
	pytest -q

source-check:
	python scripts/check_sources.py
