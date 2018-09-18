lint:
	flake8 .
	pydocstyle .
	mypy .

test:
	pytest --cov=. -k 'not slow'

ci-test:
	pytest --cov=.

install:
	pip install -r requirements.txt
	python setup.py

dev-install:
	pip install -r requirements-dev.txt
	python setup.py develop