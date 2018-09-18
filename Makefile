lint:
	flake8 --ignore=N803,N806 --exclude models .
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