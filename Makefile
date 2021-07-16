installdeps:
	poetry install

servedeployFlag:
	celery -A deployflag worker -l INFO -O fair --pool=prefork -Q deployflag_training \
	--concurrency=$(CELERY_CONCURRENCY) --without-mingle --without-gossip --hostname deployflagworker@%h

test:
	pytest --cov-report xml --cov=tests/

.DEFAULT_GOAL := test
