# The MLflow tracking server and registry. The client side uses mlflow-skinny 3.17.0;
# the server matches it and adds the Postgres driver (psycopg 3, which its SQLAlchemy URL now expects).
FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir mlflow==3.17.0 "psycopg[binary]==3.3.6"
