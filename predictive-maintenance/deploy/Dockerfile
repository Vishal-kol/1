FROM python:3.9-slim

WORKDIR /app

COPY models/xgb_v1.pkl ./models/xgb_v1.pkl
COPY inference.py .

RUN pip install joblib pandas xgboost scikit-learn kserve

CMD ["python", "inference.py"]
