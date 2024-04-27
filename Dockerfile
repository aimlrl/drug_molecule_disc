FROM python:3.10.14-bookworm

RUN pip install --upgrade pip

COPY . /app

WORKDIR /app

ENV PYTHONPATH=${PYTHONPATH}:/valid_drug_molecule_generator

RUN chmod +x valid_drug_molecule_generator/train_pipeline.py

RUN chmod +x valid_drug_molecule_generator/predict.py

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]

CMD ["./valid_drug_molecule_generator/predict.py"]