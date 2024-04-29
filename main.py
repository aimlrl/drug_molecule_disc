from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from keras.models import load_model
import numpy as np
import pandas as pd
import random

from valid_drug_molecule_generator.predict import generate_molecule
from valid_drug_molecule_generator.config import config
from valid_drug_molecule_generator.processing.data_management import load_dataset,load_nn_model
import valid_drug_molecule_generator.processing.preprocessors as pp


saved_file_name = "enc_dec_drug_molecule_gen.keras"
loaded_model = load_nn_model(saved_file_name)

cv_data = load_dataset(config.TEST_FILE)
preprocess = pp.preprocess_data()
preprocess.fit()
X_cv,_ = preprocess.transform(cv_data["X"],cv_data["Y"])



app = FastAPI(title="Drug Molecule Discovery API",
              description="An API to randomly generate a chemical formula of a valid drug molecule",
              version="0.1")

origins = ["*"]

app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


class MoleculeGenerator(BaseModel):
    start: str

@app.get('/')
def index():
    return {'message':'An App for generating chemical formula of a valid drug molecule'}


@app.post('/Generate_Molecule')
def gen_molecule(trigger: MoleculeGenerator):

    input = trigger.start
    #input = input_data['start']

    if input == 'y' or input == 'Y' or input == 'start' or input == 'Start':
        single_input = X_cv[random.randint(0,X_cv.shape[0]),:]
        single_input = single_input.reshape(1,single_input.shape[0])
        gen_mol = generate_molecule(single_input)
        return gen_mol




if __name__ == '__main__':
    uvicorn.run(app)

