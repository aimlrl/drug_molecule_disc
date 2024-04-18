import os
import pandas as pd
import tensorflow as tf

from valid_drug_molecule_generator.config import config

def load_dataset(file_name):

    file_path = os.path.join(config.DATAPATH,file_name)
    _data = pd.read_csv(file_path)
    return _data


def save_model(model_to_save):

    saved_file_name = "enc_dec_drug_molecule_gen.keras"
    save_path = os.path.join(config.SAVED_MODEL_PATH,saved_file_name)
    model_to_save.save(save_path)
    print("Model saved in ",saved_file_name)


def load_nn_model(model_to_load):

    save_path = os.path.join(config.SAVED_MODEL_PATH,model_to_load)
    trained_model = tf.keras.models.load_model(save_path)
    return trained_model