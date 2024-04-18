import sys
sys.path.append("./")
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from keras.layers import Input, Embedding, LSTM, Attention, Concatenate, Dense
from keras.models import Model
from keras.metrics import CategoricalAccuracy

from valid_drug_molecule_generator.config import config
import valid_drug_molecule_generator.processing.preprocessors as pp

def encoder_decoder_with_attn_mech():
    
    enc_input = Input(shape=(None,),name="input_to_encoder")
    enc_embedding = Embedding(input_dim=len(config.BASE_VOCABULARY)+2,
                              output_dim=(len(config.BASE_VOCABULARY)+2)//2,
                              input_length=config.MAX_INPUT_SEQUENCE_LEN,
                              name="encoder_embedding_layer")(enc_input)
    enc_lstm_output,enc_last_hidden_state,enc_last_cell_state = LSTM(units=(len(config.BASE_VOCABULARY)+2)//2,
                                                       return_state=True, return_sequences=True,
                                                       name="encoder_lstm_layer")(enc_embedding)
    
    
    dec_input = Input(shape=(None,),name="input_to_decoder")
    dec_embedding = Embedding(input_dim=len(config.BASE_VOCABULARY)+2,
                              output_dim=(len(config.BASE_VOCABULARY)+2)//2,
                              input_length=config.MAX_OUTPUT_SEQUENCE_LEN,
                              name="decoder_embedding_layer")(dec_input)
    dec_lstm_layer = LSTM(units=(len(config.BASE_VOCABULARY)+2)//2,return_sequences=True,
                          return_state=True,name="decoder_lstm_layer")
    dec_lstm_output,_,_ = dec_lstm_layer(inputs=dec_embedding,
                                         initial_state=[enc_last_hidden_state,enc_last_cell_state])
    
    
    dec_enc_attn_seq = Attention()([dec_lstm_output,enc_lstm_output])
    dec_dense_input = Concatenate()([dec_lstm_output,dec_enc_attn_seq])
    
    dec_output = Dense(units=len(config.BASE_VOCABULARY)+2,activation="softmax",
                       name="decoder_output")(dec_dense_input)
    
    
    drug_discovery_model = Model(inputs=[enc_input,dec_input],outputs=dec_output)

    drug_discovery_model.compile(loss="categorical_crossentropy",metrics=["Accuracy"])

    return drug_discovery_model

"""
drug_discovery_model = KerasClassifier(model=encoder_decoder_with_attn_mech(),
                                       loss="categorical_crossentropy",metrics=[CategoricalAccuracy()])

drug_discovery_pipeline = Pipeline([("Data Preprocessor", pp.preprocess_data()),
                                    ("Drug Discovery Model",drug_discovery_model)])
"""