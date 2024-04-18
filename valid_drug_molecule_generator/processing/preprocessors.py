import sys
sys.path.append("./")
from concurrent.futures import ThreadPoolExecutor
import os
import re
import tensorflow as tf
from keras.layers import TextVectorization
from keras.utils import to_categorical

from valid_drug_molecule_generator.config import config



class preprocess_data:

    def fit(self,X=None,y=None):
        self.input_text_vectorization_layer = TextVectorization(max_tokens=len(config.BASE_VOCABULARY)+2,standardize=None,
                                            split="character",
                                            output_sequence_length=config.MAX_INPUT_SEQUENCE_LEN,
                                            vocabulary=config.BASE_VOCABULARY)
        
        self.output_text_vectorization_layer = TextVectorization(max_tokens=len(config.BASE_VOCABULARY)+2,standardize=None,
                                                   split="character",
                                                   output_sequence_length=config.MAX_OUTPUT_SEQUENCE_LEN,
                                                   vocabulary=config.BASE_VOCABULARY)

        return self

    
    
    def transform(self,X,y=None):

        X = self.input_text_vectorization_layer(X).numpy()
        Y = self.output_text_vectorization_layer(y).numpy()

        return X,Y


        """
        def convert_to_source_str(molecule_str):
            return molecule_str.strip("\n")
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            self.preprocessed_input_data = list(pool.map(convert_to_source_str,self.preprocessed_data))


        
        def convert_to_dst_str(molecule_str):
            return molecule_str.strip("<")
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            self.preprocessed_output_data = list(pool.map(convert_to_dst_str,self.preprocessed_data))



        self.X = input_text_vectorization_layer(self.preprocessed_input_data).numpy()
        self.Y = output_text_vectorization_layer(self.preprocessed_output_data).numpy()

        return self.X, self.Y
        """

        



        