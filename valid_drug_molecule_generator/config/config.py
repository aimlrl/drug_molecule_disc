import pathlib
import os
import valid_drug_molecule_generator

PACKAGE_ROOT = pathlib.Path(valid_drug_molecule_generator.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")
SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT,"trained_models")

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

MAX_INPUT_SEQUENCE_LEN = 52
MAX_OUTPUT_SEQUENCE_LEN = 52

BASE_VOCABULARY = ['6','p','4','7','r','[','o','+','H','\\','s','#','C','<',
                    '/','%','N','0','=','.','c','3','5','S','l','1','B','I','\n','(','O',
                    '-',')',']','8','P','9','n','F','2']

INPUT_VOCABULARY = ['','[UNK]','6','p','4','7','r','[','o','+','H','\\','s','#','C','<',
                    '/','%','N','0','=','.','c','3','5','S','l','1','B','I','\n','(','O',
                    '-',')',']','8','P','9','n','F','2']

OUTPUT_VOCABULARY = ['','[UNK]','6','p','4','7','r','[','o','+','H','\\','s','#','C','<',
                     '/','%','N','0','=','.','c','3','5','S','l','1','B','I','\n','(','O',
                     '-',')',']','8','P','9','n','F','2']
