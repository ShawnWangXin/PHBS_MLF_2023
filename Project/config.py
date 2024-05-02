
import os
from os import path

class Config:

    root = os.getcwd()
    MODEL_PATH = path.join(root, "model/")
    DATA_PATH = path.join(root, 'data')
    RESULT_PATH = path.join(root, 'result')

    # RAWDATA_PATH = path.join(DATA_PATH, 'rawdata')
    # DERIVEDDATA_PATH = path.join(DATA_PATH, 'deriveddata')


