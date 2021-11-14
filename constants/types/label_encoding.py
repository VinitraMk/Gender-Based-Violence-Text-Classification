from enum import Enum

class LabelEncoding(str, Enum):
    LABEL_ENCODING = 'LABEL_ENCODING'
    ORDINAL_ENCODING = 'ORDINAL_ENCODING'
    ONEHOT_ENCODING = 'ONEHOT_ENCODING'