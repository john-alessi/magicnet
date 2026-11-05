from ..schema import *

TOKEN_DIM = 256
COMPONENT_DIM = 128
EXTERNAL_EMBEDDING_DIM = 3072
FEED_FORWARD_DIM = 1024
N_FLOW_TRANSFORMS = 4
FLOW_DIM = 256

COLOR_WHEEL = ['white', 'blue', 'black', 'red', 'green', 'colorless']

COMPONENT_KIND_INDEX = {
    'scalar': 0,
    'vector': 1,
    'color': 2,
    'flag': 3,
    'categorical': 4,
}