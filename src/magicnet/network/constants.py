from ..schema import *

TOKEN_DIM = 256
COMPONENT_DIM = 128
EXTERNAL_EMBEDDING_DIM = 3072
EXTERNAL_EMBEDDING_PCA_DIM = 32
FEED_FORWARD_DIM = 1024
N_FLOW_TRANSFORMS = 4
FLOW_DIM = 64
REL_ATTENTION_HEADS = 8
VALUE_HEAD_DIM = 256
BELIEF_NETWORK_LAYERS = 6
POLICY_TRUNK_LAYERS = 6
BELIEF_NETWORK_SAMPLES = 100

COLOR_WHEEL = ['white', 'blue', 'black', 'red', 'green', 'colorless']

COMPONENT_KIND_INDEX = {
    'scalar': 0,
    'vector': 1,
    'color': 2,
    'flag': 3,
    'categorical': 4,
}