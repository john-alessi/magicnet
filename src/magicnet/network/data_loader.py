import torch
from .constants import *

def load_data(
    state: list[dict],
    history: list[dict],
    catalog: any,
    emb_dim = 3072,
) -> dict[str, torch.Tensor]:
    # each game object component should become a [key, value, kind, idx]
    # vector (where idx corresponds to the parent object index)
    components = []
    card_names = set(
        [
            c['name']
            for c in state
            if c['type'] in ['game_object'] and c['name'] != 'HIDDEN'
        ]
    )
    card_metadata = catalog.lookup(card_names)

    for i,gobj in enumerate(state):
        if 'components' in gobj:
            for k,v in gobj['components'].items():
                if k == 'colors':
                    onehot = [
                        int(c in v)
                        for c in COLOR_WHEEL
                    ]
                    components += [(k, onehot, 'color', i)]
                elif isinstance(v, bool) and v is True:
                    components += [(k, None, 'flag', i)]
                elif isinstance(v, (int,float)):
                    components += [(k, v, 'scalar', i)]
                elif isinstance(v, list):
                    for f in v:
                        components += [(k, f, 'vector', i)]
                else:
                    components += [(k, v, 'vector', i)]
    
    keys = set(k for k,_,_,_ in components)
    key_embeddings = dict(zip(keys, catalog.key_vectors(keys)))

    vectors = [v for (_,v,kd,_) in components if kd == 'vector']
    vector_embeddings = dict(zip(vectors, catalog.embed_strings(vectors)))

    packed_scalars = torch.tensor([
        list(key_embeddings[k]) + [v, i]
        for (k,v,kd,i) in components
        if kd == 'scalar'
    ])

    packed_vectors = torch.tensor([
        list(key_embeddings[k]) + list(vector_embeddings[v]) + [i]
        for (k,v,kd,i) in components
        if kd == 'vector'
    ])

    packed_flags = torch.tensor([
        list(key_embeddings[k]) + [i]
        for (k,_,kd,i) in components
        if kd == 'flag' 
    ])

    packed_colors = torch.tensor([
        list(key_embeddings[k]) + v + [i]
        for (k,v,kd,i) in components
        if kd == 'color' 
    ])

    return {
        'scalars': packed_scalars,
        'vectors': packed_vectors,
        'colors': packed_colors,
        'flags': packed_flags,
    }