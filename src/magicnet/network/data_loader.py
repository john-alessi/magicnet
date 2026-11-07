import torch
from .constants import *
import time

def load_game(
    games: list[dict],
    catalog: any,
    emb_dim = 3072,
) -> dict[str, torch.Tensor]:
    # each game object component should become a [key, value, kind, idx]
    # vector (where idx corresponds to the parent object index)

    card_names = set(
        [
            c['name']
            for g in games
            for c in g['state']
            if 'name' in c and c['type'] in ['game_object'] and c['name'] != 'HIDDEN'
        ]
    )
    card_metadata = catalog.lookup(card_names)

    object_components = []
    event_components = []
    edges = []
    batch_indices = []
    unknown_cards = []
    max_L = 0
    # can probably remove i_batch, just return a list of batch indices
    # of each *object* (not component) and just keep the global
    # component index

    i = 0
    for i_batch,game in enumerate(games):
        max_L = max(max_L, len(game['state']) + len(game['history']))
        for event in game['history']:
            if 'components' in event:
                for k,v in event['components'].items():
                    if k in VOCAB_LOOKUP.keys():
                        event_components += [(k, v, 'categorical', i)]
                    elif isinstance(v, bool) and v is True:
                        event_components += [(k, None, 'flag', i)]
                    elif isinstance(v, (int,float)):
                        event_components += [(k, v, 'scalar', i)]
                    elif isinstance(v, list):
                        for f in v:
                            event_components += [(k, f, 'vector', i)]
                    else:
                        event_components += [(k, v, 'vector', i)]
            batch_indices += [i_batch]
            i += 1

    i = 0 
    for i_batch,game in enumerate(games):
        unknown = []
        for gobj in game['state']:
            if 'name' not in gobj and 'zone' not in gobj:
                unknown += [i]
            if 'zone' in gobj:
                object_components += [('zone', gobj['zone'], 'categorical', i)]
            if 'components' in gobj:
                for k,v in gobj['components'].items():
                    if k == 'colors':
                        onehot = [
                            int(c in v)
                            for c in COLOR_WHEEL
                        ]
                        object_components += [(k, onehot, 'color', i)]
                    elif isinstance(v, bool) and v is True:
                        object_components += [(k, None, 'flag', i)]
                    elif isinstance(v, (int,float)):
                        object_components += [(k, v, 'scalar', i)]
                    elif isinstance(v, list):
                        for f in v:
                            object_components += [(k, f, 'vector', i)]
                    else:
                        object_components += [(k, v, 'vector', i)]
            i += 1
            batch_indices += [i_batch]
            unknown_cards += [unknown]

    for i_batch,game in enumerate(games):
        for i_from, token in enumerate(game['history'] + game['state']):
            if 'edges' in token:
                for e in token['edges']:
                    edges += [(RELATIONS.index(e['rel']), i_from, indexof(e['to'], game['history'], game['state']), i_batch)]
        
    keys = set(k for k,_,_,_ in object_components + event_components)
    key_embeddings = dict(zip(keys, catalog.key_vectors(keys)))

    vectors = [v for (_,v,kd,_) in object_components + event_components if kd == 'vector']
    vector_embeddings = dict(zip(vectors, catalog.embed_strings(vectors)))

    return {
        'state': pack_components(object_components, key_embeddings, vector_embeddings, sum(len(g['state']) for g in games)),
        'history': pack_components(event_components, key_embeddings, vector_embeddings, sum(len(g['history']) for g in games)),
        'edges': torch.tensor(edges, dtype=torch.int32),
        'unknown_cards': torch.tensor(
            [(b,i) for b,g in enumerate(games) for i,c in enumerate(g['state']) if 'name' not in c and 'zone' in c],
            dtype=torch.int32,
        ),
        'batch_indices': torch.tensor(batch_indices),
        'batch_dim': (len(set(batch_indices)), max_L, TOKEN_DIM),
    }

def indexof(id: str, history, state):
    indices = [
        i for i,v in enumerate(history + state)
        if ('id' in v and v['id'] == id) or ('handles' in v and id in v['handles'])
    ]
    if len(indices) == 0:
        print(f'could not find object with id {id}')
    elif len(indices) == 1:
        return indices[0]
    else:
        raise Exception(f'Multiple tokens parsed with id {id}({len(indices)})')

def pack_components(components: list, key_embeddings: dict, vector_embeddings: dict, num_tokens: int):    
    scalars = torch.tensor([
        list(key_embeddings[k]) + [v, i]
        for (k,v,kd,i) in components
        if kd == 'scalar'
    ])

    vectors = torch.tensor([
        list(key_embeddings[k]) + list(vector_embeddings[v]) + [i]
        for (k,v,kd,i) in components
        if kd == 'vector'
    ])

    flags = torch.tensor([
        list(key_embeddings[k]) + [i]
        for (k,_,kd,i) in components
        if kd == 'flag' 
    ])

    colors = torch.tensor([
        list(key_embeddings[k]) + v + [i]
        for (k,v,kd,i) in components
        if kd == 'color' 
    ])

    categoricals = torch.tensor([
        list(key_embeddings[k]) + [VOCABS[VOCAB_LOOKUP[k]].index(v), list(VOCABS).index(VOCAB_LOOKUP[k]), i]
        for (k,v,kd,i) in components
        if kd == 'categorical'
    ])

    return {
        'scalars': scalars,
        'vectors': vectors,
        'colors': colors,
        'flags': flags,
        'categoricals': categoricals,
        'num_tokens': num_tokens,
    }