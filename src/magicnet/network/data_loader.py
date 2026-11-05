import torch
from .constants import *

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
    batches = [
        gather_components(g['state'], g['history'], catalog)
        for g in games
    ]
    object_components = [b[0] for b in batches]
    event_components = [b[1] for b in batches]
    edges = [b[2] for b in batches]

    keys = set(k for b in batches for k,_,_,_ in b[0] + b[1])
    key_embeddings = dict(zip(keys, catalog.key_vectors(keys)))

    vectors = [v for b in batches for (_,v,kd,_) in b[0] + b[1] if kd == 'vector']
    vector_embeddings = dict(zip(vectors, catalog.embed_strings(vectors)))

    return {
        'state': pack_components(object_components, key_embeddings, vector_embeddings),
        'history': pack_components(event_components, key_embeddings, vector_embeddings),
        'edges': torch.nn.utils.rnn.pad_sequence([
            torch.tensor(b, dtype=torch.int32) for b in edges
        ]).transpose(0,1),
        'unknown_cards': torch.nn.utils.rnn.pad_sequence([
            torch.tensor(
                [i for i,c in enumerate(g['state']) if 'name' not in c and 'zone' in c],
                dtype=torch.int32,
            )
            for g in games
        ]),
    }

def gather_components(state: list[dict], history: list[dict], catalog: any):
    object_components = []
    event_components = []
    edges = []
    for i,gobj in enumerate(state):
        if 'edges' in gobj:
            for e in gobj['edges']:
                edges += [(RELATIONS.index(e['rel']), i+len(history), indexof(e['to'], history, state))]
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

    for i,event in enumerate(history):
        if 'edges' in event:
            for e in event['edges']:
                edges += [(RELATIONS.index(e['rel']), i, indexof(e['to'], history, state))]
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
    return object_components, event_components, edges

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

def pack_components(batches: list[list], key_embeddings: dict, vector_embeddings: dict):
    pad = float('-inf')
    
    scalars = torch.nn.utils.rnn.pad_sequence([
        torch.tensor([
            list(key_embeddings[k]) + [v, i]
            for (k,v,kd,i) in components
            if kd == 'scalar'
        ] or [[pad]*(EXTERNAL_EMBEDDING_DIM + 2)])
        for components in batches
    ], padding_value=pad).transpose(0,1)

    vectors = torch.nn.utils.rnn.pad_sequence([
        torch.tensor([
            list(key_embeddings[k]) + list(vector_embeddings[v]) + [i]
            for (k,v,kd,i) in components
            if kd == 'vector'
        ] or [[pad]*(EXTERNAL_EMBEDDING_DIM*2 + 1)])
        for components in batches
    ], padding_value=pad).transpose(0,1)

    for components in batches:
        x = torch.tensor([
            list(key_embeddings[k]) + [i]
            for (k,_,kd,i) in components
            if kd == 'flag'
        ] or [[pad]*EXTERNAL_EMBEDDING_DIM])

    flags = torch.nn.utils.rnn.pad_sequence([
        torch.tensor([
            list(key_embeddings[k]) + [i]
            for (k,_,kd,i) in components
            if kd == 'flag'
        ] or [[pad]*(EXTERNAL_EMBEDDING_DIM + 1)])
        for components in batches
    ], padding_value=pad).transpose(0,1)

    colors = torch.nn.utils.rnn.pad_sequence([
        torch.tensor([
            list(key_embeddings[k]) + v + [i]
            for (k,v,kd,i) in components
            if kd == 'color' 
        ] or [[pad]*(EXTERNAL_EMBEDDING_DIM + len(COLOR_WHEEL))])
        for components in batches
    ], padding_value=pad).transpose(0,1)

    categoricals = torch.nn.utils.rnn.pad_sequence([
        torch.tensor([
            list(key_embeddings[k]) + [VOCABS[VOCAB_LOOKUP[k]].index(v), list(VOCABS).index(VOCAB_LOOKUP[k]), i]
            for (k,v,kd,i) in components
            if kd == 'categorical'
        ] or [[pad]*(EXTERNAL_EMBEDDING_DIM + 3)])
        for components in batches
    ], padding_value=pad).transpose(0,1)

    return {
        'scalars': scalars,
        'vectors': vectors,
        'colors': colors,
        'flags': flags,
        'categoricals': categoricals,
    }