import torch
import numpy as np
from .constants import *
from ..dump import is_region
import time

class DataLoader():
    def __init__(
        self,
        catalog: any,
    ):
        self.catalog = catalog
        self.vector_lookup = {}

    def load_game(
        self,
        games: list[dict],
    ) -> dict[str, torch.Tensor]:
        # each game object component should become a [key, value, kind, idx]
        # vector (where idx corresponds to the parent object index)

        # card_names = set(
        #     [
        #         c['name']
        #         for g in games
        #         for c in g['state']
        #         if 'name' in c and c['type'] in ['game_object'] and c['name'] != 'HIDDEN'
        #     ]
        # )
        # card_metadata = self.catalog.lookup(card_names)

        object_components = []
        event_components = []
        action_components = []
        action_refs = []
        action_decode_indices = []
        edges = []
        action_edges = []
        encode_batch_indices = []
        decode_batch_indices = []
        unknown_cards_flat = []
        global_token_indices = []
        prompt_token_indices = []

        max_L = 0
        decode_max_L = 0

        i = 0
        for i_batch,game in enumerate(games):
            max_L = max(max_L, len(game['state']) + len(game['history']))
            decode_max_L = max(decode_max_L, len(game['actions']))
            for event in game['history']:
                token_type = event['type']
                if 'components' in event:
                    for k,v in event['components'].items():
                        if vocab_for(k, token_type):
                            event_components += [(k, v, 'categorical', i, token_type)]
                        elif isinstance(v, bool) and v is True:
                            event_components += [(k, None, 'flag', i, token_type)]
                        elif isinstance(v, (int,float)):
                            event_components += [(k, v, 'scalar', i, token_type)]
                        elif isinstance(v, list):
                            for f in v:
                                event_components += [(k, f, 'vector', i, token_type)]
                        else:
                            event_components += [(k, v, 'vector', i, token_type)]
                encode_batch_indices += [i_batch]
                i += 1

        i = 0 
        for i_batch,game in enumerate(games):
            for gobj in game['state']:
                token_type = gobj['type']
                if 'name' not in gobj and 'zone' in gobj:
                    unknown_cards_flat += [i]
                if 'zone' in gobj:
                    object_components += [('zone', gobj['zone'], 'categorical', i, token_type)]
                if 'components' in gobj:
                    for k,v in gobj['components'].items():
                        if k == 'colors':
                            onehot = [
                                int(c in v)
                                for c in COLOR_WHEEL
                            ]
                            object_components += [(k, onehot, 'color', i, token_type)]
                        elif vocab_for(k, token_type):
                            object_components += [(k, v, 'categorical', i, token_type)]
                        elif isinstance(v, bool) and v is True:
                            object_components += [(k, None, 'flag', i, token_type)]
                        elif isinstance(v, (int,float)):
                            object_components += [(k, v, 'scalar', i, token_type)]
                        elif isinstance(v, list):
                            for f in v:
                                object_components += [(k, f, 'vector', i, token_type)]
                        else:
                            object_components += [(k, v, 'vector', i, token_type)]
                i += 1
                encode_batch_indices += [i_batch]

        i = 0
        for i_batch,game in enumerate(games):
            for action in game['actions']:
                if 'type' in action:
                    token_type = action['type']
                    if 'components' in action:
                        for k,v in action['components'].items():
                            if vocab_for(k, token_type):
                                action_components += [(k, v, 'categorical', i, token_type)]
                            elif isinstance(v, bool) and v is True:
                                action_components += [(k, None, 'flag', i, token_type)]
                            elif isinstance(v, (int,float)):
                                action_components += [(k, v, 'scalar', i, token_type)]
                            elif isinstance(v, list):
                                for f in v:
                                    action_components += [(k, f, 'vector', i, token_type)]
                            else:
                                action_components += [(k, v, 'vector', i, token_type)]
                    decode_batch_indices += [i_batch]
                    i += 1

        for i_batch,game in enumerate(games):
            id_lookup = {}
            for i, token in enumerate(game['history'] + game['state']):
                if token['type'] == 'global':
                    global_token_indices += [i]
                if token['type'] == 'decision': # prompt token sent through policy trunk
                    prompt_token_indices += [i]
                if 'id' in token:
                    id_lookup[token['id']] = i
                if 'handles' in token:
                    for h in token['handles']:
                        id_lookup[h] = i
            for i_from, token in enumerate(game['history'] + game['state']):
                if 'edges' in token:
                    for e in token['edges']:
                        edges += [(RELATIONS.index(e['rel']), i_from, id_lookup[e['to']], i_batch)]
            for i_action,action in enumerate(game['actions']):
                if 'ref' in action:
                    action_refs += [[i_batch, id_lookup[action['ref']], i_action]]
                else:
                    action_decode_indices += [[i_batch, i_action]]
                if 'edges' in action:
                    for e in action['edges']:
                        action_edges += [(RELATIONS.index(e['rel']), i_action, id_lookup[e['to']], i_batch)]
            
        keys = set(k for k,_,_,_,_ in object_components + event_components + action_components)
        vectors = [v for (_,v,kd,_,_) in object_components + event_components + action_components if kd == 'vector']
        emb_lookup, offset = self.get_embedding_lookup(keys, vectors)

        if len(global_token_indices) != len(games):
            raise ValueError(f'found {len(global_token_indices)} global state tokens in {len(games)} games')
                        
        return {
            'state': self.pack_components(object_components, sum(len(g['state']) for g in games)),
            'history': self.pack_components(event_components, sum(len(g['history']) for g in games)),
            'actions': self.pack_components(action_components, sum(len([a for a in g['actions'] if 'type' in a]) for g in games)),
            'edges': torch.tensor(edges, dtype=torch.int32),
            'action_edges': torch.tensor(action_edges, dtype=torch.int32),
            'unknown_cards': torch.tensor(
                [
                    (b,i)
                    for b,g in enumerate(games)
                    for i,c in enumerate(g['state'])
                    if is_region(c)
                    for _ in range(
                        c['components']['count'] if 'count' in c['components'] else 1
                    )
                ],
                dtype=torch.int32,
            ),
            'unknown_cards_sampling': torch.tensor(
                [
                    (b,i)
                    for b,g in enumerate(games)
                    for i,c in enumerate(g['state'])
                    if is_region(c)
                ],
                dtype=torch.int32,
            ),
            'action_refs': torch.tensor(action_refs, dtype=torch.int32),
            'unknown_cards_flat': torch.tensor(unknown_cards_flat),
            'global_token_indices': torch.tensor(global_token_indices),
            'prompt_token_indices': torch.tensor(prompt_token_indices),
            'batch_indices': torch.tensor(encode_batch_indices),
            'decode_batch_indices': torch.tensor(decode_batch_indices),
            'action_decode_indices': torch.tensor(action_decode_indices, dtype=torch.int32),
            'batch_dim': (len(games), max_L, TOKEN_DIM),
            'decode_batch_dim': (len(games), decode_max_L, TOKEN_DIM),
            'emb_lookup': emb_lookup,
            'emb_lookup_offset': offset,
            'pca': self.catalog.pca(EXTERNAL_EMBEDDING_PCA_DIM),
        }

    def get_embedding_lookup(self, keys: list[str], strings: list[str], return_diff_only=True):
        offset = len(self.vector_lookup)
        keys_to_embed = [k for k in keys if k not in self.vector_lookup]
        key_embeddings = dict(zip(keys_to_embed, self.catalog.key_vectors(keys_to_embed))) if keys_to_embed else {}
        self.vector_lookup |= key_embeddings

        strings_to_embed = [s for s in strings if s not in self.vector_lookup]
        string_embeddings = dict(zip(strings_to_embed, self.catalog.embed_strings(strings_to_embed))) if strings_to_embed else {}
        self.vector_lookup |= string_embeddings

        id_list = list(self.vector_lookup)
        key_lookup_indices = torch.tensor([
            id_list.index(k) for k in keys
        ])
        string_lookup_indices = torch.tensor([
            id_list.index(s) for s in strings
        ])

        if return_diff_only:
            raw_lookup = list(key_embeddings.values()) + list(string_embeddings.values())
        else:
            raw_lookup = list(self.vector_lookup.values())
            offset = 0
                
        lookup_table = torch.tensor(np.stack(raw_lookup), dtype=torch.float32) if raw_lookup else torch.empty(0)

        return lookup_table, offset

    def pack_components(self, components: list, num_tokens: int):
        id_list = list(self.vector_lookup)

        scalars = torch.tensor([
            [id_list.index(k), v, i]
            for (k,v,kd,i,_) in components
            if kd == 'scalar'
        ],dtype=torch.int32)

        vectors = torch.tensor([
            [id_list.index(k), id_list.index(v), i]
            for (k,v,kd,i,_) in components
            if kd == 'vector'
        ],dtype=torch.int32)

        flags = torch.tensor([
            [id_list.index(k), i]
            for (k,_,kd,i,_) in components
            if kd == 'flag' 
        ],dtype=torch.int32)

        colors = torch.tensor([
            [id_list.index(k)] + v + [i]
            for (k,v,kd,i,_) in components
            if kd == 'color' 
        ],dtype=torch.int32)

        categoricals = torch.tensor([
            [id_list.index(k), vocab_index(vocab_for(k, tt), v), list(VOCABS).index(vocab_for(k, tt)), i]
            for (k,v,kd,i,tt) in components
            if kd == 'categorical'
        ],dtype=torch.int32)

        return {
            'scalars': scalars,
            'vectors': vectors,
            'colors': colors,
            'flags': flags,
            'categoricals': categoricals,
            'num_tokens': num_tokens,
        }