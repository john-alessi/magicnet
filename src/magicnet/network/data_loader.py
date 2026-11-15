import torch
import re
import numpy as np
from .constants import *
from ..dump import is_region
import time

class DataLoader():
    def __init__(
        self,
        catalog: any,
        per_batch_lookups = False,
    ):
        self.catalog = catalog
        self.vector_lookup = {}
        self.card_cache = {}
        self.per_batch_lookups = per_batch_lookups
        vocab_sizes = np.array([len(v) for v in VOCABS.values()])
        offsets = np.cumsum(vocab_sizes) - vocab_sizes
        self.categorical_offsets = dict(zip(list(VOCABS.keys()), list(offsets)))

    def load_game(
        self,
        games: list[dict],
    ) -> dict[str, torch.Tensor]:
        card_names = set(
            [
                c['name']
                for g in games
                for c in g['state']
                if 'name' in c and c['type'] in ['game_object', 'library_card'] and c['name'] != 'HIDDEN'
            ]
        )
        card_metadata = self.catalog.lookup(card_names)

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
        mana_cost_component_indices = []

        max_L = 0
        decode_max_L = 0
        mana_max_L = 0
        mana_N = 0
        object_components_max_L = 0
        event_components_max_L = 0
        action_components_components_max_L = 0

        i = 0
        for i_batch,game in enumerate(games):
            max_L = max(max_L, len(game['state']) + len(game['history']))
            decode_max_L = max(decode_max_L, len(game['actions']))
            for event in game['history']:
                comp = []
                token_type = event['type']
                if 'components' in event:
                    for k,v in event['components'].items():
                        if vocab_for(k, token_type):
                            comp += [(k, v, 'categorical', i, token_type)]
                        elif isinstance(v, bool) and v is True:
                            comp += [(k, None, 'flag', i, token_type)]
                        elif isinstance(v, (int,float)):
                            comp += [(k, v, 'scalar', i, token_type)]
                        elif isinstance(v, list):
                            for f in v:
                                comp += [(k, f, 'vector', i, token_type)]
                        else:
                            comp += [(k, v, 'vector', i, token_type)]
                encode_batch_indices += [i_batch]
                event_components_max_L = max(event_components_max_L, len(comp))
                event_components += comp
                i += 1

        i = 0 
        for i_batch,game in enumerate(games):
            for gobj in game['state']:
                comp = []
                token_type = gobj['type']
                if is_region(gobj):
                    unknown_cards_flat += [i]
                if 'zone' in gobj:
                    comp += [('zone', gobj['zone'], 'categorical', i, token_type)]
                if 'name' in gobj and 'oracle_id' in gobj and card_metadata[gobj['name']] is not None:
                    comp += [('text', gobj['oracle_id'], 'vector', i, token_type)]
                    mana_cost = expand_mana_cost(card_metadata[gobj['name']].mana_cost, mana_N)
                    if mana_cost is not None:
                        comp += [('mana_cost', mana_cost , 'mana_cost', i, token_type)]
                        mana_max_L = max(mana_max_L, mana_cost.shape[0])
                        mana_N += 1

                if 'components' in gobj:
                    for k,v in gobj['components'].items():
                        if k == 'colors':
                            onehot = [
                                int(c in v)
                                for c in COLOR_WHEEL
                            ]
                            comp += [(k, onehot, 'color', i, token_type)]
                        elif vocab_for(k, token_type):
                            comp += [(k, v, 'categorical', i, token_type)]
                        elif isinstance(v, bool) and v is True:
                            comp += [(k, None, 'flag', i, token_type)]
                        elif isinstance(v, (int,float)):
                            comp += [(k, v, 'scalar', i, token_type)]
                        elif isinstance(v, list):
                            for f in v:
                                comp += [(k, f, 'vector', i, token_type)]
                        else:
                            comp += [(k, v, 'vector', i, token_type)]
                i += 1
                encode_batch_indices += [i_batch]
                object_components_max_L = max(object_components_max_L, len(comp))
                object_components += comp

        i = 0
        for i_batch,game in enumerate(games):
            for action in game['actions']:
                if 'type' in action:
                    comp = []
                    token_type = action['type']
                    if 'components' in action:
                        for k,v in action['components'].items():
                            if vocab_for(k, token_type):
                                comp += [(k, v, 'categorical', i, token_type)]
                            elif isinstance(v, bool) and v is True:
                                comp += [(k, None, 'flag', i, token_type)]
                            elif isinstance(v, (int,float)):
                                comp += [(k, v, 'scalar', i, token_type)]
                            elif isinstance(v, list):
                                for f in v:
                                    comp += [(k, f, 'vector', i, token_type)]
                            else:
                                comp += [(k, v, 'vector', i, token_type)]
                    decode_batch_indices += [i_batch]
                    i += 1
                    action_components_components_max_L = max(action_components_components_max_L, len(comp))
                    action_components += comp

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
        emb_lookup, offset, id_list = self.get_embedding_lookup(keys, vectors, card_metadata)

        if len(global_token_indices) != len(games):
            raise ValueError(f'found {len(global_token_indices)} global state tokens in {len(games)} games')

        mana_batch_dim = (mana_N, mana_max_L, MANA_SYMBOL_DIM-1)

        encode_batch_info = get_batch_info(len(games), max_L, torch.tensor(encode_batch_indices))
        decode_batch_info = get_batch_info(len(games), decode_max_L, torch.tensor(decode_batch_indices))
                        
        return {
            'state': self.pack_components(
                object_components,
                sum(len(g['state']) for g in games),
                object_components_max_L,
                id_list,
                mana_max_L
            ),
            'history': self.pack_components(
                event_components,
                sum(len(g['history']) for g in games),
                event_components_max_L,
                id_list
            ),
            'actions': self.pack_components(
                action_components,
                sum(len([a for a in g['actions'] if 'type' in a]) for g in games),
                action_components_components_max_L,
                id_list
            ),
            'edges': torch.tensor(edges, dtype=torch.int32, device='cpu'),
            'action_edges': torch.tensor(action_edges, dtype=torch.int32, device='cpu'),
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
                device='cpu',
            ),
            'unknown_cards_sampling': torch.tensor(
                [
                    (b,i)
                    for b,g in enumerate(games)
                    for i,c in enumerate(g['state'])
                    if is_region(c)
                ],
                dtype=torch.int32,
                device='cpu',
            ),
            'action_refs': torch.tensor(action_refs, dtype=torch.int32, device='cpu'),
            'unknown_cards_flat': torch.tensor(unknown_cards_flat, device='cpu'),
            'global_token_indices': torch.tensor(global_token_indices, device='cpu'),
            'prompt_token_indices': torch.tensor(prompt_token_indices, device='cpu'),
            'action_decode_indices': torch.tensor(action_decode_indices, dtype=torch.int32, device='cpu'),
            'encode_batch_info': encode_batch_info,
            'decode_batch_info': decode_batch_info,
            'emb_lookup': emb_lookup,
            'emb_lookup_offset': offset,
            'pca': self.catalog.pca(EXTERNAL_EMBEDDING_PCA_DIM),
        }

    def get_embedding_lookup(self, keys: list[str], strings: list[str], card_metadata: list):
        offset = 0 if self.per_batch_lookups else len(self.vector_lookup)
        lookup = {} if self.per_batch_lookups else self.vector_lookup
        keys_to_embed = keys if self.per_batch_lookups else [k for k in keys if k not in lookup]
        key_embeddings = dict(zip(keys_to_embed, self.catalog.key_vectors(keys_to_embed))) if keys_to_embed else {}
        lookup |= key_embeddings

        strings_to_embed = [s for s in strings if s not in lookup]
        string_embeddings = dict(zip(strings_to_embed, self.catalog.embed_strings(strings_to_embed))) if strings_to_embed else {}
        lookup |= string_embeddings

        card_embeddings = {
            c.oracle_id: c.vector
            for c in card_metadata.values()
            if c is not None
        } if self.per_batch_lookups else {
            c.oracle_id: c.vector
            for c in card_metadata.values()
            if c is not None
            if c.oracle_id not in lookup
        }
        lookup |= card_embeddings

        if not self.per_batch_lookups:
            self.vector_lookup |= lookup

        id_list = list(key_embeddings) + list(string_embeddings) + list(card_embeddings)
        raw_lookup = list(key_embeddings.values()) + list(string_embeddings.values()) + list(card_embeddings.values())   
        lookup_table = torch.tensor(np.stack(raw_lookup), dtype=torch.float32, device='cpu') if raw_lookup else torch.empty(0, device='cpu')
        return lookup_table, offset, id_list

    def pack_components(
        self,
        components: list,
        num_tokens: int,
        comp_max_L: int,
        id_list: list,
        mana_max_L=0
    ):
        scalars = torch.tensor([
            [id_list.index(k), v, i]
            for (k,v,kd,i,_) in components
            if kd == 'scalar'
        ],dtype=torch.int32, device='cpu')

        vectors = torch.tensor([
            [id_list.index(k), id_list.index(v), i]
            for (k,v,kd,i,_) in components
            if kd == 'vector'
        ],dtype=torch.int32, device='cpu')

        flags = torch.tensor([
            [id_list.index(k), i]
            for (k,_,kd,i,_) in components
            if kd == 'flag' 
        ],dtype=torch.int32, device='cpu')

        colors = torch.tensor([
            [id_list.index(k)] + v + [i]
            for (k,v,kd,i,_) in components
            if kd == 'color' 
        ],dtype=torch.int32, device='cpu')

        categoricals = torch.tensor([
            [id_list.index(k), self.get_categorical_embedding_index(k, v, tt), i]
            for (k,v,kd,i,tt) in components
            if kd == 'categorical'
        ],dtype=torch.int32, device='cpu')

        mc = [(i,v) for _,v,kd,i,_ in components if kd == 'mana_cost']
        num_mana_costs = len(mc)
        if num_mana_costs > 0:
            mana_cost_indices, mana_cost_matrices = map(list, zip(*mc))
            mana_cost_indices = torch.tensor(mana_cost_indices, dtype=torch.int32)
            mana_costs = torch.cat(mana_cost_matrices, dim=0)
            mana_cost_cost_indices = mana_costs[:,-1].to(torch.int32)
            mana_cost_batch_info = get_batch_info(num_mana_costs, mana_max_L, mana_cost_cost_indices, last_dim=COMPONENT_DIM)
            mana_costs = mana_costs[:,:-1]
        else:
            mana_costs = None
            mana_cost_batch_info = None
            mana_cost_indices = torch.empty(0, dtype=torch.int32)
        
        component_indices = torch.cat((
            scalars if scalars.numel() == 0 else scalars[:,-1],
            vectors if vectors.numel() == 0 else vectors[:,-1],
            flags if flags.numel() == 0 else flags[:,-1],
            colors if colors.numel() == 0 else colors[:,-1],
            categoricals if categoricals.numel() == 0 else categoricals[:,-1],
            mana_cost_indices,
        ))
        component_batch_info = get_batch_info(num_tokens, comp_max_L, component_indices, last_dim=COMPONENT_DIM)

        return {
            'scalars': scalars,
            'vectors': vectors,
            'colors': colors,
            'flags': flags,
            'categoricals': categoricals,
            'mana_costs': mana_costs,
            'num_tokens': num_tokens,
            'mana_cost_indices': mana_cost_indices if num_mana_costs > 0 else None,
            #'num_mana_costs': num_mana_costs,
            #'mana_max_L': mana_max_L,
            'components_max_L': comp_max_L,
            'mana_cost_batch_info': mana_cost_batch_info,
            'component_batch_info': component_batch_info,
        }

    def get_categorical_embedding_index(self, k, v, tt):
        vocab = vocab_for(k, tt)
        index = vocab_index(vocab, v)
        offset = self.categorical_offsets[vocab]
        return offset + index

def expand_mana_cost(mana_cost: str, i):
    symbol_idx = 'WUBRGSP'
    if len(mana_cost) > 0:
        symbols = re.findall(r"(?<=\{)(?:[WUBRGSP]|\d*)(?:\/(?:[WUBRGSP]|\d*))*(?=\})", mana_cost)
        result = torch.zeros(len(symbols), MANA_SYMBOL_DIM, dtype=torch.float32, device='cpu')
        result[:,MANA_SYMBOL_DIM-1] = i
        for i_symbol,s in enumerate(symbols):
            components = s.split('/')
            result[i_symbol,MANA_SYMBOL_DIM-2] = i_symbol
            for c in components:
                if c.isdigit():
                    result[i_symbol,MANA_SYMBOL_DIM-3] = int(c)
                else:
                    result[i_symbol, symbol_idx.index(c)] = 1
        return result
    else:
        return None

def get_batch_info(N: int, max_L: int, indices: torch.Tensor, last_dim=TOKEN_DIM):
    batch_dim = (N, max_L, last_dim)
    order = torch.argsort(indices, stable=True)
    counts = torch.bincount(indices, minlength=batch_dim[0])
    start_indices = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(len(indices)) - start_indices[indices[order]]
    mask = torch.ones(*batch_dim[:-1], dtype=torch.bool)
    mask[indices[order], positions] = False

    return {
        'dim': batch_dim,
        'indices': indices,
        'order': order,
        'counts': counts,
        'start_indices': start_indices,
        'positions': positions,
        'mask': mask,
    }