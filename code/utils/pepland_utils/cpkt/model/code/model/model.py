import dgl
import torch
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from functools import partial
import copy

import math
from model.util import get_func

# dgl graph utils
def reverse_edge(tensor):
    n = tensor.size(0)
    assert n%2 ==0
    delta = torch.ones(n).type(torch.long)
    delta[torch.arange(1,n,2)] = -1
    return tensor[delta+torch.tensor(range(n))]

def del_reverse_message(edge,field):
    """for g.apply_edges"""
    return {'m': edge.src[field]-edge.data['rev_h']}

def add_attn(node,field,attn):
        feat = node.data[field].unsqueeze(1)
        return {field: (attn(feat,node.mailbox['m'],node.mailbox['m'])+feat).squeeze(1)}

# nn modules

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    # p_attn = F.softmax(scores, dim = -1).masked_fill(mask, 0)  # 不影响
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Node_GRU(nn.Module):
    """GRU for graph readout. Implemented with dgl graph"""
    def __init__(self,hid_dim,bidirectional=True):
        super(Node_GRU,self).__init__()
        self.hid_dim = hid_dim
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.att_mix = MultiHeadedAttention(6,hid_dim)
        self.gru  = nn.GRU(hid_dim, hid_dim, batch_first=True, 
                           bidirectional=bidirectional)
    
    def split_batch(self, bg, ntype, field, device):
        hidden = bg.nodes[ntype].data[field]
        node_size = bg.batch_num_nodes(ntype)
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        max_num_node = max(node_size)
        # padding
        hidden_lst = []
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            assert size != 0, size
            cur_hidden = hidden.narrow(0, start, size)
            cur_hidden = torch.nn.ZeroPad2d((0,0,0,max_num_node-cur_hidden.shape[0]))(cur_hidden)
            hidden_lst.append(cur_hidden.unsqueeze(0))

        hidden_lst = torch.cat(hidden_lst, 0)

        return hidden_lst
        
    def forward(self,bg,suffix='h'):
        """
        bg: dgl.Graph (batch)
        hidden states of nodes are supposed to be in field 'h'.
        """
        self.suffix = suffix
        device = bg.device
        
        p_pharmj = self.split_batch(bg,'p',f'f_{suffix}',device)
        a_pharmj = self.split_batch(bg,'a',f'f_{suffix}',device)

        mask = (a_pharmj!=0).type(torch.float32).matmul((p_pharmj.transpose(-1,-2)!=0).type(torch.float32))==0
        h = self.att_mix(a_pharmj, p_pharmj, p_pharmj,mask) + a_pharmj

        hidden = h.max(1)[0].unsqueeze(0).repeat(self.direction,1,1)
        h, hidden = self.gru(h, hidden)
        
        # unpadding and reduce (mean) h: batch * L * hid_dim
        graph_embed = []
        node_size = bg.batch_num_nodes('p')
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            graph_embed.append(h[i, :size].view(-1, self.direction*self.hid_dim).mean(0).unsqueeze(0))
        graph_embed = torch.cat(graph_embed, 0)

        return graph_embed

def apply_custom_copy_src(g, etype, src_field, out_field, reduce_func):
    g.send_and_recv(
        g.edges(etype=etype),
        message_func=partial(copy_src, src_field=src_field, out_field=out_field),
        reduce_func=reduce_func,
        etype=etype
    )

def copy_src(edges, src_field, out_field):
    return {out_field: edges.src[src_field]}

 
class MVMP(nn.Module):
    def __init__(self,msg_func=add_attn,hid_dim=300,depth=3,view='aba',suffix='h',act=nn.ReLU()):
        """
        MultiViewMassagePassing
        view: a, ap, apj
        suffix: filed to save the nodes' hidden state in dgl.graph. 
                e.g. bg.nodes[ntype].data['f'+'_junc'(in ajp view)+suffix]
        """
        super(MVMP,self).__init__()
        self.view = view
        self.depth = depth
        self.suffix = suffix
        self.msg_func = msg_func
        self.act = act
        self.homo_etypes = [('a','b','a')]
        self.hetero_etypes = []
        self.node_types = ['a','p']
        if 'p' in view:
            self.homo_etypes.append(('p','r','p'))
        if 'j' in view:
            self.node_types.append('junc')
            self.hetero_etypes=[('a','j','p'),('p','j','a')] # don't have feature

        self.attn = nn.ModuleDict()
        for etype in self.homo_etypes + self.hetero_etypes:
            self.attn[''.join(etype)] = MultiHeadedAttention(4,hid_dim)

        self.mp_list = nn.ModuleDict()
        for edge_type in self.homo_etypes:
            self.mp_list[''.join(edge_type)] = nn.ModuleList([nn.Linear(hid_dim,hid_dim) for i in range(depth-1)])

        self.node_last_layer = nn.ModuleDict()
        for ntype in self.node_types:
            self.node_last_layer[ntype] = nn.Linear(3*hid_dim,hid_dim)

    def update_edge(self,edge,layer):
        return {'h':self.act(edge.data['x']+layer(edge.data['m']))}
    
    def update_node(self,node,field,layer):
        return {field:layer(torch.cat([node.mailbox['mail'].sum(dim=1),
                                       node.data[field],
                                       node.data['f']],1))}
    def init_node(self,node):
        return {f'f_{self.suffix}':node.data['f'].clone()}

    def init_edge(self,edge):
        return {'h':edge.data['x'].clone()}


    def forward(self,bg):
        suffix = self.suffix
        if  not bg.edges[('p','r','p')].data['x'].numel() and len(self.homo_etypes)>1:
            self.homo_etypes.remove(('p','r','p'))
        for ntype in self.node_types:
            if ntype != 'junc':
                bg.apply_nodes(self.init_node,ntype=ntype)
        for etype in self.homo_etypes:
            bg.apply_edges(self.init_edge,etype=etype)

        if 'j' in self.view:
            bg.nodes['a'].data[f'f_junc_{suffix}'] = bg.nodes['a'].data['f_junc'].clone()
            bg.nodes['p'].data[f'f_junc_{suffix}'] = bg.nodes['p'].data['f_junc'].clone()

        # 对每个边类型调用 apply_custom_copy_src

        
        update_funcs = {e:(fn.copy_e('h','m'), partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_{suffix}')) for e in self.homo_etypes }
        # update_funcs.update({e:(fn.copy_src(f'f_junc_{suffix}','mail'),partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_junc_{suffix}')) for e in self.hetero_etypes})
        # update_funcs.update({e:(fn.u_mul_e(f'f_junc_{suffix}',1.0,'m'),partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_junc_{suffix}')) for e in self.hetero_etypes})

        # message passing
        for i in range(self.depth-1):
            bg.multi_update_all(update_funcs,cross_reducer='sum')
            for e in self.hetero_etypes:
                apply_custom_copy_src(
                    bg,
                    e,
                    src_field=f'f_junc_{suffix}',
                    out_field='m',
                    reduce_func=partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_junc_{suffix}')
                )
            for edge_type in self.homo_etypes:
                bg.edges[edge_type].data['rev_h']=reverse_edge(bg.edges[edge_type].data['h'])
                bg.apply_edges(partial(del_reverse_message,field=f'f_{suffix}'),etype=edge_type)
                bg.apply_edges(partial(self.update_edge,layer=self.mp_list[''.join(edge_type)][i]), etype=edge_type)

        # last update of node feature
        update_funcs = {e:(fn.copy_e('h','mail'),partial(self.update_node,field=f'f_{suffix}',layer=self.node_last_layer[e[0]])) for e in self.homo_etypes}
        bg.multi_update_all(update_funcs,cross_reducer='sum')

        # last update of junc feature
        # bg.multi_update_all({e:(fn.copy_src(f'f_junc_{suffix}','mail'),
        #                          partial(self.update_node,field=f'f_junc_{suffix}',layer=self.node_last_layer['junc'])) for e in self.hetero_etypes},
        #                          cross_reducer='sum')
        # bg.multi_update_all({e:(fn.u_mul_e(f'f_junc_{suffix}',1.0,'mail'),
        #                     partial(self.update_node,field=f'f_junc_{suffix}',layer=self.node_last_layer['junc'])) for e in self.hetero_etypes},
        #                     cross_reducer='sum')
        # bg.multi_update_all({e:(partial(copy_src, src_field=f'f_junc_{suffix}', out_field='mail'),
        #                     partial(self.update_node,field=f'f_junc_{suffix}',layer=self.node_last_layer['junc'])) for e in self.hetero_etypes},
        #                     cross_reducer='sum')
        for e in self.hetero_etypes:
            apply_custom_copy_src(
                bg,
                e,
                src_field=f'f_junc_{suffix}',
                out_field='mail',
                reduce_func=partial(self.update_node, field=f'f_junc_{suffix}', layer=self.node_last_layer['junc'])
            )

class PharmHGT(nn.Module):
    def __init__(self, hid_dim, act, depth, atom_dim, bond_dim, pharm_dim, reac_dim, num_task=1):
        super(PharmHGT,self).__init__()
        # hid_dim = args['hid_dim']
        self.act = get_func(act)
        self.depth = depth
        self.output_dim = hid_dim
        # init
        # atom view
        self.w_atom = nn.Linear(atom_dim,hid_dim)
        self.w_bond = nn.Linear(bond_dim,hid_dim)
        # pharm view
        self.w_pharm = nn.Linear(pharm_dim,hid_dim)
        self.w_reac = nn.Linear(reac_dim,hid_dim)
        # junction view
        self.w_junc = nn.Linear(atom_dim + pharm_dim,hid_dim)

        ## define the view during massage passing
        # self.mp = MVMP(msg_func=add_attn,hid_dim=hid_dim,depth=self.depth,view='aj',suffix='h',act=self.act)
        self.mp = MVMP(msg_func=add_attn,hid_dim=hid_dim,depth=self.depth,view='apj',suffix='h',act=self.act)
        # self.mp_junc = MVMP(msg_func=add_attn,hid_dim=hid_dim,depth=self.depth,view='apj',suffix='j',act=self.act)
        
        ## readout
        self.readout = Node_GRU(hid_dim)
        self.readout_attn = Node_GRU(hid_dim)

        ## predict
        self.out = nn.Sequential(nn.Linear(4*hid_dim,hid_dim),
                                 self.act,
                                 nn.Linear(hid_dim,hid_dim),
                                 self.act,
                                 nn.Linear(hid_dim,num_task)
                                )

        self.initialize_weights()

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    def init_feature(self,bg):
        
        bg.nodes['a'].data['f'] = self.act(self.w_atom(bg.nodes['a'].data['f']))
        bg.edges[('a','b','a')].data['x'] = self.act(self.w_bond(bg.edges[('a','b','a')].data['x']))
        bg.nodes['p'].data['f'] = self.act(self.w_pharm(bg.nodes['p'].data['f']))
        if bg.edges[('p','r','p')].data['x'].numel():
            bg.edges[('p','r','p')].data['x'] = self.act(self.w_reac(bg.edges[('p','r','p')].data['x']))
        bg.nodes['a'].data['f_junc'] = self.act(self.w_junc(bg.nodes['a'].data['f_junc']))
        bg.nodes['p'].data['f_junc'] = self.act(self.w_junc(bg.nodes['p'].data['f_junc']))
        
    def forward(self,bg):
        """
        Args:
            bg: a batch of graphs
        """
        self.init_feature(bg)
        self.mp(bg)
        # self.mp_aug(bg)

        # embed_f = self.readout(bg,'h')
        # embed_aug = self.readout_attn(bg,'aug')
        # embed = torch.cat([embed_f,embed_aug],1)
        # out = self.out(embed)
        embed_f_a = bg.nodes['a'].data['f_h']
        embed_f_p = bg.nodes['p'].data['f_h']
        embed_junc_h_a = bg.nodes['a'].data['f_junc_h']
        embed_junc_h_p = bg.nodes['p'].data['f_junc_h']
        # embed_aug_a = bg.nodes['a'].data['f_aug']
        # embed_aug_p = bg.nodes['p'].data['f_aug']
        # embed_junc_aug_a = bg.nodes['a'].data['f_junc_aug']
        # embed_junc_aug_p = bg.nodes['p'].data['f_junc_aug']

        
        embed_a = torch.mean(torch.stack([embed_f_a,embed_junc_h_a],dim=1),dim=1)
        embed_p = torch.mean(torch.stack([embed_f_p,embed_junc_h_p],dim=1),dim=1)
        # embed_p = torch.mean(torch.stack([embed_f_p,embed_junc_h_p,embed_aug_p,embed_junc_aug_p],dim=1),dim=1)
        return embed_a,embed_f_p
        
