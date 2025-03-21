import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import IterableDataset
import dgl
import random
from dgl.dataloading import GraphDataLoader
from rdkit.Chem.BRICS import FindBRICSBonds, BreakBRICSBonds
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import MACCSkeys
from rdkit import RDConfig
from rdkit import RDLogger 
import pickle as pkl                                                                                                                                                              
RDLogger.DisableLog('rdApp.*')  
import os
from splitters import random_split
from torch.utils.data.distributed import DistributedSampler

import pickle
from tokenizer.pep2fragments import get_cut_bond_idx


fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)


vocab_dict = {}
# TODO change file path
with open('tokenizer/vocabs/Vocab_SIZE254.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip('\n')
        try:
            vocab_dict[line.split(':')[1]] = int(line.split(':')[0])
        except:
            # print(line)
            pass


def bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

    return fbond

def bond_mask_features():

    fbond = [
        0,  # bond is not None
        0,
        0,
        0,
        0,
        0,
        0
    ]
    fbond += onek_encoding_unk(6, list(range(6)))

    return fbond

def pharm_property_types_feats(mol,factory=factory): 
    types = [i.split('.')[1] for i in factory.GetFeatureDefs().keys()]
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)]
    result = [0] * len(types)
    for i in range(len(types)):
        if types[i] in list(set(feats)):
            result[i] = 1
    return result

def GetBricsBonds(mol):
    brics_bonds = list()
    brics_bonds_rules = list()
    bonds_tmp = FindBRICSBonds(mol)
    bonds = [b for b in bonds_tmp]
    for item in bonds:# item[0] is bond, item[1] is brics type
        brics_bonds.append([int(item[0][0]), int(item[0][1])])
        brics_bonds_rules.append([[int(item[0][0]), int(item[0][1])], GetBricsBondFeature([item[1][0], item[1][1]])])
        brics_bonds.append([int(item[0][1]), int(item[0][0])])
        brics_bonds_rules.append([[int(item[0][1]), int(item[0][0])], GetBricsBondFeature([item[1][1], item[1][0]])])

    result = []
    for bond in mol.GetBonds():
        beginatom = bond.GetBeginAtomIdx()
        endatom = bond.GetEndAtomIdx()
        if [beginatom, endatom] in brics_bonds:
            result.append([bond.GetIdx(), beginatom, endatom])
            
    return result, brics_bonds_rules

def GetBricsBondFeature(action):
    result = []
    start_action_bond = int(action[0]) if (action[0] !='7a' and action[0] !='7b') else 7
    end_action_bond = int(action[1]) if (action[1] !='7a' and action[1] !='7b') else 7
    emb_0 = [0 for i in range(17)]
    emb_1 = [0 for i in range(17)]
    emb_0[start_action_bond] = 1
    emb_1[end_action_bond] = 1
    result = emb_0 + emb_1
    return result

def maccskeys_emb(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1) # aviod index 0
    return mol

def GetFragmentFeats(mol):
    # break_bonds = [mol.GetBondBetweenAtoms(i[0][0],i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]

    # Instead of BRICS algorithm, we use fragmentation method tailored for peptide data
    break_bonds = get_cut_bond_idx(mol)

    if break_bonds == []:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol,break_bonds,addDummies=False)
    frags_idx_lst = Chem.GetMolFrags(tmp)
    frags_mol_lst = Chem.GetMolFrags(tmp, asMols=True)
    result_ap = {}
    result_p = {}
    result_frag = {}
    pharm_id = 0
    
    for frag_idx in frags_idx_lst:
        for atom_id in frag_idx:
            result_ap[atom_id] = pharm_id
        try:
            mol_pharm = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, frag_idx))
            emb_0 = maccskeys_emb(mol_pharm)
            emb_1 = pharm_property_types_feats(mol_pharm)
        except Exception:
            emb_0 = [0 for i in range(167)]
            emb_1 = [0 for i in range(27)]
            
        result_p[pharm_id] = emb_0 + emb_1
        result_frag[pharm_id] = Chem.MolToSmiles(frags_mol_lst[pharm_id], canonical=True)
        pharm_id += 1

    return result_ap, result_p, result_frag

ELEMENTS = [35, 6, 7, 8, 9, 15, 16, 17, 53]
ATOM_FEATURES = {
    'atomic_num': ELEMENTS,
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def atom_features(atom: Chem.rdchem.Atom):
    features = onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    return features

def atom_mask_features():
    features = onek_encoding_unk(10, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(10, ATOM_FEATURES['degree']) + \
           onek_encoding_unk(10, ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(10, ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(10, ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(10, ATOM_FEATURES['hybridization']) + \
           [0] + \
           [0.01]  # scaled to about the same range as other features
    return features


def atom_labels(atom):
    atom_feature = [allowable_features['possible_atomic_num_list'].
                    index(atom.GetAtomicNum())] + [allowable_features['possible_chirality_list'].
                    index(atom.GetChiralTag())]
    return atom_feature



def get_pharm_label(frag:str):
    if frag not in vocab_dict.keys():
        # print(f'Warning Unfound fragments {frag}')
        pass
    return vocab_dict.get(frag, len(vocab_dict))


class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

    def __call__(self, g, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """
        num_atoms = g.number_of_nodes('a')
        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)
        node_mask = torch.zeros(num_atoms, dtype=torch.bool) 
        node_mask[masked_atom_indices] = True
        g.nodes['a'].data['mask'] = node_mask
        # create mask node label by copying atom feature of mask atom
        # mask_node_labels_list = []
        # for atom_idx in masked_atom_indices:
            # print(data.x[atom_idx])
            # print(data.x[atom_idx].view(1, -1))
            # mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        # data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        # print(data.mask_node_label)
        # data.masked_atom_indices = torch.tensor(masked_atom_indices)
        
        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            g.nodes['a'].data['f'][node_mask==True] = torch.FloatTensor(atom_mask_features())
        
        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            # for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
            #     for atom_idx in masked_atom_indices:
            #         if atom_idx in set((u, v)) and \
            #             bond_idx not in connected_edge_indices:
            #             connected_edge_indices.append(bond_idx)
            for atom_idx in masked_atom_indices:
                edge1 = g.in_edges(atom_idx, form = 'eid', etype = 'b')
                for e in edge1:
                    if e not in connected_edge_indices:
                        connected_edge_indices.append(int(e.detach().cpu()))
                # edge2 = g.out_edges(atom_idx, form='eid', etype= 'b')
                # if edge1 not in connected_edge_indices:
                #     connected_edge_indices.append(edge1)
                # if edge2 not in connected_edge_indices:
                #     connected_edge_indices.append(edge2)
                    
            num_edges = g.number_of_edges('b')
            
            # create mask edge labels by copying bond features of the bonds connected to
            # the mask atoms
            edge_mask = torch.zeros(num_edges, dtype=torch.bool) 
            edge_mask[connected_edge_indices] = True
            g.edges['b'].data['mask'] = edge_mask
            # mask_edge_labels_list = []
            # for bond_idx in connected_edge_indices[::2]: # because the
                # edge ordering is such that two directions of a single
                # edge occur in pairs, so to get the unique undirected
                # edge indices, we take every 2nd edge index from list
                # mask_edge_labels_list.append(
                #     data.edge_attr[bond_idx].view(1, -1))

            # data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
            # modify the original bond features of the bonds connected to the mask atoms
            # for bond_idx in connected_edge_indices:
            #     data.edge_attr[bond_idx] = torch.tensor(
            #         [self.num_edge_type, 0])
            for bond_idx in connected_edge_indices:
                g.edges['b'].data['x'][bond_idx] =  torch.FloatTensor(bond_mask_features())
            # data.connected_edge_indices = torch.tensor(
            #     connected_edge_indices[::2])

        return g

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)


def Mol2HeteroGraph(mol, masked_atom_indices=None, num_atom_type=119,num_edge_type=5):
    
    # build graphs
    edge_types = [('a','b','a'),('p','r','p'),('a','j','p'), ('p','j','a')]
    edges = {k:[] for k in edge_types}
    # if mol.GetNumAtoms() == 1:
    #     g = dgl.heterograph(edges, num_nodes_dict={'a':1,'p':1})
    # else:

    # result_ap: Dict {atom_id:pharm_id}
    # result_p: Dict {pharm_id:fragments_feature}
    result_ap, result_p, result_frag = GetFragmentFeats(mol)
    reac_idx, bbr = GetBricsBonds(mol)

    for bond in mol.GetBonds(): 
        edges[('a','b','a')].append([bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])
        edges[('a','b','a')].append([bond.GetEndAtomIdx(),bond.GetBeginAtomIdx()])
        # print(bond.GetEndAtomIdx())

    for r in reac_idx:
        begin = r[1]
        end = r[2]
        edges[('p','r','p')].append([result_ap[begin],result_ap[end]])
        edges[('p','r','p')].append([result_ap[end],result_ap[begin]])

    for k,v in result_ap.items():
        edges[('a','j','p')].append([k,v])
        edges[('p','j','a')].append([v,k])

    g = dgl.heterograph(edges)
    f_atom = []
    src,dst = g.edges(etype=('a','b','a'))


    atom_label_list = []
    for idx in g.nodes('a'):
        atom = mol.GetAtomWithIdx(idx.item())
        atom_label_list.append(atom_labels(atom))
        f_atom.append(atom_features(atom))
    # print(f_atom)
    f_atom = torch.FloatTensor(f_atom)
    g.nodes['a'].data['f'] = f_atom
    atom_label_list = torch.LongTensor(atom_label_list)
    g.nodes['a'].data['label'] = atom_label_list
    dim_atom = len(f_atom[0])

    f_pharm = []
    pharm_label_list = []
    for k,v in result_p.items():
        frag = result_frag[k]
        pharm_label_list.append(get_pharm_label(frag))
        f_pharm.append(v)
            
    g.nodes['p'].data['f'] = torch.FloatTensor(f_pharm)
    dim_pharm = len(f_pharm[0])
    g.nodes['p'].data['label'] = pharm_label_list


    
    dim_atom_padding = g.nodes['a'].data['f'].size()[0]#节点数量
    dim_pharm_padding = g.nodes['p'].data['f'].size()[0]

    g.nodes['a'].data['f_junc'] = torch.cat([g.nodes['a'].data['f'], torch.zeros(dim_atom_padding, dim_pharm)], 1)
    g.nodes['p'].data['f_junc'] = torch.cat([torch.zeros(dim_pharm_padding, dim_atom), g.nodes['p'].data['f']], 1)
    

    f_bond = []
    src,dst = g.edges(etype=('a','b','a'))
    for i in range(g.num_edges(etype=('a','b','a'))):
        f_bond.append(bond_features(mol.GetBondBetweenAtoms(src[i].item(),dst[i].item())))
    g.edges[('a','b','a')].data['x'] = torch.FloatTensor(f_bond)

    f_reac = []
    src, dst = g.edges(etype=('p','r','p'))
    for idx in range(g.num_edges(etype=('p','r','p'))):
        p0_g = src[idx].item()
        p1_g = dst[idx].item()
        for i in bbr:
            p0 = result_ap[i[0][0]]
            p1 = result_ap[i[0][1]]
            if p0_g == p0 and p1_g == p1:
                f_reac.append(i[1])
    g.edges[('p','r','p')].data['x'] = torch.FloatTensor(f_reac)
    print('end\n\n\n\n\n')
    return g



class MolGraphSet(IterableDataset):
    # def __init__(self,df,target,log=print):
    def __init__(self,df,log=print, transform=None):
        # self.data = df.head(10)
        self.data = df
        self.mol_count = len(self.data)
        # self.graphs = []
        self.transform = transform
        self.log = log
    def __len__(self):
        return self.mol_count
    
    def get_data(self,data):
        for i,row in data.iterrows():
            # graphs = []
            smi = row['smiles']
            # label = row[target].values.astype(float)
            try:
                
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    self.log('invalid',smi)
                else:
                    g = Mol2HeteroGraph(mol)
                    # print(g)
                    if g.num_nodes('a') == 0:
                        self.log('no edge in graph',smi)
                    else:

                        if self.transform:
                            yield self.transform(g)
                        else:
                            
                            yield g
            except Exception as e:
                self.log(e,'invalid',smi)
                pass
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # 单进程运行，直接迭代所有数据
            return self.get_data(self.data)
        else:
            # 多进程运行，计算每个进程负责处理的数据
            per_worker = int(len(self.data) / worker_info.num_workers)
            start = worker_info.id * per_worker
            end = start + per_worker
            if worker_info.id == worker_info.num_workers - 1:
                end = len(self.data)
            return self.get_data(self.data.iloc[start:end])
        # return self.data_gen





def create_dataset(path,transform,mask_rate=0.1,mask_edge=False,shuffle=True):
    # transform_func = MaskAtom(num_atom_type=119, num_edge_type=5, mask_rate=mask_rate, mask_edge=mask_edge)
    dataset = MolGraphSet(pd.read_csv(path),transform=transform)
   
    # dataset = MolGraphSet(f,transform=transform)
    
    # dataset = MolGraphSet(pd.read_csv(args['path'], sep=',', compression='gzip',dtype='str'))
    return dataset



def make_loaders(ddp, dataset, world_size=0, global_rank=0, batch_size=512, num_workers=0, transform=None):
    # dataset = create_dataloader('/mnt/data/xiuyuting/delaney-processed.csv',transform=transform,shuffle=True)
    # train_dataset, valid_dataset, test_dataset = random_split(dataset, task_idx=None, null_value=0, frac_train=0.9,frac_valid=0.05, frac_test=0.05)
    # print(len(train_dataset),valid_dataset,test_dataset)
    train_dataset = create_dataset('/mnt/data/xiuyuting/'+dataset+'/train.csv',transform=transform)
    test_dataset = create_dataset('/mnt/data/xiuyuting/'+dataset+'/test.csv',transform=transform)
    valid_dataset = create_dataset('/mnt/data/xiuyuting/'+dataset+'/valid.csv',transform=transform)

    if ddp:
        train_smapler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)
        valid_smapler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=global_rank)
        test_smapler = DistributedSampler(test_dataset, num_replicas=world_size, rank=global_rank)
        train_loader = GraphDataLoader(train_dataset, batch_size=batch_size,  num_workers=num_workers, pin_memory=True, sampler=train_smapler)
        valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=valid_smapler)
        test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, num_workers = num_workers, pin_memory=True, sampler=test_smapler)

    else:
        train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, num_workers = num_workers)
        valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, num_workers = num_workers)
        test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, num_workers = num_workers)

    dataloaders = {'train': train_loader,
                'valid': valid_loader,
                'test': test_loader}

    return dataloaders

    
    
def random_split(load_path,save_dir,num_fold=5,sizes = [0.9,0.05,0.05],seed=0):
    df = pd.read_csv(load_path, header = None, names=['smiles'])
    # df = pd.read_csv(load_path, sep=',', compression='gzip',dtype='str')

    n = len(df)
    os.makedirs(save_dir,exist_ok=True)
    torch.manual_seed(seed)
    for fold in range(num_fold):

        df = df.loc[torch.randperm(n)].reset_index(drop=True)
        train_size = int(sizes[0] * n)
        train_val_size = int((sizes[0] + sizes[1]) * n)
        train = df[:train_size]
        val = df[train_size:train_val_size]
        test = df[train_val_size:]
        train.to_csv(os.path.join(save_dir)+f'train.csv',index=False)
        val.to_csv(os.path.join(save_dir)+f'valid.csv',index=False)
        test.to_csv(os.path.join(save_dir)+f'test.csv',index=False)

        # train.to_csv(os.path.join(save_dir)+f'{seed}_fold_{fold}_train.csv',index=False)
        # val.to_csv(os.path.join(save_dir)+f'{seed}_fold_{fold}_valid.csv',index=False)
        # test.to_csv(os.path.join(save_dir)+f'{seed}_fold_{fold}_test.csv',index=False)

import hydra
from omegaconf import DictConfig, OmegaConf
@hydra.main(config_path="configs", config_name="pretrain_masking.yaml", version_base='1.2')
def main(cfg: DictConfig):
    
    batch_size = 32
    shuffle = True
    # dataset = create_dataloader('/mnt/data/xiuyuting/delaney-processed.csv',shuffle=True)
    dataloaders = make_loaders(ddp=cfg.mode.ddp,
                            dataset='/mnt/data/xiuyuting/test/',
                            world_size=0,
                            global_rank=0,
                            batch_size=cfg.train.batch_size,
                            num_workers=cfg.train.num_workers,
                            transform=MaskAtom(num_atom_type=119, num_edge_type=5, mask_rate=cfg.train.mask_rate, mask_edge=cfg.train.mask_edge))    
    # print(cfg.train)
    

if __name__=='__main__':
    import sys
    import json
    path = '/mnt/data/xiuyuting/pep_atlas_uniparc_smiles_30_withnnaa.txt'
    random_split(path,'/mnt/data/xiuyuting/pep_atlas_uniparc_smiles_30_withnnaa/',1)
    # df = pd.read_csv('/mnt/data/xiuyuting/pep_atlas_uniparc_smiles_30_withnnaa/train.csv')
    # print(df.head(10))
    # print(vocab_dict)
    # smiles= 'CC(C(=O)NCCC(=O)NCC(C)C(=O)NC)N'
    # print(GetFragmentFeats(Chem.MolFromSmiles(smiles)))
    # path = '/mnt/data/public/seq_mpnn/dataset/pep_atlas_uniparc_smiles_30/raw/data.csv.gz'
    # random_split(path,'/mnt/data/xiuyuting/pep_atlas_uniparc_smiles_30/',num_fold=1)
    # main()
    # for i,batch in enumerate(train_loader):
    #     print(batch.nodes['a'].data['label'])
    #     if i==10: break
    # print(len(train_loader))
    # main()
