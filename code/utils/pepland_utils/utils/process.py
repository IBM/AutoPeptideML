import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from rdkit import Chem
import torch
import dgl
from rdkit.Chem.BRICS import FindBRICSBonds, BreakBRICSBonds
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import MACCSkeys
from rdkit import RDConfig
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
from ..tokenizer.pep2fragments import get_cut_bond_idx, get_atom_parentAA

fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
vocab_dict = {}

with open(os.path.join(root_dir, 'tokenizer/vocabs/Vocab_SIZE258.txt'),
          'r') as f:
    idx = 0
    for line in f.readlines():
        line = line.strip('\n')
        try:
            vocab_dict[line] = idx
            idx += 1
        except:
            # print(line)
            pass

# print(f'vocab dict size {len(vocab_dict)}')

ELEMENTS = [35, 6, 7, 8, 9, 15, 16, 17, 53]
ATOM_FEATURES = {
    'atomic_num':
    ELEMENTS,
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list':
    list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


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


def atom_labels(atom):
    atom_feature = [
        allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())
    ] + [
        allowable_features['possible_chirality_list'].index(
            atom.GetChiralTag())
    ]
    return atom_feature


def get_pharm_label(frag: str):
    if frag not in vocab_dict.keys():
        print(f'Warning Unfound fragments {frag}')
        pass
    return vocab_dict.get(frag, len(vocab_dict))


def GetFragmentFeats(mol, side_chain_cut=True):
    # break_bonds = [mol.GetBondBetweenAtoms(i[0][0],i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]

    # Instead of BRICS algorithm, we use fragmentation method tailored for peptide data
    break_bonds, break_bonds_atoms = get_cut_bond_idx(mol)
    #mol_fragments = Chem.GetMolFrags(Chem.FragmentOnBonds(mol, cut_bond_set, addDummies=False), asMols=True)

    if break_bonds == []:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol, break_bonds, addDummies=False)

    # if side_chain_cut:
    #     after_brics_fragments = []
    #     for frag in mol_fragments:
    #         after_brics_fragments.extend(brics_molecule(frag))

    #((1,2,3),(4,5,6))
    frags_idx_lst = Chem.GetMolFrags(tmp)
    # ('CO','CCCC')
    frags_mol_lst = Chem.GetMolFrags(tmp, asMols=True)

    result_ap = {}
    result_p = {}
    result_frag = {}
    pharm_id = 0

    for frag_idx in frags_idx_lst:
        for atom_id in frag_idx:
            result_ap[atom_id] = pharm_id
        try:
            mol_pharm = Chem.MolFromSmiles(
                Chem.MolFragmentToSmiles(mol, frag_idx))
            emb_0 = maccskeys_emb(mol_pharm)
            emb_0 = emb_0 + [0]
            emb_1 = pharm_property_types_feats(mol_pharm)
            emb_1 = emb_1 + [0]
        except Exception:
            emb_0 = [0 for i in range(168)]
            emb_1 = [0 for i in range(28)]

        result_p[pharm_id] = emb_0 + emb_1
        result_frag[pharm_id] = Chem.MolToSmiles(frags_mol_lst[pharm_id],
                                                 canonical=True)
        pharm_id += 1

    #生成fragments之间的的edge feature
    brics_bonds = list()
    brics_bonds_rules = list()
    for item in break_bonds_atoms:  # item[0] is bond, item[1] is brics type
        brics_bonds.append([int(item[0]), int(item[1])])
        brics_bonds_rules.append(
            [[int(item[0]), int(item[1])],
             bond_features(mol.GetBondBetweenAtoms(int(item[0]),
                                                   int(item[1])))])
        brics_bonds.append([int(item[1]), int(item[0])])
        brics_bonds_rules.append(
            [[int(item[1]), int(item[0])],
             bond_features(mol.GetBondBetweenAtoms(int(item[0]),
                                                   int(item[1])))])

    result = []
    for bond in mol.GetBonds():
        beginatom = bond.GetBeginAtomIdx()
        endatom = bond.GetEndAtomIdx()
        if [beginatom, endatom] in brics_bonds:
            result.append([bond.GetIdx(), beginatom, endatom])

    return result_ap, result_p, result_frag, result, brics_bonds_rules


def pharm_property_types_feats(mol, factory=factory):
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
    for item in bonds:  # item[0] is bond, item[1] is brics type
        brics_bonds.append([int(item[0][0]), int(item[0][1])])
        brics_bonds_rules.append([[int(item[0][0]),
                                   int(item[0][1])],
                                  GetBricsBondFeature([item[1][0],
                                                       item[1][1]])])
        brics_bonds.append([int(item[0][1]), int(item[0][0])])
        brics_bonds_rules.append([[int(item[0][1]),
                                   int(item[0][0])],
                                  GetBricsBondFeature([item[1][1],
                                                       item[1][0]])])

    result = []
    for bond in mol.GetBonds():
        beginatom = bond.GetBeginAtomIdx()
        endatom = bond.GetEndAtomIdx()
        if [beginatom, endatom] in brics_bonds:
            result.append([bond.GetIdx(), beginatom, endatom])

    return result, brics_bonds_rules


def GetBricsBondFeature(action):
    result = []
    start_action_bond = int(action[0]) if (action[0] != '7a'
                                           and action[0] != '7b') else 7
    end_action_bond = int(action[1]) if (action[1] != '7a'
                                         and action[1] != '7b') else 7
    emb_0 = [0 for i in range(17)]
    emb_1 = [0 for i in range(17)]
    emb_0[start_action_bond] = 1
    emb_1[end_action_bond] = 1
    result = emb_0 + emb_1
    return result


def maccskeys_emb(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))


def Mol2HeteroGraph(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol == None:
        print(smi)
    # build graphs
    edge_types = [('a', 'b', 'a'), ('p', 'r', 'p'), ('a', 'j', 'p'),
                  ('p', 'j', 'a')]
    edges = {k: [] for k in edge_types}

    result_ap, result_p, result_frag, reac_idx, bbr = GetFragmentFeats(mol)
    atom2aa_label_map = get_atom_parentAA(mol)
    for bond in mol.GetBonds():
        edges[('a', 'b',
               'a')].append([bond.GetBeginAtomIdx(),
                             bond.GetEndAtomIdx()])
        edges[('a', 'b',
               'a')].append([bond.GetEndAtomIdx(),
                             bond.GetBeginAtomIdx()])

    for r in reac_idx:
        begin = r[1]
        end = r[2]
        edges[('p', 'r', 'p')].append([result_ap[begin], result_ap[end]])
        edges[('p', 'r', 'p')].append([result_ap[end], result_ap[begin]])

    for k, v in result_ap.items():
        edges[('a', 'j', 'p')].append([k, v])
        edges[('p', 'j', 'a')].append([v, k])

    g = dgl.heterograph(edges)
    f_atom = []
    src, dst = g.edges(etype=('a', 'b', 'a'))

    atom_label_list = []
    atom_aa_label_list = []
    for idx in g.nodes('a'):
        atom = mol.GetAtomWithIdx(idx.item())
        atom_label_list.append(atom_labels(atom))
        atom_aa_label_list.append(atom2aa_label_map[idx.item()])
        f_atom.append(atom_features(atom))

    f_atom = torch.FloatTensor(f_atom)
    g.nodes['a'].data['f'] = f_atom
    dim_atom = len(f_atom[0])

    f_pharm = []
    pharm_label_list = []
    for k, v in result_p.items():
        frag = result_frag[k]
        pharm_label_list.append(get_pharm_label(frag))
        f_pharm.append(v)

    g.nodes['p'].data['f'] = torch.FloatTensor(f_pharm)
    dim_pharm = len(f_pharm[0])

    dim_atom_padding = g.nodes['a'].data['f'].size()[0]  #节点数量
    dim_pharm_padding = g.nodes['p'].data['f'].size()[0]

    g.nodes['a'].data['f_junc'] = torch.cat(
        [g.nodes['a'].data['f'],
         torch.zeros(dim_atom_padding, dim_pharm)], 1)
    g.nodes['p'].data['f_junc'] = torch.cat(
        [torch.zeros(dim_pharm_padding, dim_atom), g.nodes['p'].data['f']], 1)

    f_bond = []
    src, dst = g.edges(etype=('a', 'b', 'a'))
    for i in range(g.num_edges(etype=('a', 'b', 'a'))):
        f_bond.append(
            bond_features(mol.GetBondBetweenAtoms(src[i].item(),
                                                  dst[i].item())))
    g.edges[('a', 'b', 'a')].data['x'] = torch.FloatTensor(f_bond)

    f_reac = []

    src, dst = g.edges(etype=('p', 'r', 'p'))

    for idx in range(g.num_edges(etype=('p', 'r', 'p'))):
        p0_g = src[idx].item()
        p1_g = dst[idx].item()

        for i in bbr:
            p0 = result_ap[i[0][0]]
            p1 = result_ap[i[0][1]]
            if p0_g == p0 and p1_g == p1:
                f_reac.append(i[1])
                break

    g.edges[('p', 'r', 'p')].data['x'] = torch.FloatTensor(f_reac)

    return g
