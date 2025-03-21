import os,re

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import  Draw
from rdkit.Chem import rdChemReactions
# from DrawSVG import DrawSVG
from IPython.display import SVG
IPythonConsole.ipython_useSVG = True
import pandas as pd
from rdkit import Chem
from rdkit.Chem.BRICS import FindBRICSBonds
import re


reaction_inter = rdChemReactions.ReactionFromSmarts(
    '[CX3:3](=[OX1])[NX3H1,NX3H0:4]>>[C:3](=O)[O].[N:4]')
reaction_intra = rdChemReactions.ReactionFromSmarts(
    '[CX3:3](=[OX1])[NX3H1,NX3H0:4]>>([C:3](=O)[O].[N:4])')
patt = Chem.MolFromSmarts("[CX3:3](=[OX1])[NX3H1,NX3H0:4]")

amino_acids = {'G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S', 'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T'}
aa2smiles = {k: Chem.MolToSmiles(Chem.MolFromSequence(k), canonical = True) for k in amino_acids}
smiles2aa = {Chem.MolToSmiles(Chem.MolFromSequence(k), canonical = True):k for k in amino_acids}


def neutralize_atoms(mol):
    """ Neutralize positive and negative atoms.
    """
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol

def sanitize(mol):
    """ Sanitize molecule.
    """
    
    # mol = Chem.MolFromSmiles(smi,sanitize=False)
    # mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol),sanitize=False)
    flag = 1
    valence_dict = {"C":4,"O":2,"N":3}
    while flag:
        try:
            Chem.SanitizeMol(mol)
            flag = 0
        except Exception as e:
            print(e)
            error_str = e.args[0]
            error_atom_idx = re.findall(r"\# (\d+)",error_str)
            if len(error_atom_idx) > 0:
                error_atom_idx = int(error_atom_idx[0])
                a = mol.GetAtomWithIdx(error_atom_idx)
                init_formal_charge = a.GetFormalCharge()
                if a.GetSymbol() == "C":
                    mol = None
                    break
                try:
                    correct_formal_charge = a.GetExplicitValence() - valence_dict[a.GetSymbol()] 
                except:
                    correct_formal_charge = init_formal_charge + 1
                a.SetFormalCharge(correct_formal_charge)
                print("Set #{} {} formal charge: {} -> {}".format(error_atom_idx,a.GetSymbol(),
                                                            init_formal_charge,correct_formal_charge))
            else:
                mol = None
                flag = 0
    return mol

def hydrolysis(mol, reaction_inter, reaction_intra, patt):
    """ Hydrolysis. 
    """
    
    # rxn = rdChemReactions.ReactionFromSmarts(
    # '[NX3H2&!R,NX3H1&R:1][CX4H:2][CX3:3](=[OX1])[N:4]>>[N:1][C:2][C:3](=O)O.[N:4]'
    # "[CX3;!R:3](=[OX1])[NX3H1;!R:4]>>[C:3](=O)[99#0].[N:4][88#0]"
    # )
    
    residues = []
    reacts = (mol,)
    while len(reacts) > 0:
        products = reaction_inter.RunReactants(reacts)
        if len(products) > 0:
            p1 = products[0][0]
            p2 = products[0][1]
            Chem.SanitizeMol(p1)
            Chem.SanitizeMol(p2)
            total_heavy_atom_num = reacts[0].GetNumHeavyAtoms()
            if p1.GetNumHeavyAtoms() + p2.GetNumHeavyAtoms() != total_heavy_atom_num+1:
                products_ = reaction_intra.RunReactants(reacts)
                Chem.SanitizeMol(products_[0][0])
                reacts = (products_[0][0],)
            else:
                if p1.HasSubstructMatch(patt):
                    residues_forward = hydrolysis(p1, reaction_inter, reaction_intra, patt)
                    residues = residues_forward + residues
                else:
                    residues.append(Chem.MolToSmiles(p1))
                reacts = (p2,)
        else:
            residues.append(Chem.MolToSmiles(reacts[0]))
            reacts = ()
    return residues

def cut_peptide(mol,patt=None):
    """ Cut peptide bond.
        args:
            mol: rdkit mol object
            patt: rdkit mol object
        return:
            frags_idxs: list of list of atom indexes
    """
    
    if type(patt) != Chem.rdchem.Mol:
        patt = Chem.MolFromSmarts("[CX3:3](=[OX1])[NX3H1,NX3H0,NX4H2,NX4H1:4]")
        
    ## Get bonds to cut
    bond_idxs = []
    matched_idxs = mol.GetSubstructMatches(patt)
    # print(len(matched_idxs))
    
    if len(matched_idxs) == 0:
        print("No substructures matched")
        return ()
    for sub_idx in matched_idxs:
        bond = mol.GetBondBetweenAtoms(sub_idx[0],sub_idx[2])
        bond_idxs.append(bond.GetIdx())

    ## Cut molecule to fragments
    frags = Chem.FragmentOnBonds(mol,bond_idxs,addDummies=True)
    frags_idxs = Chem.GetMolFrags(frags)
    frags_mols = Chem.GetMolFrags(frags,asMols=True)

    ## Get fragment tuples of each cutted bond
    frag_idx_tuples = []
    for matched_idx in matched_idxs:
        pre_frag = None
        back_frag = None
        for frag_idx in frags_idxs:
            if (matched_idx[0] in frag_idx) and (matched_idx[1] in frag_idx):
                pre_frag = frag_idx
            elif matched_idx[2] in frag_idx:
                back_frag = frag_idx
            else:
                pass
        frag_idx_tuples.append([pre_frag,back_frag])
        
    ## Record the count of fragments and correct pairwise order
    counts = []
    start_count = 0
    end_count = 0
    #     print(len(frags_idxs))
    for frag_idx in frags_idxs:
        pre_count = 0
        back_count = 0
        for i,tup in enumerate(frag_idx_tuples):
            if tup[0] == frag_idx:
                pre_count += 1
            elif tup[1] == frag_idx:
                back_count += 1
            else:
                pass
        count_tup = [pre_count,back_count]
        if (count_tup == [1,0]):
            if (start_count==0):
                start_count += 1
            else:
                frag_idx_tuples[i] = [frag_idx_tuples[i][1], frag_idx_tuples[i][0]]
                count_tup = [back_count,pre_count]
                end_count += 1
        elif (count_tup == [0,1]):
            if (end_count==0):
                end_count += 1
            else:
                frag_idx_tuples[i] = [frag_idx_tuples[i][1], frag_idx_tuples[i][0]]
                count_tup = [back_count,pre_count]
                start_count += 1
        elif (sum(count_tup) % 2 == 0) and (count_tup[0] != count_tup[1]): 
            frag_idx_tuples[i] = [frag_idx_tuples[i][1], frag_idx_tuples[i][0]]
            count_tup = [sum(count_tup)//2]*2
        else:
            pass
        counts.append(count_tup)
        
    ## Get the sorted fragments
    sorted_list = []
    try:
        start_id = counts.index([1,0])
        start_frag_idx = frags_idxs[start_id]
        for tup in frag_idx_tuples:
            tup[0] == start_frag_idx
            sorted_list.extend(tup)
            frag_idx_tuples.remove(tup)
            break
    except:
        sorted_list.extend(frag_idx_tuples[0])
        frag_idx_tuples.remove(frag_idx_tuples[0])

        
    init_len = len(frag_idx_tuples)+1
    new_len = len(frag_idx_tuples)
    while (len(frag_idx_tuples) > 0) and (init_len - new_len > 0):
        init_len -= 1
        for tup in frag_idx_tuples:
            if tup[0] == sorted_list[-1]:
                sorted_list.append(tup[1])
                frag_idx_tuples.remove(tup)
                new_len -= 1 
            elif tup[1] == sorted_list[0]:
                sorted_list.insert(0,tup[0])
                frag_idx_tuples.remove(tup)
                new_len -= 1
            
    if len(sorted_list) == len(frags_idxs):
        sorted_frags = [frags_mols[frags_idxs.index(t)] for t in sorted_list]
        return sorted_frags
    else:
        # print("Length of sorted list didn't match the length of frag list", len(sorted_list),len(frags_idxs))
        return frags_mols

def AddTail(frag,reaction):
    reacts = (frag,)
    while len(reacts) > 0:
        products = reaction.RunReactants(reacts)
        if len(products) > 0:
            p1 = products[0][0]
            Chem.SanitizeMol(p1)
            reacts = (p1,)
        else:
            frag = reacts[0]
            reacts = ()
    return frag


def to_fragment(smi):
    """ args:
            smi: a string of smiles
        return:
            a list of fragments, each of which is a tuple of (smiles, type). Like: [('C1CC1', 'A'), ('C1CC1', 'non-natural')]
    """

    res = []
    
    patt = Chem.MolFromSmarts("[CX3:3](=[OX1])[NX3H1,NX3H0,NX4H2,NX4H1:4]")
    smi_list = smi.split(".")
    smi_list.sort(key=lambda s: len(re.findall('[a-zA-Z]', s)))
    smi = smi_list[-1]
    mol = Chem.MolFromSmiles(smi,sanitize=False)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol),sanitize=False)
    mol = sanitize(mol)

    matched_idxs = mol.GetSubstructMatches(patt)

    sorted_frags = cut_peptide(mol)
    reaction_c = rdChemReactions.ReactionFromSmarts("[C:1](=O)[#0]>>[C:1](=O)O")
    reaction_n = rdChemReactions.ReactionFromSmarts("[N:2][#0]>>[N:2]")
    residues = [AddTail(AddTail(frag,reaction_c),reaction_n) for frag in sorted_frags]
    residues = [neutralize_atoms(mol) for mol in residues]

    for residue in residues:
        s_ = Chem.MolToSmiles(residue, canonical = True)
        if s_ in smiles2aa:
            res.append((s_, smiles2aa[s_]))
        else:
            res.append((s_, "non-natural"))
            
    return res


def cut_side_chain_of_aa(mol, core = "[NX3H2][CX4H]C(=O)O"):
    """ args: 
            mol: a rdkit mol object
            core: a SMARTS string of the core of the amino acid
        return:
            a tuple of (mol, matched), where mol is the rdkit mol object of the side chain, and matched is a boolean value indicating whether the core structure is matched
    """

    matched = True
    
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(core))
    
    if not matches:
        
        matched = False 
        return mol, matched

    # 删除匹配的原子，保留侧链
    rm_atoms = []
    for match in matches:
        rm_atoms.extend(match)  # 仅删除匹配的前两个原子，即氮原子和碳原子，保留侧链

    # 创建一个原子索引列表
    all_atoms = set(range(mol.GetNumAtoms()))

    # 计算要保留的原子
    keep_atoms = all_atoms - set(rm_atoms)

    # 生成侧链分子
    side_chain = Chem.EditableMol(Chem.Mol())
    atom_map = {}
    for atom in keep_atoms:
        new_idx = side_chain.AddAtom(mol.GetAtomWithIdx(atom))
        atom_map[atom] = new_idx

    # 添加键
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if start in keep_atoms and end in keep_atoms:
            side_chain.AddBond(atom_map[start], atom_map[end], bond.GetBondType())

    # 获得最终的侧链分子
    side_chain_mol = side_chain.GetMol()

    # # 输出侧链的SMILES表示
    # print(Chem.MolToSmiles(side_chain_mol))
    return side_chain_mol, matched


def brics_molecule(mol):
    """
    Fragment a molecule using the BRICS algorithm.
    Args:
        smiles (str): A SMILES string representing the molecule.
    Returns:
        list of str: A list of SMILES strings representing the fragments.
    """
    # Convert SMILES to RDKit molecule object
    if isinstance(object, str):
        mol = Chem.MolFromSmiles(mol)

    try:
        Chem.SanitizeMol(mol)
    except ValueError as e:
        print(f"Error sanitizing molecule: {e}")
        return []
        
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0],i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
    if break_bonds == []:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol,break_bonds,addDummies=False)
    frags_idx_lst = Chem.GetMolFrags(tmp)
    
    all_fragments = []
    for frag_idx in frags_idx_lst:
        mol_pharm = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, frag_idx))
        if mol_pharm:
            all_fragments.append(mol_pharm)

    return break_bonds, all_fragments, [Chem.MolToSmiles(x, canonical = True) for x in all_fragments]


def map_atom_indices(fragment, mol):
    return [mol.GetAtomWithIdx(int(atom.GetProp('orig_idx'))).GetIdx() for atom in fragment.GetAtoms()]


def is_carbon_carbon_single_bond(bond):

    return (bond.GetBeginAtom().GetAtomicNum() == 6 and \
    bond.GetEndAtom().GetAtomicNum() == 6 and \
    bond.GetBondType() == Chem.rdchem.BondType.SINGLE)


def is_carbon_nitrogen_single_bond(mol, bond):
    if bond.GetBeginAtom().GetAtomicNum() == 6:  # 碳原子
        for neighbor in bond.GetBeginAtom().GetNeighbors():
            if neighbor.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(bond.GetBeginAtom().GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE:
                return False 

        return True
    elif bond.GetEndAtom().GetAtomicNum() == 6:
        for neighbor in bond.GetEndAtom().GetNeighbors():
            if neighbor.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(bond.GetEndAtom().GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE:
                return False 
        return True

    return False


def get_cut_bond_idx(mol, side_chain_cut = True):
    cut_bond_set = []
    cut_bond_atom_set = []
    smart_mol = "C(=O)N"

    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smart_mol))
    for amino_bond in matches:
        assert len(amino_bond) == 3, 'amino bond should contain only three atoms'
        for atom_idx in amino_bond:
            atom = mol.GetAtomWithIdx(atom_idx)
            # 切割碳碳单键
            if atom.GetAtomicNum() == 6:
                bonds = atom.GetBonds()
                for bond in bonds:
                    if is_carbon_carbon_single_bond(bond):
                        cut_bond_set.append(bond.GetIdx())
                        cut_bond_atom_set.append([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
            # 切割碳氮单键，注意这里不能切割酰胺键
            if atom.GetAtomicNum() == 7:
                bonds = atom.GetBonds()
                for bond in bonds:
                    if is_carbon_nitrogen_single_bond(mol, bond):
                        cut_bond_set.append(bond.GetIdx())
                        cut_bond_atom_set.append([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
    
    if not cut_bond_set:
        print('nothing to cut, not a peptide', Chem.MolToSmiles(mol))
        return [],[]
    
    if side_chain_cut:
        for atom in mol.GetAtoms():
            atom.SetProp('orig_idx', str(atom.GetIdx()))
        #在沿着酰胺切的基础上对单个分子切

        mol_fragments = Chem.GetMolFrags(Chem.FragmentOnBonds(mol, cut_bond_set, addDummies=False), asMols=True)
        
        for frag in mol_fragments:
            break_bonds, all_fragments, fragments_smiles = brics_molecule(frag)
            # 将片段中的原子索引映射回原始分子中的索引
            atom_indices_map = map_atom_indices(frag, mol)
            for bond_idx in break_bonds:
                bond = frag.GetBondWithIdx(bond_idx)
                begin_atom_idx = atom_indices_map[bond.GetBeginAtomIdx()]
                end_atom_idx = atom_indices_map[bond.GetEndAtomIdx()]
                original_bond = mol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
                original_bond_idx = original_bond.GetIdx()
                
                cut_bond_set.append(original_bond_idx)
                cut_bond_atom_set.append([begin_atom_idx, end_atom_idx])

            

    return cut_bond_set, cut_bond_atom_set




def break_peptide_by_amino_bind(smiles, side_chain_cut = True):
    """
    Fragment a molecule using the BRICS algorithm.
    Args:
        smiles (str): A SMILES string representing the molecule.
    Returns:
        list of str: A list of smiles representing the fragments.
    """
    # Convert SMILES to RDKit molecule object
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
    except Exception as e:
        print(f"Error sanitizing molecule: {e}")
        return ['ERROR', smiles, e]
    

    cut_bond_set = get_cut_bond_idx(mol)
    if cut_bond_set:
        mol_fragments = Chem.GetMolFrags(Chem.FragmentOnBonds(mol, cut_bond_set, addDummies=False), asMols=True)
    else:
        return [Chem.MolToSmiles(mol)]

    if side_chain_cut:
        after_brics_fragments = []
        for frag in mol_fragments:
            after_brics_fragments.extend(brics_molecule(frag))
        return after_brics_fragments
    else:
        return [Chem.MolToSmiles(x, canonical = True) for x in mol_fragments]


def get_cut_bond_idx_by_breaking_ammino_bond(mol, side_chain_cut = True):
    cut_bond_set = []
    cut_bond_atom_set = []
    smart_mol = "C(=O)N"

    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smart_mol))
    for amino_bond in matches:
        assert len(amino_bond) == 3, 'amino bond should contain only three atoms'
        for atom_idx in amino_bond:
            atom = mol.GetAtomWithIdx(atom_idx)
            # 切割酰胺键
            if atom.GetAtomicNum() == 6:
                bonds = atom.GetBonds()
                for bond in bonds:
                    if (bond.GetBeginAtom().GetAtomicNum() == 6 and bond.GetEndAtom().GetAtomicNum() == 7 and bond.GetEndAtom().GetTotalNumHs() != 2)  or (bond.GetBeginAtom().GetAtomicNum() == 7 and bond.GetEndAtom().GetAtomicNum() == 6 and bond.GetBeginAtom().GetTotalNumHs() != 2):
                        cut_bond_set.append(bond.GetIdx())
                        cut_bond_atom_set.append([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
    return cut_bond_set, cut_bond_atom_set


# For Atom Pretraining, mask whole AA's atom    
def get_atom_parentAA(mol):   

    for atom in mol.GetAtoms():
        atom.SetProp('orig_idx', str(atom.GetIdx()))

    cut_bond_set, cut_bond_atom_set = get_cut_bond_idx(mol, side_chain_cut=False)
    if cut_bond_set:
        mol_fragments = Chem.GetMolFrags(Chem.FragmentOnBonds(mol, cut_bond_set, addDummies=False), asMols=True)
    else:
        aa_label = {}
        atom_indices_map = map_atom_indices(mol, mol)
        for atom_idx in atom_indices_map:
            aa_label[atom_idx] = 1
        return aa_label
    aa_idx = 0
    aa_label = {}
    for frag in mol_fragments:
            # 将片段中的原子索引映射回原始分子中的索引
            atom_indices_map = map_atom_indices(frag, mol)
            for atom_idx in atom_indices_map:
                aa_label[atom_idx] = aa_idx
            aa_idx += 1
    return aa_label



if __name__ == '__main__':
    smiles = 'NC(N)=NCc1cccc(CC=O)c1'
    #print(smiles, Chem.MolFromSmiles(smiles))
    mol = Chem.MolFromSmiles(smiles)
    break_bonds, break_bonds_atoms = get_cut_bond_idx(mol)
    print(break_bonds)

    tmp = Chem.FragmentOnBonds(mol, break_bonds,addDummies=False)
    frags_mol_lst = Chem.GetMolFrags(tmp, asMols=True)

    print([Chem.MolToSmiles(mol, canonical=True) for mol in frags_mol_lst ])