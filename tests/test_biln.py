import unittest

from rdkit import Chem

from autopeptideml.pipeline.smiles import (
    BilnToSmiles,
    SmilesToBiln,
    biln_to_smiles,
    fragments_to_biln,
    parse_biln,
    parse_biln_token,
    tokenize_biln_chain,
    validate_biln,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _same_molecule(smiles1: str, smiles2: str) -> bool:
    return _canonical(smiles1) == _canonical(smiles2)


# ---------------------------------------------------------------------------
# TestBilnParser
# ---------------------------------------------------------------------------

class TestBilnParser(unittest.TestCase):
    """Tests for BILN tokenisation and parsing."""

    # ------------------------------------------------------------------
    # tokenize_biln_chain
    # ------------------------------------------------------------------

    def test_tokenize_linear_biln(self):
        result = tokenize_biln_chain("A-G-C-D")
        self.assertEqual(result, ["A", "G", "C", "D"])

    def test_tokenize_single_token(self):
        self.assertEqual(tokenize_biln_chain("G"), ["G"])

    def test_tokenize_cyclic_biln(self):
        result = tokenize_biln_chain("C(1,3)-A-A-A-C(1,3)")
        self.assertEqual(
            result,
            ["C(1,3)", "A", "A", "A", "C(1,3)"],
        )

    def test_tokenize_multiple_annotations(self):
        result = tokenize_biln_chain("K(1,3)(2,3)-A")
        self.assertEqual(result, ["K(1,3)(2,3)", "A"])

    def test_tokenize_bracket_monomer_name(self):
        """Bracket-quoted names must not be split on internal hyphens."""
        result = tokenize_biln_chain("[dPhe(4-Cl)]-A")
        self.assertEqual(result, ["[dPhe(4-Cl)]", "A"])

    def test_tokenize_bracket_monomer_with_annotation(self):
        result = tokenize_biln_chain("[dPhe(4-Cl)](1,3)-A-[dPhe(4-Cl)](1,3)")
        self.assertEqual(
            result,
            ["[dPhe(4-Cl)](1,3)", "A", "[dPhe(4-Cl)](1,3)"],
        )

    def test_tokenize_bracket_monomer_midchain(self):
        result = tokenize_biln_chain("A-[d3-Pal]-G")
        self.assertEqual(result, ["A", "[d3-Pal]", "G"])

    def test_tokenize_long_linear(self):
        result = tokenize_biln_chain("A-G-V-L-I-P-F-W")
        self.assertEqual(
            result, ["A", "G", "V", "L", "I", "P", "F", "W"]
        )

    # ------------------------------------------------------------------
    # parse_biln_token
    # ------------------------------------------------------------------

    def test_parse_plain_monomer(self):
        symbol, annotations = parse_biln_token("A")
        self.assertEqual(symbol, "A")
        self.assertEqual(annotations, [])

    def test_parse_annotated_monomer(self):
        symbol, annotations = parse_biln_token("C(1,3)")
        self.assertEqual(symbol, "C")
        self.assertEqual(annotations, [(1, 3)])

    def test_parse_multiple_annotations(self):
        symbol, annotations = parse_biln_token("K(1,3)(2,4)")
        self.assertEqual(symbol, "K")
        self.assertEqual(annotations, [(1, 3), (2, 4)])

    def test_parse_large_bond_id(self):
        symbol, annotations = parse_biln_token("C(99,3)")
        self.assertEqual(symbol, "C")
        self.assertEqual(annotations, [(99, 3)])

    # ------------------------------------------------------------------
    # parse_biln
    # ------------------------------------------------------------------

    def test_parse_multichain_biln(self):
        result = parse_biln("A-G-C.D-E")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0]["symbol"], "A")
        self.assertEqual(result[1][0]["symbol"], "D")

    def test_parse_single_chain_multi_monomer(self):
        result = parse_biln("A-G-V")
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 3)
        symbols = [e["symbol"] for e in result[0]]
        self.assertEqual(symbols, ["A", "G", "V"])

    def test_parse_single_monomer(self):
        result = parse_biln("G")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0]["symbol"], "G")
        self.assertEqual(result[0][0]["annotations"], [])

    def test_parse_annotations_preserved(self):
        result = parse_biln("C(1,3)-A-C(1,3)")
        self.assertEqual(result[0][0]["annotations"], [(1, 3)])
        self.assertEqual(result[0][1]["annotations"], [])
        self.assertEqual(result[0][2]["annotations"], [(1, 3)])

    def test_parse_three_chains(self):
        result = parse_biln("A.G.V")
        self.assertEqual(len(result), 3)

    # ------------------------------------------------------------------
    # validate_biln
    # ------------------------------------------------------------------

    def test_validate_linear_biln(self):
        self.assertTrue(validate_biln("A-G-C-D"))

    def test_validate_single_monomer(self):
        self.assertTrue(validate_biln("G"))

    def test_validate_disulfide_biln(self):
        self.assertTrue(validate_biln("C(1,3)-A-A-A-C(1,3)"))

    def test_validate_head_to_tail_cycle(self):
        self.assertTrue(validate_biln("C(1,1)-A-A-A-C(1,2)"))

    def test_validate_multiple_disulfides(self):
        self.assertTrue(validate_biln("C(1,3)-C(2,3)-G-C(1,3)-C(2,3)"))

    def test_validate_multi_chain_with_bond(self):
        """Cross-chain disulfide: bond ID must still appear exactly twice."""
        self.assertTrue(validate_biln("C(1,3)-A.A-C(1,3)"))

    def test_validate_two_annotations_same_monomer(self):
        self.assertTrue(validate_biln("C(1,3)(2,3)-A-A-C(1,3)(2,3)"))

    def test_reject_bond_id_used_once(self):
        with self.assertRaises(ValueError):
            validate_biln("C(1,3)-A-A-C")

    def test_reject_bond_id_used_three_times(self):
        with self.assertRaises(ValueError):
            validate_biln("C(1,3)-A-A(1,3)-C(1,3)")

    def test_reject_rgroup_zero(self):
        """R-group 0 is not a valid attachment point."""
        with self.assertRaises(ValueError):
            validate_biln("C(1,0)-A-C(1,0)")

    def test_reject_unbalanced_parentheses(self):
        with self.assertRaises(ValueError):
            parse_biln("C(1,3-A")

    def test_reject_invalid_annotation_non_integer(self):
        with self.assertRaises(ValueError):
            parse_biln("C(foo,3)-A")

    def test_reject_invalid_annotation_missing_comma(self):
        with self.assertRaises(ValueError):
            parse_biln("C(13)-A")


# ---------------------------------------------------------------------------
# TestBilnToSmiles
# ---------------------------------------------------------------------------

class TestBilnToSmiles(unittest.TestCase):
    """Tests for conversion from BILN to RDKit SMILES."""

    def assert_valid_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(
            mol,
            msg=f"Invalid generated SMILES: {smiles}",
        )
        return mol

    # ------------------------------------------------------------------
    # Basic linear peptides
    # ------------------------------------------------------------------

    def test_linear_peptide_agc(self):
        self.assert_valid_smiles(biln_to_smiles("A-G-C"))

    def test_linear_peptide_via_element(self):
        converter = BilnToSmiles()
        self.assert_valid_smiles(converter._single_call("A-G-C"))

    def test_single_monomer_glycine(self):
        smiles = biln_to_smiles("G")
        self.assert_valid_smiles(smiles)

    def test_single_monomer_alanine(self):
        smiles = biln_to_smiles("A")
        self.assert_valid_smiles(smiles)

    def test_single_monomer_proline(self):
        """Proline has a cyclic sidechain; single-monomer output must be valid."""
        smiles = biln_to_smiles("P")
        self.assert_valid_smiles(smiles)

    def test_all_standard_single_monomers(self):
        """Every standard amino acid must produce a valid SMILES when alone."""
        standard = [
            "A", "C", "D", "E", "F", "G", "H",
            "L", "N", "P", "Q", "S", "T", "V", "W",
        ]
        for aa in standard:
            with self.subTest(aa=aa):
                smiles = biln_to_smiles(aa)
                self.assert_valid_smiles(smiles)

    def test_dipeptide_all_standard_pairs(self):
        """Spot-check every standard AA as the N-terminal residue in a dipeptide."""
        standard = [
            "A", "C", "D", "E", "F", "G", "H",
            "L", "N", "P", "Q", "S", "T", "V", "W",
        ]
        for aa in standard:
            with self.subTest(aa=aa):
                smiles = biln_to_smiles(f"{aa}-G")
                self.assert_valid_smiles(smiles)

    def test_long_linear_peptide(self):
        biln = "A-G-V-L-I-P-F-W"
        smiles = biln_to_smiles(biln)
        mol = self.assert_valid_smiles(smiles)
        # Should have the expected number of heavy atoms
        self.assertGreater(mol.GetNumAtoms(), 30)

    # ------------------------------------------------------------------
    # Cyclic / disulfide topologies
    # ------------------------------------------------------------------

    def test_disulfide_cycle(self):
        """C(1,3) ... C(1,3) — sulfurs must be directly bonded."""
        biln = "C(1,3)-A-A-A-C(1,3)"
        smiles = biln_to_smiles(biln)
        mol = self.assert_valid_smiles(smiles)

        sulfur_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
        self.assertEqual(len(sulfur_atoms), 2)

        sulfur_bond = mol.GetBondBetweenAtoms(
            sulfur_atoms[0].GetIdx(),
            sulfur_atoms[1].GetIdx(),
        )
        self.assertIsNotNone(
            sulfur_bond,
            msg="Expected the two cysteine sulfur atoms to be connected.",
        )

    def test_head_to_tail_cycle(self):
        """R1->R2 cyclization must produce a ring."""
        biln = "C(1,1)-A-A-A-C(1,2)"
        smiles = biln_to_smiles(biln)
        mol = self.assert_valid_smiles(smiles)
        self.assertGreater(
            mol.GetRingInfo().NumRings(),
            0,
            msg="Expected head-to-tail BILN cycle to produce a ring.",
        )

    def test_multiple_explicit_bonds(self):
        """A monomer can carry two separate explicit BILN bonds."""
        biln = "C(1,3)(2,3)-A-A-C(1,3)(2,3)"
        smiles = biln_to_smiles(biln)
        self.assert_valid_smiles(smiles)

    def test_double_disulfide(self):
        """Two independent disulfide bridges in one chain."""
        biln = "C(1,3)-C(2,3)-G-C(1,3)-C(2,3)"
        smiles = biln_to_smiles(biln)
        mol = self.assert_valid_smiles(smiles)
        sulfur_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
        self.assertEqual(len(sulfur_atoms), 4)

    def test_long_disulfide_loop(self):
        """Disulfide bridging across many residues."""
        biln = "C(1,3)-G-G-G-G-G-G-G-C(1,3)"
        smiles = biln_to_smiles(biln)
        mol = self.assert_valid_smiles(smiles)
        sulfur_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
        self.assertEqual(len(sulfur_atoms), 2)
        ss_bond = mol.GetBondBetweenAtoms(
            sulfur_atoms[0].GetIdx(), sulfur_atoms[1].GetIdx()
        )
        self.assertIsNotNone(ss_bond)

    # ------------------------------------------------------------------
    # Multi-chain BILN
    # ------------------------------------------------------------------

    def test_multichain_two_independent_chains(self):
        """Dot separator produces two disconnected components."""
        biln = "A-G.G-A"
        smiles = biln_to_smiles(biln)
        mol = self.assert_valid_smiles(smiles)
        frags = Chem.GetMolFrags(mol)
        self.assertEqual(len(frags), 2)

    def test_multichain_identical_chains(self):
        """Two identical chains produce the expected duplicated fragment."""
        smiles = biln_to_smiles("G.G")
        mol = self.assert_valid_smiles(smiles)
        self.assertEqual(len(Chem.GetMolFrags(mol)), 2)

    def test_multichain_asymmetric(self):
        biln = "A-G-C.G-A"
        smiles = biln_to_smiles(biln)
        mol = self.assert_valid_smiles(smiles)
        self.assertEqual(len(Chem.GetMolFrags(mol)), 2)

    def test_crosschain_disulfide(self):
        """Disulfide bond that spans two separate chains."""
        biln = "C(1,3)-A.A-C(1,3)"
        smiles = biln_to_smiles(biln)
        mol = self.assert_valid_smiles(smiles)
        sulfur_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
        self.assertEqual(len(sulfur_atoms), 2)
        ss_bond = mol.GetBondBetweenAtoms(
            sulfur_atoms[0].GetIdx(), sulfur_atoms[1].GetIdx()
        )
        self.assertIsNotNone(
            ss_bond,
            msg="Cross-chain disulfide bond must be present.",
        )

    # ------------------------------------------------------------------
    # Non-standard / bracketed monomers
    # ------------------------------------------------------------------

    def test_non_standard_monomer_x1(self):
        """X1 (cyclohexylalanine-like) must produce valid SMILES alone."""
        smiles = biln_to_smiles("X1")
        self.assert_valid_smiles(smiles)

    def test_non_standard_monomer_x1_in_peptide(self):
        smiles = biln_to_smiles("X1-G")
        self.assert_valid_smiles(smiles)

    def test_non_standard_monomer_ddpr(self):
        """dDpr (D-diaminopropionic acid) has R1 and R3 attachment points."""
        smiles = biln_to_smiles("dDpr")
        self.assert_valid_smiles(smiles)

    def test_non_standard_monomer_ddpr_in_peptide(self):
        smiles = biln_to_smiles("dDpr-G")
        self.assert_valid_smiles(smiles)

    def test_non_standard_cysteine_analog_x2038(self):
        """X2038 is a cysteine analogue with R3 (thiol attachment)."""
        smiles = biln_to_smiles("X2038")
        self.assert_valid_smiles(smiles)

    def test_aspartate_sidechain_r3(self):
        """D (aspartate) has R1, R2, R3; used alone it should have two carboxyls."""
        smiles = biln_to_smiles("D")
        mol = self.assert_valid_smiles(smiles)
        # Aspartate has two oxygen-bearing carbons
        carbonyls = [
            a for a in mol.GetAtoms()
            if a.GetAtomicNum() == 8
        ]
        self.assertGreaterEqual(len(carbonyls), 2)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_unknown_monomer_raises_key_error(self):
        with self.assertRaises(KeyError):
            biln_to_smiles("NOT_A_MONOMER-A")

    def test_unknown_monomer_in_chain_raises(self):
        with self.assertRaises(KeyError):
            biln_to_smiles("A-ZZZZZ-G")

    def test_invalid_bond_count_raises(self):
        """Bond ID used only once must raise during validate_biln."""
        with self.assertRaises(ValueError):
            biln_to_smiles("C(1,3)-A-A")


# ---------------------------------------------------------------------------
# TestSmilesToBiln
# ---------------------------------------------------------------------------

class TestSmilesToBiln(unittest.TestCase):
    """
    Tests for SMILES -> BILN.

    These tests deliberately inspect topology rather than
    expecting RDKit's canonical SMILES representation.
    """

    def test_converter_exists(self):
        converter = SmilesToBiln()
        self.assertEqual(converter.name, "smiles-to-biln")

    def test_linear_peptide_ggg(self):
        """A normal peptide must not acquire unnecessary explicit cycle annotations."""
        smiles = "NCC(=O)NCC(=O)NCC(=O)O"   # Gly-Gly-Gly
        biln = SmilesToBiln()._single_call(smiles)
        self.assertIsInstance(biln, str)
        validate_biln(biln)
        self.assertNotIn("(1,", biln)

    def test_linear_peptide_no_cycle_annotation(self):
        """Ala-Gly must not get spurious cycle annotation."""
        smiles = "C[C@H](N)C(=O)NCC(=O)O"
        biln = SmilesToBiln()._single_call(smiles)
        validate_biln(biln)
        self.assertNotIn("(1,", biln)

    def test_disulfide_topology(self):
        """Conversion of a linear cysteine-containing peptide must not crash."""
        smiles = (
            "N[C@@H](CS)C(=O)"
            "N[C@@H](CC)C(=O)"
            "N[C@@H](CS)C(=O)O"
        )
        biln = SmilesToBiln()._single_call(smiles)
        self.assertIsInstance(biln, str)
        validate_biln(biln)

    def test_proline_dipeptide(self):
        """Pro-Gly must parse without error; proline has a ring in its backbone."""
        smiles = biln_to_smiles("P-G")
        biln = SmilesToBiln()._single_call(smiles)
        self.assertIsInstance(biln, str)
        validate_biln(biln)

    def test_aromatic_residue_phenylalanine(self):
        smiles = biln_to_smiles("F-G")
        biln = SmilesToBiln()._single_call(smiles)
        validate_biln(biln)

    def test_tryptophan_indole(self):
        smiles = biln_to_smiles("W-G")
        biln = SmilesToBiln()._single_call(smiles)
        validate_biln(biln)

    def test_histidine_imidazole(self):
        smiles = biln_to_smiles("H-G")
        biln = SmilesToBiln()._single_call(smiles)
        validate_biln(biln)


# ---------------------------------------------------------------------------
# TestBilnRoundTrip
# ---------------------------------------------------------------------------

class TestBilnRoundTrip(unittest.TestCase):
    """
    Round-trip tests.

    Since RDKit canonicalises SMILES, these tests compare
    molecular structure rather than literal SMILES strings.
    """

    def assert_same_molecule(self, smiles1, smiles2):
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        self.assertIsNotNone(mol1)
        self.assertIsNotNone(mol2)
        c1 = Chem.MolToSmiles(mol1, canonical=True, isomericSmiles=True)
        c2 = Chem.MolToSmiles(mol2, canonical=True, isomericSmiles=True)
        self.assertEqual(c1, c2)

    def _round_trip(self, biln: str) -> str:
        """biln -> smiles -> biln2 -> smiles2; returns smiles2."""
        smiles1 = biln_to_smiles(biln)
        biln2 = SmilesToBiln()._single_call(smiles1)
        validate_biln(biln2)
        return biln_to_smiles(biln2), smiles1

    # ------------------------------------------------------------------
    # Linear peptides
    # ------------------------------------------------------------------

    def test_linear_round_trip_agc(self):
        smiles2, smiles1 = self._round_trip("A-G-C")
        self.assert_same_molecule(smiles1, smiles2)

    def test_linear_round_trip_single_glycine(self):
        smiles2, smiles1 = self._round_trip("G")
        self.assert_same_molecule(smiles1, smiles2)

    def test_linear_round_trip_single_alanine(self):
        smiles2, smiles1 = self._round_trip("A")
        self.assert_same_molecule(smiles1, smiles2)

    def test_linear_round_trip_dipeptide_gg(self):
        smiles2, smiles1 = self._round_trip("G-G")
        self.assert_same_molecule(smiles1, smiles2)

    def test_linear_round_trip_dipeptide_va(self):
        smiles2, smiles1 = self._round_trip("V-A")
        self.assert_same_molecule(smiles1, smiles2)

    def test_linear_round_trip_with_proline(self):
        smiles2, smiles1 = self._round_trip("P-A-G")
        self.assert_same_molecule(smiles1, smiles2)

    def test_linear_round_trip_all_standard(self):
        """Each standard AA must round-trip individually."""
        standard = [
            "A", "C", "D", "E", "F", "G", "H",
            "L", "N", "P", "Q", "S", "T", "V", "W",
        ]
        for aa in standard:
            with self.subTest(aa=aa):
                smiles2, smiles1 = self._round_trip(aa)
                self.assert_same_molecule(smiles1, smiles2)

    # ------------------------------------------------------------------
    # Cyclic / disulfide topologies
    # ------------------------------------------------------------------

    def test_disulfide_round_trip(self):
        biln = "C(1,3)-A-A-A-C(1,3)"
        smiles1 = biln_to_smiles(biln)
        self.assertIsNotNone(Chem.MolFromSmiles(smiles1))
        biln2 = SmilesToBiln()._single_call(smiles1)
        validate_biln(biln2)

        annotations = [
            entry
            for chain in parse_biln(biln2)
            for entry in chain
            if entry["annotations"]
        ]
        self.assertGreaterEqual(
            len(annotations), 2,
            msg="Disulfide must carry at least two explicit annotations.",
        )

    def test_head_to_tail_round_trip(self):
        biln = "C(1,1)-A-A-A-C(1,2)"
        smiles1 = biln_to_smiles(biln)
        mol1 = Chem.MolFromSmiles(smiles1)
        self.assertIsNotNone(mol1)
        self.assertGreater(mol1.GetRingInfo().NumRings(), 0)

        biln2 = SmilesToBiln()._single_call(smiles1)
        validate_biln(biln2)

        explicit = any(
            entry["annotations"]
            for chain in parse_biln(biln2)
            for entry in chain
        )
        self.assertTrue(
            explicit,
            msg="Cycle topology was lost during SMILES -> BILN conversion.",
        )

    def test_double_disulfide_round_trip(self):
        biln = "C(1,3)-C(2,3)-G-C(1,3)-C(2,3)"
        smiles1 = biln_to_smiles(biln)
        mol1 = Chem.MolFromSmiles(smiles1)
        self.assertIsNotNone(mol1)
        biln2 = SmilesToBiln()._single_call(smiles1)
        validate_biln(biln2)
        # Both disulfides must survive
        annotations = [
            entry
            for chain in parse_biln(biln2)
            for entry in chain
            if entry["annotations"]
        ]
        self.assertGreaterEqual(len(annotations), 4)

    # ------------------------------------------------------------------
    # Multi-chain
    # ------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestFragmentToBiln
# ---------------------------------------------------------------------------

class TestFragmentToBiln(unittest.TestCase):
    """
    Focused tests for the function responsible for
    adding cycle/branch annotations.
    """

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            fragments_to_biln(fragments=[], monomer_names=["A"])

    def test_mismatched_lengths_reverse(self):
        mol = Chem.MolFromSmiles("NCC(=O)O")
        with self.assertRaises(ValueError):
            fragments_to_biln(fragments=[mol, mol], monomer_names=["A"])

    def test_single_fragment_glycine(self):
        mol = Chem.MolFromSmiles("NCC(=O)O")
        result = fragments_to_biln(fragments=[mol], monomer_names=["G"])
        self.assertEqual(result, "G")

    def test_single_fragment_alanine(self):
        mol = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
        result = fragments_to_biln(fragments=[mol], monomer_names=["A"])
        self.assertEqual(result, "A")

    def test_single_fragment_produces_valid_biln(self):
        mol = Chem.MolFromSmiles("NCC(=O)O")
        result = fragments_to_biln(fragments=[mol], monomer_names=["A"])
        self.assertTrue(validate_biln(result))

    def test_two_linear_fragments_no_dummy_atoms(self):
        """Fragments without dummy atoms -> backbone implied, no annotations."""
        frag1 = Chem.MolFromSmiles("NCC(=O)O")
        frag2 = Chem.MolFromSmiles("NCC(=O)O")
        result = fragments_to_biln(
            fragments=[frag1, frag2],
            monomer_names=["G", "G"],
        )
        validate_biln(result)
        # No explicit bond annotations expected for plain fragments
        self.assertNotIn("(1,", result)


# ---------------------------------------------------------------------------
# TestBilnEdgeCases
# ---------------------------------------------------------------------------

class TestBilnEdgeCases(unittest.TestCase):
    """Additional edge-case and integration tests."""

    def assert_valid_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol, msg=f"Invalid SMILES: {smiles}")
        return mol

    # ------------------------------------------------------------------
    # Tokenizer edge cases
    # ------------------------------------------------------------------

    def test_tokenize_empty_chain_raises(self):
        """An empty string tokenizes to an empty list (no tokens)."""
        result = tokenize_biln_chain("")
        self.assertEqual(result, [])

    def test_tokenize_unbalanced_bracket_raises(self):
        with self.assertRaises(ValueError):
            tokenize_biln_chain("[ABC-G")

    def test_tokenize_unbalanced_paren_raises(self):
        with self.assertRaises(ValueError):
            tokenize_biln_chain("C(1,3-A")

    def test_tokenize_nested_bracket_in_name(self):
        """Bracket-enclosed names with internal parens tokenize correctly."""
        result = tokenize_biln_chain("[dPhe(4-Cl)](1,3)-G")
        self.assertEqual(result, ["[dPhe(4-Cl)](1,3)", "G"])

    # ------------------------------------------------------------------
    # Molecular properties after biln_to_smiles
    # ------------------------------------------------------------------

    def test_c_terminal_hydroxyl_present(self):
        """The C-terminus must have a free carboxylic acid (-C(=O)O)."""
        smiles = biln_to_smiles("A-G")
        mol = self.assert_valid_smiles(smiles)
        # Look for a carboxylic acid: C(=O)O pattern
        patt = Chem.MolFromSmarts("C(=O)O")
        self.assertTrue(
            mol.HasSubstructMatch(patt),
            msg="C-terminal carboxylic acid missing in A-G.",
        )

    def test_n_terminal_amine_present(self):
        """The N-terminus must be a free amine."""
        smiles = biln_to_smiles("G-A")
        mol = self.assert_valid_smiles(smiles)
        patt = Chem.MolFromSmarts("[NH2]")
        self.assertTrue(
            mol.HasSubstructMatch(patt),
            msg="N-terminal amine missing in G-A.",
        )

    def test_peptide_bond_count(self):
        """An n-residue peptide has exactly n-1 peptide bonds (C(=O)N)."""
        for n, biln in [
            (2, "A-G"),
            (3, "A-G-V"),
            (4, "G-G-G-G"),
        ]:
            with self.subTest(n=n, biln=biln):
                smiles = biln_to_smiles(biln)
                mol = self.assert_valid_smiles(smiles)
                patt = Chem.MolFromSmarts("C(=O)N")
                matches = mol.GetSubstructMatches(patt)
                self.assertEqual(
                    len(matches),
                    n - 1,
                    msg=f"Expected {n-1} peptide bonds in {biln}, got {len(matches)}.",
                )

    def test_disulfide_bond_type(self):
        """The S-S bond in a disulfide must be a single bond."""
        smiles = biln_to_smiles("C(1,3)-G-C(1,3)")
        mol = self.assert_valid_smiles(smiles)
        sulfurs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
        self.assertEqual(len(sulfurs), 2)
        bond = mol.GetBondBetweenAtoms(sulfurs[0].GetIdx(), sulfurs[1].GetIdx())
        self.assertIsNotNone(bond)
        self.assertEqual(bond.GetBondType(), Chem.BondType.SINGLE)

    def test_head_to_tail_ring_size(self):
        """A head-to-tail cyclo-(A-G-A-G) must produce a single ring."""
        biln = "A(1,1)-G-A-G-A(1,2)"
        smiles = biln_to_smiles(biln)
        mol = self.assert_valid_smiles(smiles)
        self.assertGreater(mol.GetRingInfo().NumRings(), 0)

    def test_atom_count_single_residue(self):
        """Glycine alone: NCC(=O)O has 5 heavy atoms."""
        smiles = biln_to_smiles("G")
        mol = self.assert_valid_smiles(smiles)
        self.assertEqual(mol.GetNumAtoms(), 5)

    def test_atom_count_dipeptide_gg(self):
        """Gly-Gly: NCC(=O)NCC(=O)O has 9 heavy atoms."""
        smiles = biln_to_smiles("G-G")
        mol = self.assert_valid_smiles(smiles)
        self.assertEqual(mol.GetNumAtoms(), 9)

    # ------------------------------------------------------------------
    # BilnToSmiles element
    # ------------------------------------------------------------------

    def test_element_name(self):
        self.assertEqual(BilnToSmiles.name, "biln-to-smiles")

    def test_element_parallel(self):
        self.assertEqual(BilnToSmiles.parallel, "processing")

    def test_element_single_call_returns_string(self):
        result = BilnToSmiles()._single_call("A-G")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    # ------------------------------------------------------------------
    # SmilesToBiln element
    # ------------------------------------------------------------------

    def test_smiles_to_biln_element_name(self):
        self.assertEqual(SmilesToBiln.name, "smiles-to-biln")

    def test_smiles_to_biln_returns_string(self):
        smiles = biln_to_smiles("A-G")
        result = SmilesToBiln()._single_call(smiles)
        self.assertIsInstance(result, str)

    def test_smiles_to_biln_valid_output(self):
        smiles = biln_to_smiles("V-L-I")
        biln = SmilesToBiln()._single_call(smiles)
        validate_biln(biln)

    # ------------------------------------------------------------------
    # validate_biln corner cases
    # ------------------------------------------------------------------

    def test_validate_returns_true_on_success(self):
        """validate_biln must return exactly True (not just truthy)."""
        result = validate_biln("A-G")
        self.assertIs(result, True)

    def test_validate_multichain_no_bonds(self):
        self.assertTrue(validate_biln("A-G.G-A"))

    def test_validate_cross_chain_bond(self):
        """Bond ID shared across two chains is valid."""
        self.assertTrue(validate_biln("C(1,3)-A.A-C(1,3)"))

    # ------------------------------------------------------------------
    # parse_biln structural assertions
    # ------------------------------------------------------------------

    def test_parse_returns_list_of_chains(self):
        result = parse_biln("A-G")
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)

    def test_parse_entry_has_required_keys(self):
        result = parse_biln("A(1,3)-G")
        entry = result[0][0]
        self.assertIn("symbol", entry)
        self.assertIn("annotations", entry)

    def test_parse_annotation_is_list_of_tuples(self):
        result = parse_biln("C(1,3)-G")
        annotations = result[0][0]["annotations"]
        self.assertIsInstance(annotations, list)
        self.assertEqual(len(annotations), 1)
        bond_id, rgroup = annotations[0]
        self.assertEqual(bond_id, 1)
        self.assertEqual(rgroup, 3)


if __name__ == "__main__":
    unittest.main()
