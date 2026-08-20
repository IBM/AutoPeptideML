import unittest

from rdkit import Chem

from autopeptideml.pipeline.smiles import (
    SmilesToBiln,
    biln_to_smiles,
    parse_biln,
    validate_biln,
)


class TestSmilesToBilnComprehensive(unittest.TestCase):
    """Behavioral and structural tests for SMILES -> BILN conversion."""

    @classmethod
    def setUpClass(cls):
        cls.converter = SmilesToBiln()

        cls.standard = [
            "A", "C", "D", "E", "F",
            "G", "H", "I", "K", "L",
            "M", "N", "P", "Q", "R",
            "S", "T", "V", "W", "Y",
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def convert(self, smiles):
        result = self.converter._single_call(smiles)
        self.assertIsInstance(result, str)
        self.assertTrue(result)
        return result

    def assert_valid_biln(self, biln):
        self.assertIsInstance(biln, str)
        self.assertTrue(biln)
        self.assertTrue(validate_biln(biln))
        return biln

    def assert_same_molecule(self, smiles1, smiles2):
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        self.assertIsNotNone(mol1, smiles1)
        self.assertIsNotNone(mol2, smiles2)

        canonical1 = Chem.MolToSmiles(
            mol1,
            canonical=True,
            isomericSmiles=True,
        )
        canonical2 = Chem.MolToSmiles(
            mol2,
            canonical=True,
            isomericSmiles=True,
        )

        self.assertEqual(canonical1, canonical2)

    def assert_biln_residues(self, biln, expected):
        """Assert that a simple linear BILN has exactly the expected residues."""
        chains = parse_biln(biln)

        self.assertEqual(
            len(chains),
            1,
            msg=f"Expected one chain, got {chains!r}",
        )

        symbols = [entry["symbol"] for entry in chains[0]]

        self.assertEqual(symbols, expected)

    def parsed_symbols(self, biln):
        chains = parse_biln(biln)
        return [
            [entry["symbol"] for entry in chain]
            for chain in chains
        ]

    def parsed_annotations(self, biln):
        return [
            annotation
            for chain in parse_biln(biln)
            for entry in chain
            for annotation in entry["annotations"]
        ]

    # ------------------------------------------------------------------
    # Standard amino acids
    #
    # These use independently specified SMILES rather than
    # biln_to_smiles(), so the SMILES -> BILN mapping is actually tested.
    # ------------------------------------------------------------------

    STANDARD_SMILES = {
        "A": "N[C@@H](C)C(=O)O",
        "C": "N[C@@H](CS)C(=O)O",
        "D": "N[C@@H](CC(=O)O)C(=O)O",
        "E": "N[C@@H](CCC(=O)O)C(=O)O",
        "F": "N[C@@H](Cc1ccccc1)C(=O)O",
        "G": "NCC(=O)O",
        "H": "N[C@@H](Cc1cnc[nH]1)C(=O)O",
        "I": "N[C@@H]([C@H](C)CC)C(=O)O",
        "K": "N[C@@H](CCCCN)C(=O)O",
        "L": "N[C@@H](CC(C)C)C(=O)O",
        "M": "N[C@@H](CCSC)C(=O)O",
        "N": "N[C@@H](CC(=O)N)C(=O)O",
        "P": "N1[C@@H](CCC1)C(=O)O",
        "Q": "N[C@@H](CCC(=O)N)C(=O)O",
        "R": "N[C@@H](CCCNC(N)=N)C(=O)O",
        "S": "N[C@@H](CO)C(=O)O",
        "T": "N[C@@H]([C@H](O)C)C(=O)O",
        "V": "N[C@@H](C(C)C)C(=O)O",
        "W": "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
        "Y": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
    }

    def test_all_twenty_standard_amino_acids(self):
        """Each canonical L-amino acid is recognized correctly."""
        for aa in self.standard:
            with self.subTest(aa=aa):
                biln = self.convert(self.STANDARD_SMILES[aa])

                self.assert_valid_biln(biln)
                self.assert_biln_residues(biln, [aa])

    def test_all_twenty_standard_amino_acids_in_dipeptides(self):
        """Each canonical residue survives in the N-terminal position."""
        for aa in self.standard:
            with self.subTest(aa=aa):
                original = f"{aa}-G"
                smiles = biln_to_smiles(original)

                biln = self.convert(smiles)

                self.assert_valid_biln(biln)
                self.assert_biln_residues(biln, [aa, "G"])

    def test_all_twenty_standard_amino_acids_as_c_terminal_residue(self):
        """Each canonical residue survives in the C-terminal position."""
        for aa in self.standard:
            with self.subTest(aa=aa):
                original = f"G-{aa}"
                smiles = biln_to_smiles(original)

                biln = self.convert(smiles)

                self.assert_valid_biln(biln)
                self.assert_biln_residues(biln, ["G", aa])

    # ------------------------------------------------------------------
    # Direct semantic recognition
    # ------------------------------------------------------------------

    def test_glycine(self):
        self.assertEqual(
            self.convert("NCC(=O)O"),
            "G",
        )

    def test_alanine(self):
        biln = self.convert("N[C@@H](C)C(=O)O")
        self.assert_biln_residues(biln, ["A"])

    def test_alanine_glycine(self):
        smiles = biln_to_smiles("A-G")
        biln = self.convert(smiles)
        self.assert_biln_residues(biln, ["A", "G"])

    def test_glycine_alanine(self):
        smiles = biln_to_smiles("G-A")
        biln = self.convert(smiles)
        self.assert_biln_residues(biln, ["G", "A"])

    def test_fwl_peptide(self):
        smiles = biln_to_smiles("F-W-L")
        biln = self.convert(smiles)
        self.assert_biln_residues(biln, ["F", "W", "L"])

    def test_polar_peptide(self):
        smiles = biln_to_smiles("S-T-N-Q")
        biln = self.convert(smiles)
        self.assert_biln_residues(biln, ["S", "T", "N", "Q"])

    # ------------------------------------------------------------------
    # Equivalent SMILES representations
    #
    # These test canonicalization only if the converter promises it.
    # ------------------------------------------------------------------

    def test_equivalent_alanine_smiles(self):
        # All three encode L-alanine (CIP S).
        # Note: C[C@@H](N)C(=O)O has CIP R (D-alanine) and is intentionally
        # excluded here; the test_l_alanine_and_d_alanine_are_distinguished
        # test verifies L/D distinction.
        smiles_variants = [
            "N[C@@H](C)C(=O)O",        # standard L-Ala SMILES
            "O=C(O)[C@@H](N)C",        # same molecule, different atom order
            "C[C@H](N)C(=O)O",         # RDKit canonical form of L-Ala
        ]

        results = [self.convert(s) for s in smiles_variants]

        self.assertEqual(
            len(set(results)),
            1,
            msg=f"Equivalent alanine SMILES produced different BILN: {results}",
        )

    def test_equivalent_glycine_smiles(self):
        smiles_variants = [
            "NCC(=O)O",
            "C(C(=O)O)N",
            "O=C(O)CN",
        ]

        results = [self.convert(s) for s in smiles_variants]

        self.assertEqual(
            len(set(results)),
            1,
            msg=f"Equivalent glycine SMILES produced different BILN: {results}",
        )

    def test_equivalent_dipeptide_smiles(self):
        original = "A-G"
        reference = biln_to_smiles(original)

        mol = Chem.MolFromSmiles(reference)
        self.assertIsNotNone(mol)

        variants = [
            reference,
            Chem.MolToSmiles(
                mol,
                canonical=False,
                isomericSmiles=True,
            ),
            Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=True,
            ),
        ]

        results = [self.convert(s) for s in variants]

        self.assertEqual(
            len(set(results)),
            1,
            msg=f"Equivalent peptide SMILES produced different BILN: {results}",
        )
        self.assert_biln_residues(results[0], ["A", "G"])

    # ------------------------------------------------------------------
    # Stereochemistry
    # ------------------------------------------------------------------

    def test_l_alanine_and_d_alanine_are_distinguished(self):
        l_alanine = "N[C@@H](C)C(=O)O"
        d_alanine = "N[C@H](C)C(=O)O"

        l_biln = self.convert(l_alanine)
        d_biln = self.convert(d_alanine)

        self.assert_valid_biln(l_biln)
        self.assert_valid_biln(d_biln)

        self.assert_biln_residues(l_biln, ["A"])
        self.assert_biln_residues(d_biln, ["dA"])

        self.assertNotEqual(l_biln, d_biln)

    def test_stereochemistry_survives_round_trip(self):
        original = biln_to_smiles("A-G")

        biln = self.convert(original)
        regenerated = biln_to_smiles(biln)

        self.assert_same_molecule(original, regenerated)

    def test_unspecified_stereochemistry_does_not_crash(self):
        smiles = "NC(C)C(=O)O"

        biln = self.convert(smiles)

        self.assert_valid_biln(biln)

    # ------------------------------------------------------------------
    # Proline
    # ------------------------------------------------------------------

    def test_proline(self):
        biln = self.convert("N1[C@@H](CCC1)C(=O)O")
        self.assert_biln_residues(biln, ["P"])

    def test_proline_dipeptide(self):
        biln = self.convert(biln_to_smiles("P-G"))
        self.assert_biln_residues(biln, ["P", "G"])

    def test_peptide_containing_multiple_prolines(self):
        biln = self.convert(biln_to_smiles("P-A-P-G"))
        self.assert_biln_residues(biln, ["P", "A", "P", "G"])

    # ------------------------------------------------------------------
    # Aromatic and charged residues
    # ------------------------------------------------------------------

    def test_aromatic_residues(self):
        for aa in ["F", "W", "Y", "H"]:
            with self.subTest(aa=aa):
                biln = self.convert(biln_to_smiles(f"{aa}-G"))
                self.assert_biln_residues(biln, [aa, "G"])

    def test_charged_residues(self):
        for aa in ["D", "E", "K", "R", "H"]:
            with self.subTest(aa=aa):
                biln = self.convert(biln_to_smiles(f"{aa}-G"))
                self.assert_biln_residues(biln, [aa, "G"])

    # ------------------------------------------------------------------
    # Terminal chemistry
    # ------------------------------------------------------------------

    def test_linear_peptide_has_free_terminals(self):
        biln = self.convert(biln_to_smiles("G-A"))

        self.assert_biln_residues(biln, ["G", "A"])

        # A normal linear peptide should not acquire arbitrary
        # terminal annotations.
        annotations = self.parsed_annotations(biln)
        self.assertEqual(annotations, [])

    def test_n_acetylated_input_preserves_structure(self):
        smiles = "CC(=O)N[C@@H](C)C(=O)O"

        biln = self.convert(smiles)
        self.assert_valid_biln(biln)

        regenerated = biln_to_smiles(biln)
        self.assert_same_molecule(smiles, regenerated)

    def test_c_terminal_amide_input_preserves_structure(self):
        smiles = "N[C@@H](C)C(=O)N"

        biln = self.convert(smiles)
        self.assert_valid_biln(biln)

        regenerated = biln_to_smiles(biln)
        self.assert_same_molecule(smiles, regenerated)

    # ------------------------------------------------------------------
    # Disulfide topology
    # ------------------------------------------------------------------

    def test_single_disulfide(self):
        original = "C(1,3)-A-A-A-C(1,3)"
        smiles = biln_to_smiles(original)

        biln = self.convert(smiles)
        self.assert_valid_biln(biln)

        annotations = self.parsed_annotations(biln)

        self.assertEqual(len(annotations), 2)

        self.assertEqual(
            annotations[0][0],
            annotations[1][0],
        )
        self.assertEqual(
            [a[1] for a in annotations],
            [3, 3],
        )

        bond_id = annotations[0][0]
        self.assertEqual(
            sum(a[0] == bond_id for a in annotations),
            2,
        )

        regenerated = biln_to_smiles(biln)
        self.assert_same_molecule(smiles, regenerated)

    def test_double_disulfide(self):
        original = "C(1,3)-C(2,3)-G-C(1,3)-C(2,3)"
        smiles = biln_to_smiles(original)

        biln = self.convert(smiles)
        self.assert_valid_biln(biln)

        annotations = self.parsed_annotations(biln)

        self.assertEqual(len(annotations), 4)

        bond_counts = {}
        for bond_id, _ in annotations:
            bond_counts[bond_id] = bond_counts.get(bond_id, 0) + 1

        self.assertEqual(
            sorted(bond_counts.values()),
            [2, 2],
        )

        self.assertTrue(
            all(rgroup == 3 for _, rgroup in annotations)
        )

        regenerated = biln_to_smiles(biln)
        self.assert_same_molecule(smiles, regenerated)

    # ------------------------------------------------------------------
    # Head-to-tail cyclic peptides
    # ------------------------------------------------------------------

    def test_head_to_tail_cycle(self):
        original = "C(1,1)-A-A-A-C(1,2)"
        smiles = biln_to_smiles(original)

        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol)
        self.assertGreater(mol.GetRingInfo().NumRings(), 0)

        biln = self.convert(smiles)
        self.assert_valid_biln(biln)

        annotations = self.parsed_annotations(biln)

        self.assertEqual(len(annotations), 2)

        self.assertEqual(
            {bond_id for bond_id, _ in annotations},
            {1},
        )
        self.assertEqual(
            sorted(rgroup for _, rgroup in annotations),
            [1, 2],
        )

        regenerated = biln_to_smiles(biln)
        self.assert_same_molecule(smiles, regenerated)

    # ------------------------------------------------------------------
    # Multiple chains
    # ------------------------------------------------------------------

    def test_two_independent_chains(self):
        original = "A-G.G-A"
        smiles = biln_to_smiles(original)

        biln = self.convert(smiles)
        self.assert_valid_biln(biln)

        self.assertEqual(
            self.parsed_symbols(biln),
            [["A", "G"], ["G", "A"]],
        )

        regenerated = biln_to_smiles(biln)
        self.assert_same_molecule(smiles, regenerated)

    def test_three_independent_chains(self):
        original = "G.A.V"
        smiles = biln_to_smiles(original)

        biln = self.convert(smiles)
        self.assert_valid_biln(biln)

        chains = parse_biln(biln)
        self.assertEqual(len(chains), 3)

        # RDKit reorders disconnected components by internal atom index,
        # so the chain order in the BILN may differ from the original BILN
        # string.  Check that each residue appears in exactly one chain.
        all_symbols = sorted(
            symbol
            for chain in self.parsed_symbols(biln)
            for symbol in chain
        )
        self.assertEqual(all_symbols, ["A", "G", "V"])

    def test_cross_chain_disulfide(self):
        original = "C(1,3)-A.A-C(1,3)"
        smiles = biln_to_smiles(original)

        biln = self.convert(smiles)
        self.assert_valid_biln(biln)

        chains = parse_biln(biln)
        self.assertEqual(len(chains), 2)

        # Chain order is not semantically significant; check that both
        # chains are present with the expected residues (order-independent).
        parsed = self.parsed_symbols(biln)
        self.assertEqual(sorted(parsed), sorted([["C", "A"], ["A", "C"]]))

        annotations = self.parsed_annotations(biln)

        self.assertEqual(len(annotations), 2)
        self.assertEqual(
            annotations[0][0],
            annotations[1][0],
        )
        self.assertEqual(annotations[0][1], 3)
        self.assertEqual(annotations[1][1], 3)

        regenerated = biln_to_smiles(biln)
        self.assert_same_molecule(smiles, regenerated)

    # ------------------------------------------------------------------
    # Noncanonical / modified residues
    #
    # These deliberately test round-trip preservation rather than
    # assuming an exact library spelling for the residue.
    # ------------------------------------------------------------------

    def test_x1_round_trip(self):
        original = "X1"
        smiles = biln_to_smiles(original)

        biln = self.convert(smiles)

        self.assert_valid_biln(biln)

        regenerated = biln_to_smiles(biln)
        self.assert_same_molecule(smiles, regenerated)

    def test_ddpr_round_trip(self):
        original = "dDpr"
        smiles = biln_to_smiles(original)

        biln = self.convert(smiles)

        self.assert_valid_biln(biln)

        regenerated = biln_to_smiles(biln)
        self.assert_same_molecule(smiles, regenerated)

    def test_x2038_round_trip(self):
        original = "X2038"
        smiles = biln_to_smiles(original)

        biln = self.convert(smiles)

        self.assert_valid_biln(biln)

        regenerated = biln_to_smiles(biln)
        self.assert_same_molecule(smiles, regenerated)

    def test_noncanonical_residue_in_peptide(self):
        original = "A-X1-G"
        smiles = biln_to_smiles(original)

        biln = self.convert(smiles)

        self.assert_valid_biln(biln)

        regenerated = biln_to_smiles(biln)
        self.assert_same_molecule(smiles, regenerated)

    # ------------------------------------------------------------------
    # Longer peptides
    # ------------------------------------------------------------------

    def test_long_linear_peptide(self):
        original = "A-G-V-L-I-P-F-W-Y-S-T-N-Q-K-R-M-H-D-E-C"
        smiles = biln_to_smiles(original)

        biln = self.convert(smiles)

        self.assert_valid_biln(biln)
        self.assert_biln_residues(
            biln,
            [
                "A", "G", "V", "L", "I",
                "P", "F", "W", "Y", "S",
                "T", "N", "Q", "K", "R",
                "M", "H", "D", "E", "C",
            ],
        )

    # ------------------------------------------------------------------
    # Invalid SMILES
    # ------------------------------------------------------------------

    def test_empty_smiles(self):
        with self.assertRaises((ValueError, TypeError)):
            self.convert("")

    def test_none_smiles(self):
        with self.assertRaises((ValueError, TypeError)):
            self.convert(None)

    def test_malformed_smiles(self):
        with self.assertRaises((ValueError, TypeError)):
            self.convert("C1CC")

    def test_garbage_input(self):
        with self.assertRaises((ValueError, TypeError)):
            self.convert("this-is-not-smiles")

    def test_unbalanced_smiles(self):
        with self.assertRaises((ValueError, TypeError)):
            self.convert("NCC(=O")

    # ------------------------------------------------------------------
    # Valid but unsupported/non-peptide molecules
    # ------------------------------------------------------------------

    def test_benzene_is_rejected(self):
        with self.assertRaises((ValueError, KeyError)):
            self.convert("c1ccccc1")

    def test_ethanol_is_rejected(self):
        with self.assertRaises((ValueError, KeyError)):
            self.convert("CCO")

    def test_water_is_rejected(self):
        with self.assertRaises((ValueError, KeyError)):
            self.convert("O")

    def test_sodium_chloride_is_rejected(self):
        with self.assertRaises((ValueError, KeyError)):
            self.convert("[Na+].[Cl-]")

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_conversion_is_deterministic(self):
        smiles = biln_to_smiles("A-G-V-L-I-P")

        results = {
            self.convert(smiles)
            for _ in range(20)
        }

        self.assertEqual(
            len(results),
            1,
            msg=f"Non-deterministic BILN results: {results}",
        )

    # ------------------------------------------------------------------
    # Structural round trips
    # ------------------------------------------------------------------

    def test_round_trip_all_standard_amino_acids(self):
        for aa in self.standard:
            with self.subTest(aa=aa):
                smiles1 = biln_to_smiles(aa)
                biln = self.convert(smiles1)
                smiles2 = biln_to_smiles(biln)

                self.assert_same_molecule(smiles1, smiles2)

    def test_round_trip_representative_peptides(self):
        peptides = [
            "A-G",
            "G-G-G",
            "F-W-Y",
            "P-A-G",
            "D-E-K-R",
            "S-T-N-Q",
            "A-G-V-L-I-P-F-W",
        ]

        for original in peptides:
            with self.subTest(original=original):
                smiles1 = biln_to_smiles(original)
                biln = self.convert(smiles1)
                smiles2 = biln_to_smiles(biln)

                self.assert_same_molecule(smiles1, smiles2)

    def test_round_trip_disulfide(self):
        original = "C(1,3)-A-A-A-C(1,3)"

        smiles1 = biln_to_smiles(original)
        biln = self.convert(smiles1)
        smiles2 = biln_to_smiles(biln)

        self.assert_same_molecule(smiles1, smiles2)

    def test_round_trip_cross_chain_disulfide(self):
        original = "C(1,3)-A.A-C(1,3)"

        smiles1 = biln_to_smiles(original)
        biln = self.convert(smiles1)
        smiles2 = biln_to_smiles(biln)

        self.assert_same_molecule(smiles1, smiles2)


if __name__ == "__main__":
    unittest.main()
