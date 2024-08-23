from autopeptideml import AutoPeptideML


def test_load():
    apml = AutoPeptideML()
    df = apml.curate_dataset('examples/AB_positives.csv')
    assert len(df) == 6_583
