from .pipeline import Pipeline


def _tosmiles(substitution: str = 'G') -> Pipeline:
    from .smiles import SequenceToSmiles, CanonicalizeSmiles, FilterSmiles
    from .sequence import CanonicalCleaner
    stream_1a = Pipeline([FilterSmiles(keep_smiles=False),
                         CanonicalCleaner(substitution=substitution),
                         SequenceToSmiles()],
                         name='to-smiles-1a')
    stream_1b = Pipeline([FilterSmiles(keep_smiles=True)],
                         name='to-smiles-1b')
    stream_1 = Pipeline([stream_1a, stream_1b],
                        name='to-smiles-1',
                        aggregate=True)

    pipe = Pipeline([stream_1, CanonicalizeSmiles()],
                    name='to-smiles')
    return pipe


def _tosmiles_fast(substitution: str = 'G') -> Pipeline:
    from .smiles import SequenceToSmiles, CanonicalizeSmiles, FilterSmiles
    from .sequence import CanonicalCleaner
    stream_1a = Pipeline([FilterSmiles(keep_smiles=False),
                         CanonicalCleaner(substitution=substitution),
                         SequenceToSmiles()],
                         name='to-smiles-1a')
    stream_1b = Pipeline([FilterSmiles(keep_smiles=True)],
                         name='to-smiles-1b')
    stream_1 = Pipeline([stream_1a, stream_1b],
                        name='to-smiles-1',
                        aggregate=True)

    pipe = Pipeline([stream_1],
                    name='to-smiles')
    return pipe


def _to_sequences(
    substitution: str = 'G',
    keep_analog: bool = True
) -> Pipeline:
    from .smiles import FilterSmiles, SmilesToSequence
    from .sequence import CanonicalCleaner
    stream_1a = Pipeline([FilterSmiles(keep_smiles=False)],
                         name='to-sequences-1a')
    stream_1b = Pipeline([FilterSmiles(keep_smiles=True),
                          SmilesToSequence(keep_analog=keep_analog)],
                         name='to-sequences-1b')
    stream_1 = Pipeline([stream_1a, stream_1b],
                        name='to-sequences-1',
                        aggregate=True)

    pipe = Pipeline([stream_1, CanonicalCleaner(substitution)],
                    name='to-sequences')
    return pipe


PIPELINES = {
    'to-smiles': _tosmiles,
    'to-smiles-fast': _tosmiles_fast,
    'to-sequences': _to_sequences
}


def get_pipeline(name: str, **kwargs) -> Pipeline:
    if name in PIPELINES:
        return PIPELINES[name](**kwargs)
    else:
        raise ValueError(f"Pipeline {name} does not exist. Please try: {', '.join(PIPELINES.keys())}")
