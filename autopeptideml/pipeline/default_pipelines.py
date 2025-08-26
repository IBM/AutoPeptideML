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
    from .smiles import SequenceToSmiles, FilterSmiles
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
    substitution: str = 'X',
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
    """
    Retrieve a predefined processing pipeline by name.

    Supported pipeline names:

    - ``'to-smiles'``: Converts a sequence to a canonical SMILES representation.
      Performs cleaning and full canonicalization.

    - ``'to-smiles-fast'``: Converts a sequence to SMILES quickly with minimal processing.
      Skips canonicalization for performance.

    - ``'to-sequences'``: Converts SMILES back to a cleaned sequence, optionally preserving analog information.

    Keyword arguments depend on the pipeline selected:

    - For ``'to-smiles'`` and ``'to-smiles-fast'``:
        - ``substitution`` (str): A character used to replace non-canonical residues in sequences.
          Default is ``'G'``.

    - For ``'to-sequences'``:
        - ``substitution`` (str): A character used for cleaning sequences. Default is ``'G'``.
        - ``keep_analog`` (bool): Whether to substitute non-canonical residues for their natural information during SMILES-to-sequence conversion. Default is ``True``.

    :param name: The name of the pipeline to retrieve. Must be one of: ``'to-smiles'``, ``'to-smiles-fast'``, ``'to-sequences'``.
    :type name: str
    :param kwargs: Additional keyword arguments passed to the selected pipeline constructor.
    :raises ValueError: If the provided name is not a valid pipeline.
    :return: An instance of the requested processing pipeline.
    :rtype: Pipeline
    """
    if name in PIPELINES:
        return PIPELINES[name](**kwargs)
    else:
        raise ValueError(f"Pipeline {name} does not exist. Please try: {', '.join(PIPELINES.keys())}")
