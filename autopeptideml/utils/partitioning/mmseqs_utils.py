"""
Code adapted from the Graph-Part 
Github Repository:
https://github.com/graph-part/graph-part

under BSD 3-Clause License;

Copyright (c) 2023, F. Teufel and M. H. Gíslason

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from multiprocessing import cpu_count
import os
import shutil
import subprocess

import networkx as nx


def _write_fasta(sequences: list, filename: str):
    with open(filename, 'w') as file:
        for idx, sequence in enumerate(sequences):
            file.write(f'>{idx}\n{sequence}\n')

def generate_edges_mmseqs(
    full_graph: nx.classes.graph.Graph,
    sequences: list,
    outputdir : str = 'tmp',
    denominator: str = 'longest',
    is_nucleotide: bool = False,
    use_prefilter: bool = True,
    verbose: int = 2,
    threads: int = cpu_count(),
    threshold: float = 0.3,
    **kwargs
) -> None:
    if verbose > 2:
        mmseqs_v = 3
    else:
        mmseqs_v = verbose
    if verbose > 0:
        if use_prefilter:
            print('Calculating pairwise alignments using MMSeqs2 algorithm with prefilter...')
        else:
            print('Calculating pairwise alignments using MMSeqs2 algorithm...')

    if shutil.which('mmseqs') is None:
        raise RuntimeError('MMseqs2 was not found. Please run `conda install -c conda-forge -c bioconda mmseqs2`')

    os.makedirs(outputdir, exist_ok=True)
    db_file = os.path.join(outputdir, 'db.fasta')
    _write_fasta(sequences, db_file)

    # Run all mmseqs ops to get a tab file that contains the alignments.
    typ = '2' if is_nucleotide else '1'
    subprocess.run(['mmseqs', 'createdb', '--dbtype', typ, db_file, f'{outputdir}/seq_db', '-v', '1'])

    # However, this function will not work with nucleotidenucleotide searches, 
    # since we need to have a valid diagonal for the banded alignment.

    if is_nucleotide or use_prefilter:
        subprocess.run([
            'mmseqs', 'prefilter', '-s', '7.5', f'{outputdir}/seq_db', 
            f'{outputdir}/seq_db', f'{outputdir}/pref', '-v', f'{mmseqs_v}'
        ])
    else:
        fake_prefilter = os.path.join(
            os.path.realpath(os.path.dirname(__file__)), 'mmseqs_fake_prefilter.sh'
        )
        subprocess.run(
            [
                fake_prefilter, 
                f'{outputdir}/seq_db', f'{outputdir}/seq_db', f'{outputdir}/pref', 'seq_db']
        )

    # 0: alignment length 1: shorter, 2: longer sequence
    id_mode = {'n_aligned': '0', 'shortest': '1', 'longest': '2'}[denominator]
    
    command = [
        'mmseqs', 'align',  f'{outputdir}/seq_db', f'{outputdir}/seq_db',
        f'{outputdir}/pref', f'{outputdir}/align_db',
        '--alignment-mode', '3', '-e', 'inf', '--seq-id-mode', id_mode, '-v', f'{mmseqs_v}',
        '--threads', f'{threads}'
    ]
    subprocess.run(command)

    file = os.path.join(outputdir, 'alignments.tab')
    subprocess.run(
        ['mmseqs', 'convertalis', f'{outputdir}/seq_db', f'{outputdir}/seq_db', f'{outputdir}/align_db', 
         file, '-v', '1']
    )

    filedata = open(file).readlines()

    if verbose > 0:
        print(f'Building Graph...')

    for row in filedata:
        row = row.strip('\n').split('\t')
        query, target, metric = int(row[0]), int(row[1]), float(row[2])

        if query == target:
            continue
        if metric < threshold:
            continue
        if full_graph.has_edge(query, target):
            if full_graph[query][target]['metric'] < metric:
                full_graph.add_edge(query, target, metric=metric) # Notes: Adding an edge that already exists updates the edge data. 
        else:
            full_graph.add_edge(query, target, metric=metric)  

    shutil.rmtree(outputdir)
