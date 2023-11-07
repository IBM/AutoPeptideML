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
import networkx as nx
import os
import shutil
import math
from itertools import groupby
from typing import Dict, List, Tuple
import concurrent.futures

import pandas as pd
from tqdm.auto import tqdm


NORMALIZATIONS = {'shortest': lambda a,b,c: a/min(b,c), # a identity b len(seq1) c len(seq2)
                  'longest': lambda a,b,c: a/max(b,c),
                  'mean' : lambda a,b,c: a/((b+c)/2),
                  }


def get_len_dict(ids: List[str], seqs: List[str]) -> Dict[str,int]:
    '''Get a dictionary that contains the length of each sequence.'''
    len_dict = {}
    for id, seq in zip(ids, seqs):
        len_dict[id] = len(seq)
    
    return len_dict

def parse_fasta(fastafile: str, sep='|') -> Tuple[List[str],List[str]]:
    '''
    Parses fasta file into lists of identifiers and sequences.
	Can handle multi-line sequences and empty lines.
    Needleall seems to fail when a '|' is between the identifier and the rest of
    the fasta header, so we split the identifier and only return that.

    '''
    ids = []
    seqs = []
    with open(fastafile, 'r') as f:

        id_seq_groups = (group for group in groupby(f, lambda line: line.startswith(">")))

        for is_id, id_iter in id_seq_groups:
            if is_id: # Only needed to find first id line, always True thereafter
                ids.append(next(id_iter).strip().split(sep)[0])
                seqs.append("".join(seq.strip() for seq in next(id_seq_groups)[1]))
        
    return ids, seqs

def chunk_fasta_file(
    ids: List[str], 
    seqs: List[str],
    n_chunks: int,
    outputdir: StopAsyncIteration
) -> int:
    '''
    Break up fasta file into multiple smaller files that can be
    used for multiprocessing.
    Returns the number of generated chunks.
    '''

    chunk_size = math.ceil(len(ids)/n_chunks)

    empty_chunks = 0
    for i in range(n_chunks):
        # because of ceil() we sometimes make less partitions than specified.
        if i*chunk_size>=len(ids):
            empty_chunks +=1
            continue

        chunk_ids = ids[i*chunk_size:(i+1)*chunk_size]
        chunk_seqs = seqs[i*chunk_size:(i+1)*chunk_size]
        path = os.path.join(outputdir, f'{i}.fasta.tmp')
        with open(path, 'w') as f:
            for id, seq in zip(chunk_ids, chunk_seqs):
                f.write('>'+str(id)+'\n')
                f.write(seq+'\n')

    return n_chunks - empty_chunks

def compute_edges(query_fp: str,
                  library_fp: str,
                  threshold: float,
                  seq_lens: Dict[str,int],
                  denominator = 'full',
                  delimiter: str = '|',
                  is_nucleotide: bool = False,
                  gapopen: float = 10,
                  gapextend: float = 0.5,
                  endweight: bool = False,
                  endopen: float = 10,
                  endextend: float = 0.5,
                  matrix: str = 'EBLOSUM62',
                  ) -> Tuple[int, List[Tuple[str,str,float]]]:
    '''
    Run needleall on query_fp and library_fp,
    Retrieve pairwise similiarities, transform and
    insert into edge_dict.
    '''
    identity_list = []

    if is_nucleotide:
        type_1, type_2, = '-snucleotide1', '-snucleotide2'
    else:
        type_1, type_2 = '-sprotein1', '-sprotein2'

    command = ["needleall","-auto","-stdout", 
               "-aformat", "pair", 
               "-gapopen", str(gapopen),
               "-gapextend", str(gapextend),
               "-endopen", str(endopen),
               "-endextend", str(endextend),
               "-datafile", matrix,
               type_1, type_2, query_fp, library_fp]
    if endweight:
        command = command + ["-endweight"]

    count = 0
    import subprocess
    with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True) as proc:
        for line_nr, line in enumerate(proc.stdout):  
            if  line.startswith('# 1:'):

                # # 1: P0CV73
                this_qry = int(line[5:].split()[0].split('|')[0])

            elif line.startswith('# 2:'):
                this_lib = int(line[5:].split()[0].split('|')[0])

            elif line.startswith('# Identity:'):
                identity_line = line

            elif line.startswith('# Gaps:'):
                count +=1

                # Gaps:           0/142 ( 0.0%)
                gaps, rest = line[7:].split('/')
                gaps = int(gaps)
                length = int(rest.split('(')[0])

                
                # Compute different sequence identities as needed.
                if denominator == 'full': # full is returned by default. just need to parse
                    identity = float(identity_line.split('(')[1][:-3])/100
                elif denominator == 'no_gaps':
                    n_matches =  int(identity_line[11:].split('/')[0]) #int() does not mind leading spaces
                    identity = float(n_matches/(length-gaps))
                else:
                    n_matches =  int(identity_line[11:].split('/')[0]) #int() does not mind leading spaces
                    identity = NORMALIZATIONS[denominator](n_matches, seq_lens[this_qry], seq_lens[this_lib])
                #line = "# Identity:      14/443 ( 3.2%)"
                # n_matches =  int(line[11:].split('/')[0]) #int() does not mind leading spaces
                
                try:
                    metric = identity
                except ValueError or TypeError:
                    raise TypeError("Failed to interpret the identity value %r. Please ensure that the ggsearch36 output is correctly formatted." % (identity))
                
                if this_qry == this_lib:
                    continue
                
                identity_list.append((this_qry, this_lib, metric))
                # NOTE this case should raise an error - graph was constructed from same file before, and so all the nodes should be there.
 
    return (count, identity_list)

def generate_edges_needle(
    full_graph: nx.classes.graph.Graph,
    sequences: list,
    verbose: int,
    outputdir: str,
    denominator: str,
    threads: float,
    threshold: float,
    n_chunks: int = 10,
    n_procs: int = 4,
    parallel_mode: str = 'multithread',
    triangular: bool = False,
    delimiter: str = '|',
    is_nucleotide: bool = False,
    gapopen: float = 10,
    gapextend: float = 0.5,
    endweight: bool = True,
    endopen: float = 10,
    endextend: float = 0.5,
    matrix: str = 'EBLOSUM62',
    **kwargs
) -> None:
    '''
    Call needleall to compute all pairwise sequence identities in the dataset.
    Uses chunked fasta files and multiple threads with needelall subprocesses 
    to speed up computation.
    '''
    if shutil.which('needleall') is None:
        raise RuntimeError('EMBOSS needleall was not found. Please run `conda install -c bioconda emboss`')
    os.makedirs(outputdir, exist_ok=True)
    # chunk the input
    ids, seqs = [i for i, seq in enumerate(sequences)], sequences
    seq_lens = get_len_dict(ids, seqs)

    n_chunks = chunk_fasta_file(ids, seqs, n_chunks, outputdir) #get the actual number of generated chunks.

    # start n_procs threads, each thread starts a subprocess
    # Because of threading's GIL we can write edges directly to the full_graph object.
    jobs = []

    # this is approximate, but good enough for progress bar drawing.
    if triangular:
        n_alignments = 0
        chunk_size = math.ceil(len(ids)/n_chunks)
        for i in range(n_chunks):
            for j in range(i, n_chunks):
                n_alignments += chunk_size*chunk_size
    else:
        n_alignments = len(ids)*len(ids)

    # define in which interval each thread updates the progress bar.
    # if all update all the time, this would slow down the loop and make runtimes estimate unstable.
    # update every 1000 OR update every 0.05% of the total, divided by number of procs. 
    # this worked well on large datasets with 64 threads - fewer threads should then be unproblematic.
    pbar_update_interval = int((n_alignments * 0.0005)/n_procs) 
    pbar_update_interval = min(1000, pbar_update_interval)

    #pbar = tqdm(total= n_alignments)

    if parallel_mode == 'multithread':
        executor_cls = concurrent.futures.ThreadPoolExecutor
    elif parallel_mode == 'multiprocess':
        executor_cls = concurrent.futures.ProcessPoolExecutor

    with executor_cls(max_workers=n_procs) as executor:
        for i in range(n_chunks):
            start = i if triangular else 0
            for j in range(start, n_chunks):
                q = os.path.join(outputdir, f'{i}.fasta.tmp')
                l = os.path.join(outputdir, f'{j}.fasta.tmp')
                future = executor.submit(compute_edges, q, l, threshold, seq_lens, denominator, delimiter, is_nucleotide, gapopen, gapextend, endweight, endopen, endextend, matrix)
                jobs.append(future)

        if verbose > 1:
            pbar = tqdm(jobs)
        else:
            pbar = jobs
        for job in pbar:
            if job.exception() is not None:
                print(job.exception())
                raise RuntimeError('One of the alignment processes did not complete sucessfully.')
            else:
                count, chunk_identities = job.result()

                # while we wait on more jobs to finish, we can parse results as they come in.
                for this_qry, this_lib, metric in chunk_identities:
                    if not full_graph.has_node(this_qry) or not full_graph.has_node(this_lib):
                        raise RuntimeError(f'Tried to insert edge {this_qry}-{this_lib} into the graph, but did not find nodes. This should not happen, please report a bug.')
                    if full_graph.has_edge(this_qry, this_lib):
                        if full_graph[this_qry][this_lib]['metric'] > metric:
                            full_graph.add_edge(this_qry, this_lib, metric=metric) #Notes: Adding an edge that already exists updates the edge data. 
                            
                            # this seems to be slow:
                            #nx.set_edge_attributes(full_graph,{(this_qry,this_lib):metric}, 'metric')
                    else:
                        full_graph.add_edge(this_qry, this_lib, metric=metric) 
                # pbar.update(count)

    #delete the chunks
    for i in range(n_chunks):
        os.remove(os.path.join(outputdir, f'{i}.fasta.tmp'))
