import os
from multiprocessing import Manager
import warnings
import scanpy as sc
from threading import Thread
from typing import (
    Any, Callable, Optional, Sequence, Union, Iterable
)
from joblib import delayed, Parallel
import numpy as np
from scipy.sparse import issparse
import pandas as pd 
import tqdm 
from tqdm.contrib.concurrent import process_map, thread_map
from tqdm.asyncio import tqdm as asyn_tqdm
from multiprocessing import cpu_count
import itertools 
import scipy
from scipy.sparse import csr_matrix
from typing import Sequence, Tuple, Union, Optional, Callable
import contextlib
import joblib
# from ..utils._compat import Literal

class Parallelizer:
    def __init__(self, n_jobs:int):
        self.n_jobs = self.get_n_jobs(n_jobs=n_jobs)
        self._msg_shown = False 
        
    def get_n_jobs(self, n_jobs):
        if n_jobs is None or (n_jobs < 0 and os.cpu_count() + 1 + n_jobs <= 0):
            return 1
        elif n_jobs > os.cpu_count():
            return os.cpu_count()
        elif n_jobs < 0:
            return os.cpu_count() + 1 + n_jobs
        else:
            return n_jobs

    @staticmethod
    def __update__(pbar, queue, n_total):
        n_finished = 0
        while n_finished < n_total:
            try:
                res = queue.get()
            except EOFError as e:
                if not n_finished != n_total:
                    raise RuntimeError(
                        f"Finished only `{n_finished} out of `{n_total}` tasks.`"
                    ) from e
                break
            assert res in (None, (1, None), 1, 0)  # (None, 1) means only 1 job
            if res == (1, None):
                n_finished += 1
                if pbar is not None:
                    pbar.update()
            elif res == 0:
                n_finished += 1
            elif res == 1:
                if pbar is not None:
                    pbar.update()
        if pbar is not None:
            pbar.close()
        
    def parallelize(
        self,
        map_func: Callable[[Any], Any],
        map_data: Union[Sequence[Any], Iterable[Any]],
        n_split: Optional[int] = None,
        progress: bool = False,
        progress_unit: str = "",
        use_ixs: bool = False,
        backend: str = "loky",
        reduce_func: Optional[Callable[[Any], Any]] = None,
        reduce_as_array: bool = True,
    ):
        if progress:
            try:
                try:
                    from tqdm.asyncio import tqdm
                except ImportError:
                    try:
                        from tqdm.notebook import tqdm
                    except ImportError:
                        try:
                            from tqdm import tqdm_notebook as tqdm
                        except ImportError:
                            from tqdm import tqdm
                    import ipywidgets  # noqa
            except ImportError:
                global _msg_shown
                tqdm = None

                self._msg_shown = True
        else:
            tqdm = None

        col_len = map_data.shape[0] if issparse(map_data) else len(map_data)

        if n_split is None:
            n_split = self.n_jobs

        if issparse(map_data):
            if n_split == map_data.shape[0]:
                map_datas = [map_data[[ix], :] for ix in range(map_data.shape[0])]
            else:
                step = map_data.shape[0] // n_split

                ixs = [
                    np.arange(i * step, min((i + 1) * step, map_data.shape[0]))
                    for i in range(n_split)
                ]
                ixs[-1] = np.append(
                    ixs[-1], np.arange(ixs[-1][-1] + 1, map_data.shape[0])
                )

                map_datas = [map_data[ix, :] for ix in filter(len, ixs)]
        else:
            map_datas = list(filter(len, np.array_split(map_data, n_split)))

        pass_queue = not hasattr(map_func, "py_func")  # we'd be inside a numba function

        def wrapper(*args, **kwargs):
            if pass_queue and progress:
                pbar = None if tqdm is None else tqdm(total=len(map_data), unit=progress_unit)
                queue = Manager().Queue()
                thread = Thread(target=Parallelizer.__update__, args=(pbar, queue, len(map_datas)))
                thread.start()
            else:
                pbar, queue, thread = None, None, None

            res = Parallel(n_jobs=self.n_jobs, backend=backend)(
                delayed(map_func)(
                    *((i, cs) if use_ixs else (cs,)),
                    *args,
                    **kwargs,
                    queue=queue,
                )
                for i, cs in enumerate(map_datas)
            )

            res = np.array(res) if reduce_as_array else res
            if thread is not None:
                thread.join()

            return res if reduce_func is None else reduce_func(res)
        return wrapper

def parallel_leiden_computation(
    X: np.ndarray,
    n_neighbors_list: Iterable[int] = (30, 50, 70, 90),
    resolution_list: Iterable[float] = (0.4, 0.6, 0.8, 1.0, 1.2),
):
    def par_func(data, queue=None):
        out = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") 
            for i in data:
                X, n_neighbors, resolution = i["X"], i["n_neighbors"], i["resolution"]
                adata = sc.AnnData(X=X)
                sc.pp.neighbors(adata, n_neighbors=n_neighbors)
                sc.tl.leiden(adata, resolution=resolution)
                if queue is not None:
                    queue.put(1)
                out[(n_neighbors, resolution)] = adata.obs["leiden"].values
        if queue is not None:
            queue.put(0)
        return out 
    map_data=[{"X": X, "n_neighbors": n_neighbors, "resolution": resolution} for n_neighbors in n_neighbors_list for resolution in resolution_list]
    p = Parallelizer(n_jobs=min(len(map_data), os.cpu_count()))
    result = p.parallelize(
        map_func=par_func, 
        map_data=map_data, 
        reduce_func=lambda x: x,
        progress=True,
        backend="threading"
    )()
    return pd.DataFrame(
        list(map(lambda x: list(x.values())[0], result)), 
        columns=list(map(lambda x: list(x.keys())[0], result))
    )


class ParallelPairwiseCalculator:
    def __init__(self, 
        pairwise_func: Callable,
        n_jobs=None, 
        block_size: int = 50,
        backend: str = "thread",
        cutoff: float = 0.0,
        dtype = np.uint8
    ):
        self.pairwise_func = pairwise_func
        self.n_jobs = n_jobs if n_jobs else cpu_count()
        self.block_size = block_size
        self.backend = backend
        self.cutoff = cutoff
        self.DTYPE = dtype
        
    @staticmethod
    def _block_iter(
        seqs1: Sequence[str],
        seqs2: Optional[Sequence[str]] = None,
        block_size: Optional[int] = 50,
    ):
        """Iterate over sequences in blocks.

        Parameters
        ----------
        seqs1
            array containing (unique) sequences
        seqs2
            array containing other sequences. If `None` compute
            the square matrix of `seqs1` and iterate over the upper triangle (including
            the diagonal) only.
        block_size
            side length of a block (will have `block_size ** 2` elements.)

        Yields
        ------
        seqs1
            subset of length `block_size` of seqs1
        seqs2
            subset of length `block_size` of seqs2. If seqs2 is None, this will
            be `None` if the block is on the diagonal, or a subset of seqs1 otherwise.
        origin
            (row, col) coordinates of the origin of the block.
        """
        square_mat = seqs2 is None
        if square_mat:
            seqs2 = seqs1
        for row in range(0, len(seqs1), block_size):
            start_col = row if square_mat else 0
            for col in range(start_col, len(seqs2), block_size):
                if row == col and square_mat:
                    # block on the diagonal.
                    # yield None for seqs2 to indicate that we only want the upper
                    # diagonal.
                    yield seqs1[row : row + block_size], None, (row, row)
                else:
                    yield seqs1[row : row + block_size], seqs2[
                        col : col + block_size
                    ], (row, col)

    def _calculate(self, seqs1, seqs2, origin):
        origin_row, origin_col = origin
        if seqs2 is not None:
            # compute the full matrix
            coord_iterator = itertools.product(enumerate(seqs1), enumerate(seqs2))
        else:
            # compute only upper triangle in this case
            coord_iterator = itertools.combinations_with_replacement(
                enumerate(seqs1), r=2
            )

        result = []
        for (row, s1), (col, s2) in coord_iterator:
            d = self.pairwise_func(s1, s2)
            if self.cutoff < 0 or d <= self.cutoff:
                result.append((d + 1, origin_row + row, origin_col + col))

        return result

    def calculate(self, seqs: Sequence[str], seqs2: Optional[Sequence[str]] = None):
        blocks = list(self._block_iter(seqs, seqs2, self.block_size))
        if self.backend == "thread":
            block_results = thread_map(
                self._calculate,
                *zip(*blocks),
                max_workers=self.n_jobs if self.n_jobs is not None else cpu_count(),
                chunksize=50,
                tqdm_class=asyn_tqdm,
                total=len(blocks),
            )
        elif self.backend == "process":
            block_results = process_map(
                self._calculate,
                *zip(*blocks),
                max_workers=self.n_jobs if self.n_jobs is not None else cpu_count(),
                chunksize=50,
                tqdm_class=asyn_tqdm,
                total=len(blocks),
            )
        else:
            raise ValueError("Invalid backend. Must be one of 'thread' or 'process'.")
        try:
            dists, rows, cols = zip(*itertools.chain(*block_results))
        except ValueError:
            # happens when there is no match at all
            dists, rows, cols = (), (), ()

        shape = (len(seqs), len(seqs2)) if seqs2 is not None else (len(seqs), len(seqs))
        score_mat = scipy.sparse.coo_matrix(
            (dists, (rows, cols)), dtype=self.DTYPE, shape=shape
        )
        score_mat.eliminate_zeros()
        score_mat = score_mat.tocsr()

        if seqs2 is None:
            score_mat = self.squarify(score_mat)

        return score_mat

    @staticmethod
    def squarify(triangular_matrix: csr_matrix) -> csr_matrix:
        """Mirror a triangular matrix at the diagonal to make it a square matrix.

        The input matrix *must* be upper triangular to begin with, otherwise
        the results will be incorrect. No guard rails!
        """
        assert (
            triangular_matrix.shape[0] == triangular_matrix.shape[1]
        ), "needs to be square matrix"
        # The matrix is already upper diagonal. Use the transpose method, see
        # https://stackoverflow.com/a/58806735/2340703.
        return (
            triangular_matrix
            + triangular_matrix.T
            - scipy.sparse.diags(triangular_matrix.diagonal())
        )


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def joblib_parallel_map_with_progress_bar(
    par_func: Callable,
    data: Sequence,
    n_jobs: int = -1,
):
    with tqdm_joblib(
        tqdm.tqdm(
            total=len(data),
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",  # noqa
            position=0,
            leave=True
        )
    ) as pbar:
        res = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(par_func)(d) for d in data
        )
    return res