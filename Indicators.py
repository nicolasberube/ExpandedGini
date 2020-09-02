#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:56:16 2020

@author: berube
"""
import os
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
from bisect import bisect_left, bisect_right
import pickle
from itertools import product
import scipy.sparse as sparse
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib import cm


class sparse_lowmem():
    """
    Custom low memory version of a binary sparse matrix.

    Has index files to quickly retrieve non-null element index from columns.

    Can also be used from memory maps for low-memory usage.

    Parameters
    ----------
    file_path: str or Path
        Path to the numpy memory map file of the matrix.
        The index file will be saved as Path(file_path).stem + '_map.npy'
        The hyper parameters will be saved as
        Path(file_path).stem + '_hyp.pkl'

    memmap: bool, optional
        If True, will use Numpy memory maps to access. Access will be about
        10 times slower, but RAM used will be close to nothing.
        Default is False.

    typ: str, optional
        Type of integer to describe the index elements of the matrix.
        Default is 'u4', which covers ranges from 65k to 4,2 billion.
    """

    def __init__(self,
                 file_path,
                 memmap=False,
                 typ='u4'):
        self.file_path = Path(file_path)
        self.map_path = (self.file_path.parent /
                         (self.file_path.stem + '_map.npy'))
        self.hyper_path = (self.file_path.parent /
                           (self.file_path.stem + '_hyp.pkl'))
        self.memmap = memmap
        self.typ = typ
        self.table = None
        self.map = None
        self.table_shape = None
        self.map_shape = None
        self.ref_dict = None
        typ_dict = {'u2': np.uint16,
                    'u4': np.uint32,
                    'i2': np.int16,
                    'i4': np.int32,
                    'f4': np.float32,
                    'f8': np.float64}
        self.np_typ = typ_dict[self.typ]

        if os.path.exists(self.hyper_path):
            self.table_shape, self.map_shape, self.ref_dict = \
                pickle.load(open(self.hyper_path, 'rb'))

        self.load()

    def __iter__(self,
                 chunksize=1000000):
        """Iterator that goes through all the matrix element in order

        Is equivalent to [X, self.ref(X) for X in self.map[0]], but handles
        memmaps quickly by loading chunks in RAM.

        Parameters
        ----------
        chunksize: int, optional
            Number of elements to be loaded in RAM at a time.
            Note that the size taken will be approximately 2*chucksize*self.typ
            Default is 1 000 000,
            which will load a few Mb in the RAM at a time.

        Yields
        ------
        (int, Numpy array)
            First element returned is the column integer identifier
            Second element returned are the integers of the indexes that
            have non-null boolean element in the matrix
        """
        chunksize = max(1, chunksize)
        i_map_start = 0
        i_table_start = 0

        # Loop over each chunk load in RAM
        while i_table_start < self.table.shape[0]:
            # Selecting the indexes of self.map to put in RAM
            # The self.map in RAM will contain one extra data point, since
            # self.map[1] needs that extra data to compute id_table_end
            # This extra data will also be present is self.map_shape[0] == 1
            # even though it is not needed
            if self.map_shape[0] == 1:
                i_map_end = i_table_start + chunksize + 1
                i_table_end = i_map_end - 1
            else:
                temp = np.array([i_table_start+chunksize],
                                dtype=self.typ)[0]
                i_map_end = bisect_right(self.map[1], temp)
                if i_map_end < i_map_start+2:
                    i_map_end = i_map_start+2
                if i_map_end - 1 >= self.map_shape[1]:
                    i_table_end = self.table.shape[0]
                else:
                    i_table_end = self.map[1][i_map_end - 1]

            # If this is the last chunk of the table, include the last column
            # By adding a dud extra data point to map_ram
            if i_table_end >= self.table.shape[0]:
                i_table_end = self.table.shape[0]
                map_ram = np.concatenate((
                    self.map[:, i_map_start:i_map_end],
                    np.array([[-1]]*(self.map_shape[0]-1)+[[i_table_end]],
                             dtype=self.typ)
                    ), axis=1)
            # If this is not the last chunk
            else:
                map_ram = np.array(self.map[:, i_map_start:i_map_end])

            # Putting self.table into RAM
            table_ram = np.array(self.table[i_table_start: i_table_end])

            # Iterating over all values in RAM
            for inda, id_col in enumerate(map_ram[0][:-1]):
                if map_ram.shape[0] == 1:
                    idx1 = inda
                    idx2 = inda + 1
                else:
                    idx1 = map_ram[1][inda] - i_table_start
                    idx2 = map_ram[1][inda + 1] - i_table_start
                yield id_col, table_ram[idx1: idx2]

            # Next chunk
            i_map_start = i_map_end - 1
            i_table_start = i_table_end

    def load(self):
        """Load matrices from model files
        """
        del self.map
        if os.path.exists(self.map_path) and self.map_shape:
            self.map = np.memmap(self.map_path,
                                 dtype=self.typ,
                                 mode='r',
                                 shape=self.map_shape)
            if not self.memmap:
                self.map = np.array(self.map)
        del self.table
        if os.path.exists(self.file_path) and self.table_shape:
            self.table = np.memmap(self.file_path,
                                   dtype=self.typ,
                                   mode='r',
                                   shape=self.table_shape)
            if not self.memmap:
                self.table = np.array(self.table)

    def change_memmap(self,
                      memmap):
        """Change the memmap value of the object and reloads files

        Parameters
        ----------
        memmap: bool
            The memmap value to change to
        """
        if memmap == self.memmap:
            return
        self.memmap = memmap
        self.load()

    def compute(self,
                data_path,
                header=1,
                sep='\t',
                col_right=False,
                fmax=0,
                verbose=True,
                encoding='utf-8',
                ref_dict=False,
                char_del=''):
        """
        Computes model files of the matrix based on textual data dump

        Parameters
        ----------
        data_path: str or Path
            Path to the textual data dump of the file to import
            as a matrix. Each line should be include two integer indexes
            representing a non-null element of the matrix, separated by
            a non-line-return blank character specificed with sep
            (default is tab).

        header: int, optional
            Number of header lines in the textual file. Default is 1.

        sep: str, optional
            Character to be considered separator between colums.
            Default is tab '\t'

        col_right: bool, optional
            If True, indicates that the column label, the column which will
            serve as an index of a function to return all non-null element of,
            is the second column of the textual file. If False, it will be the
            first column of the textual file.
            Default is False

        fmax: int, optional
            If non-zero, indicates the maximum amount of element (lines)
            to read from the textual file.
            Default is 0.

        verbose: bool, optional
            If True, prints progress in console. Default is True.

        encoding: str, optional
            Encoding of the file. Files generated on Windows might be
            'latin-1'.
            Default is 'utf-8'.

        ref_dict: bool, optional
            Indicated that the reference column is not an integer and
            would need a dictionary to translate the values into
            integers
            Default is False.

        char_del: str, optional
            List of characters to delete and ignore from the file, in case of
            encoding problems
        """
        data_path = Path(data_path)

        if verbose:
            print(f'{datetime.now()}\tImporting data')

        if ref_dict:
            self.ref_dict = {}
        if col_right:
            col_idx = 1
            ref_idx = 0
        else:
            col_idx = 0
            ref_idx = 1

        # Calculates size of the matrix/file
        if fmax > 0:
            flen = fmax
        if ref_dict:
            with open(data_path, encoding=encoding) as f:
                for _ in range(header):
                    _ = f.readline()
                for i, lin in enumerate(f):
                    for c in '\n'+char_del:
                        lin = lin.replace(c, '')
                    ref = lin.split(sep)[ref_idx]
                    if ref not in self.ref_dict:
                        self.ref_dict[ref] = len(self.ref_dict)
                    if fmax > 0 and i == fmax - header - 1:
                        break
            flen = i+header+1
        elif fmax == 0:
            with open(data_path, encoding=encoding) as f:
                for i, _ in enumerate(f):
                    pass
            flen = i+1

        if verbose:
            print(f'{datetime.now()}\tTable length: {flen} lines')
            time.sleep(0.5)

        self.table_shape = flen-header
        column = np.empty(self.table_shape, dtype='u4')
        table = np.memmap(self.file_path,
                          dtype=self.typ,
                          mode='w+',
                          shape=self.table_shape)
        with open(data_path, 'r', encoding=encoding) as f:
            for h in range(header):
                _ = f.readline()
            if verbose:
                lin_iter = tqdm(enumerate(f), total=self.table_shape)
            else:
                lin_iter = enumerate(f)
            for i, lin in lin_iter:
                if i == self.table_shape:
                    break
                for c in '\n'+char_del:
                    lin = lin.replace(c, '')
                ls = lin.split(sep)
                column[i] = int(ls[col_idx])
                if ref_dict:
                    table[i] = int(self.ref_dict[ls[ref_idx]])
                else:
                    table[i] = int(ls[ref_idx])

        if verbose:
            time.sleep(0.5)
            print(f'{datetime.now()}\tSorting table')
        arg_sort = np.argsort(column)
        column = column[arg_sort]
        table[:] = table[arg_sort]
        del arg_sort
        del table

        if verbose:
            print(f'{datetime.now()}\tBuilding index map')

        all_id_col = np.unique(column)
        self.map_shape = (2 - 1*(len(all_id_col) == len(column)),
                          len(all_id_col))
        pickle.dump((self.table_shape, self.map_shape, self.ref_dict),
                    open(self.hyper_path, 'wb'))
        id_map = np.memmap(self.map_path,
                           dtype=self.typ,
                           mode='w+',
                           shape=self.map_shape)
        id_map[0] = all_id_col
        del all_id_col

        if self.map_shape[0] > 1:
            id_map[1][0] = 0
            id_map[1][1:] = np.where(column[1:] !=
                                     column[:-1])[0].astype('u4')+1
        del column
        del id_map

        self.load()

        if verbose:
            print(f'{datetime.now()}\tDone')

    def ref(self,
            id_col):
        """
        Returns all non-null element index from a specific column.

        Parameters
        ----------
        id_col: int
            Column integer identifier

        Returns
        -------
        Numpy array
            Integers of the indexes that have non-null boolean element
            in the matrix
        """
        if isinstance(id_col, self.np_typ):
            id_col_corr = id_col
        else:
            id_col_corr = np.array([id_col], dtype=self.typ)[0]

        inda = bisect_left(self.map[0], id_col_corr)

        # Column not found
        if inda == self.map.shape[1] or self.map[0][inda] != id_col_corr:
            return self.table[0:0]

        if self.map.shape[0] == 1:
            idx1 = inda
            idx2 = inda+1
        else:
            idx1 = self.map[1][inda]
            if inda == self.map.shape[1]-1:
                idx2 = len(self.table)
            else:
                idx2 = self.map[1][inda+1]

        return self.table[idx1:idx2]


def cosinesim(clus_cits,
              verbose=True):
    """Computes cosine similarity matrix based on citation data

    Parameters
    ----------
    clus_cits: 2D sparse/numpy array
        The citation matrix, where the [i, j] element is the number of citation
        from class (or cluster) index i to class index j.
        Can hanlde numpy and sparse_cscmatrix format.

    verbose: bool, optional
        If True, prints progress in console. Default is True.

    Returns
    -------
    2D sparse.csc_matrix
        The cosine similarity between the classes.
        The matrix is symetrical, its diagonal should be 1.
        However, for storage purpose, the diagonal element has been removed,
        as well as the lower part, i.e. A[i, j<=i] = 0
        The cosine was calculated according to
        corresponding to the formula of Zhang-Rousseau-Glanzel,
        Diversity of References as an Indicator of the Interdisciplinarity
        of Journals: Taking Similarity Between Subject Fields Into Account.
    """
    clus_sums = np.sqrt(np.array(clus_cits.sum(axis=0) +
                                 clus_cits.sum(axis=1).T -
                                 2*clus_cits.diagonal()).reshape(-1))

    # If sparse
    if isinstance(clus_cits, sparse.csc_matrix):
        if verbose:
            print(f'{datetime.now()}\tConverting to numpy array')
        clus_cos = clus_cits.toarray()
        if verbose:
            print(f'{datetime.now()}\tComputing')
            time.sleep(0.5)
    clus_cos = clus_cos.astype('f8')
    clus_cos = clus_cos + clus_cos.T
    it = range(clus_cos.shape[0])
    if verbose:
        it = tqdm(it)
    for i in it:
        clus_cos[i, :i+1] = 0
        if clus_sums[i] != 0:
            clus_cos[i] /= clus_sums[i]
            clus_cos[:, i] /= clus_sums[i]
        else:
            clus_cos[i] = 0
            clus_cos[:, i] = 0

    return sparse.csc_matrix(clus_cos)


def newmatrix(clus_cits,
              verbose=True):
    """Computes expanded gini similarity matrix based on citation data

    Parameters
    ----------
    clus_cits: 2D sparse/numpy array
        The citation matrix, where the [i, j] element is the number of citation
        from class (or cluster) index i to class index j.
        Can hanlde numpy and sparse_cscmatrix format.

    verbose: bool, optional
        If True, prints progress in console. Default is True.

    Returns
    -------
    2D Numpy array
        The (non-symetrical) matrix to calculate the expanded gini coefficient
    """
    clus_sums = -2*np.array(clus_cits.sum(axis=1)).reshape(-1)

    if isinstance(clus_cits, sparse.csc_matrix):
        if verbose:
            print(f'{datetime.now()}\tConverting to numpy array')
        clus_nmat = clus_cits.toarray()
        if verbose:
            print(f'{datetime.now()}\tComputing')
            time.sleep(0.5)

    clus_nmat = clus_nmat.astype('f8')
    it = range(clus_nmat.shape[0])
    if verbose:
        it = tqdm(it)
    for i in it:
        clus_nmat[i, i] = 0.
        if clus_sums[i] != 0:
            clus_nmat[i] /= clus_sums[i]
        else:
            clus_nmat[i] = 0

    clus_sums = -2*np.array(clus_nmat.sum(axis=0)).reshape(-1)

    for i in range(clus_nmat.shape[0]):
        clus_nmat[i, i] = max(1, clus_sums[i])

    return sparse.csc_matrix(clus_nmat)


class cluster_class():
    """
    Class that contain info about the cluster

    Parameters
    ----------
    name: str
        Name of the cluster. Will be tied to identifying the model files

    path: str or Path, optional
        Paths to the Id_Art/cluster_id textual data dump (tab-separated with
        1-line header).
        They are only used if the model files corresponding to the
        clusters are not present and need to be recomputed.
        Default is None.

    char_del: str, optional
        Parameter to the sparse_lowmem.compute() function when computing
        the model files.
        List of characters to delete and ignore from the file, in case of
        encoding problems.
        Default is the empty string ''

    mat: sparse_lowmem() object, optional
        The sparse_lowmem() object containing the data of the cluster.
        Default is None
    """

    def __init__(self,
                 name,
                 path=None,
                 char_del='',
                 mat=None):
        self.name = name
        self.path = path
        self.char_del = ''
        self.mat = mat


class interdisc():
    """
    Group of objects and methods to calculate interdisciplinarity metrics

    Parameters
    ----------
    clusters: list of cluster_class()
        List of the clusters to be used with their necessary
        attributes identified in the cluster_class() object.

    cits_path: str or Path, optional
        Path to Id_Art_Cite/Id_Art_Citant textual data dump
        (tab separated with 1-line header) from WOS.
        Is only necessary if the model files don't already exist
        in the model_path directory and need to be recomputed.
        All model files will be saved in Path(root_folder) / 'models' /
        Default is None.

    model_folder: str or Path, optional
        Path fo put the model files. Will need to contains 15 Gb of files.
        Default is None, which will put them in the ./models/ repository.

    Attributes
    ----------
    citations
    irefs
    icoss
    inews
    artdatas
    """

    def __init__(self,
                 clusters,
                 cits_path=None,
                 model_folder=None):

        if model_folder is None:
            self.model_folder = Path().absolute() / 'models'
        else:
            self.model_folder = Path(model_folder)
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)

        self.clusters = clusters
        self.cits_paths = cits_path
        self.citations = None
        self.irefs = [None for _ in range(len(clusters))]
        self.icoss = [None for _ in range(len(clusters))]
        self.inews = [None for _ in range(len(clusters))]
        self.artdatas = [None for _ in range(len(clusters))]
        self.indicators = []

    def import_data(self):
        """
        Computes the model files.

        Takes approximately 50 minutes to compute the model files.
        """
        print(f'\n{datetime.now()}\tImporting files')

        # flen = 689900913
        self.citations = sparse_lowmem('models/citations.npy',
                                       memmap=True)
        if self.citations.table is None or self.citations.map is None:
            if self.cits_path is None or not os.path.exists(self.cits_path):
                raise NameError('Path to citations file not found')
            print(f'{datetime.now()}\tComputing citations matrix')
            self.citations.compute('/Volumes/Rafiki/pca/citations.txt',
                                   col_right=True)

        for cluster in self.clusters:
            cluster.mat = sparse_lowmem(f'models/clusters_{cluster.name}.npy',
                                        memmap=True)
            if cluster.mat.table is None or cluster.mat.map is None:
                if (cluster.path is None or
                        not os.path.exists(cluster.path)):
                    raise NameError('Path to cluster file not found')
                print(f'{datetime.now()}\tComputing '
                      f'{cluster.name} clusters matrix')
                cluster.mat.compute(cluster.path,
                                    ref_dict=True,
                                    char_del=cluster.char_del)

    def import_irefs(self,
                     quick=True):
        """Imports/computes number of references from one cluster to another

        Parameters
        ----------
        quick: bool, optional
            If True, will load the RAM (approximately 3 Gb) instead
            of work from numpy memmaps. Will make the calculation
            approximately twice as fast (if the numpy are stored on SSD).
            Default is True.
        """
        print(f'{datetime.now()}\tImporting clusters references')

        self.irefs = [None for _ in range(len(self.clusters))]

        model_files = os.listdir(self.model_folder)
        list_iclus = []

        for i, cluster in enumerate(self.clusters):
            iref_path = f'irefs_{cluster.name}.npz'
            if iref_path in model_files:
                self.irefs[i] = sparse.load_npz(self.model_folder /
                                                iref_path)
            else:
                list_iclus.append(i)
                if cluster.mat.map_shape[0] == 1:
                    typ = 'u4'
                else:
                    typ = 'f8'
                n_cluster = len(cluster.mat.ref_dict)
                if n_cluster > 1000:
                    self.irefs[i] = sparse.lil_matrix((n_cluster, n_cluster),
                                                      dtype=typ)
                else:
                    self.irefs[i] = np.zeros((n_cluster, n_cluster),
                                             dtype=typ)
        if not list_iclus:
            return

        print(f'{datetime.now()}\tComputing clusters cosine similarities')
        time.sleep(0.5)

        # For quicker computation, loading clusters in RAM
        if quick:
            self.citations.change_memmap(False)
            for cluster in self.clusters:
                cluster.mat.change_memmap(False)

        # This could be paralelized. It lasts about 10h.
        for id_art in tqdm(self.citations.map[0]):
            idxs1_list = [self.clusters[j].mat.ref(id_art)
                          for j in list_iclus]
            for id_ref in self.citations.ref(id_art):
                for j in list_iclus:
                    idxs1 = idxs1_list[j]
                    idxs2 = self.clusters[j].mat.ref(id_ref)
                    if idxs1.size and idxs2.size:
                        if idxs1.size == 1 and idxs2.size == 1:
                            val = 1
                        else:
                            val = 1/(len(idxs1)*len(idxs2))
                        # Yes, I swear this is for loop more efficient because
                        # the idxs arrays are small here
                        for i1, i2 in product(idxs1, idxs2):
                            self.irefs[j][i1, i2] += val

        for i, mat in enumerate(self.irefs):
            self.irefs[i] = sparse.csc_matrix(mat)
            sparse.save_npz((self.model_folder /
                             f'irefs_{self.clusters[i].name}.npz'),
                            self.irefs[i])

        if quick:
            self.citations.change_memmap(True)
            for cluster in self.clusters:
                cluster.mat.change_memmap(True)

    def import_similarity(self):
        """Imports/Computes similarity matrices icoss and inews based on irefs
        """
        print(f'{datetime.now()}\tImporting clusters cosine similarities')

        self.icoss = [None for _ in range(len(self.clusters))]
        self.inews = [None for _ in range(len(self.clusters))]

        model_files = os.listdir(self.model_folder)

        for i, iref in enumerate(self.irefs):
            icoss_path = f'icoss_{self.clusters[i].name}.npz'
            if icoss_path in model_files:
                self.icoss[i] = sparse.load_npz(self.model_folder /
                                                icoss_path)
            else:
                time.sleep(0.5)
                print(f'{datetime.now()}\tComputing '
                      f'{self.clusters[i].name} cosine')
                time.sleep(0.5)
                self.icoss[i] = cosinesim(iref, verbose=True)
                sparse.save_npz((self.model_folder /
                                 icoss_path),
                                self.icoss[i])

            inews_path = f'inews_{self.clusters[i].name}.npz'
            if inews_path in model_files:
                self.inews[i] = sparse.load_npz(self.model_folder /
                                                inews_path)
            else:
                time.sleep(0.5)
                print(f'{datetime.now()}\tComputing '
                      f'{self.clusters[i].name} new similarity matrix')
                time.sleep(0.5)
                self.inews[i] = newmatrix(iref, verbose=True)
                sparse.save_npz((self.model_folder /
                                 inews_path),
                                self.inews[i])

    def import_articles(self,
                        parallel=True,
                        quick=True):
        """Imports/Computes diversity indexes for all articles

        Parameters
        ----------
        parallel: bool, optional
            If True, will run the code in parallel. Default is True

        quick: bool, optional
            If True, will load the RAM (approximately 8 Gb) instead
            of work from numpy memmaps. Will make the calculation
            approximately twice as fast (if the numpy are stored on SSD).
            Default is True.
        """
        print(f'{datetime.now()}\tImporting article interdisciplinarity data')
        time.sleep(0.5)

        # List of (Name, function) of each indicator to consider
        # in self.all_ind
        self.indicators = ['variety',
                           'gini',
                           'shannon',
                           'simpson',
                           'avgdis',
                           'rao',
                           'lcdiv1',
                           'lcdiv2',
                           'newdiv']

        self.artdatas = [None for _ in range(len(self.clusters))]

        model_files = os.listdir(self.model_folder)

        list_iclus = []
        for i, cluster in enumerate(self.clusters):
            artdata_path = f'artdata_{cluster.name}.npy'
            if artdata_path in model_files:
                self.artdatas[i] = \
                    np.memmap((self.model_folder / artdata_path),
                              dtype='f8',
                              mode='r',
                              shape=(len(self.citations.map[0]),
                                     len(self.indicators)))
            else:
                list_iclus.append(i)
        if not list_iclus:
            return

        # For quicker computation, loading clusters in RAM
        if quick:
            # self.citations.change_memmap(False)
            for i, cluster in enumerate(self.clusters):
                cluster.mat.change_memmap(False)
                self.icoss[i] = self.icoss[i].toarray()
                self.icoss[i] = self.icoss[i] + self.icoss[i].T
                np.fill_diagonal(self.icoss[i], 1)
                self.inews[i] = self.inews[i].toarray()

        self.artdatas = [np.memmap((self.model_folder /
                                    f'artdata_{cluster.name}.npy'),
                                   dtype='f8',
                                   mode='w+',
                                   shape=(len(self.citations.map[0]),
                                          len(self.indicators)))
                         for cluster in self.clusters]

        # This could be paralelized. It lasts about 15h.
        for row_i, (id_art, id_refs) in \
                tqdm(enumerate(self.citations),
                     total=self.citations.map_shape[1]):
            for clus_i in list_iclus:
                cluster = self.clusters[clus_i]
                # Calculating the proportion of each clusters in
                # the article's references, where the sum
                # of proportions is 1.
                # classprop = {cluster_id: proportion}
                classprop = {}
                for id_ref in id_refs:
                    ref_clus = cluster.mat.ref(id_ref)
                    if ref_clus.size:
                        val = 1/len(ref_clus)/len(id_refs)
                    for i_clus in ref_clus:
                        if i_clus not in classprop:
                            classprop[i_clus] = val
                        else:
                            classprop[i_clus] += val

                # All indicators
                self.artdatas[clus_i][row_i] = \
                    self.all_ind(classprop, clus_i)

        if quick:
            # self.citations.change_memmap(True)
            for i, cluster in enumerate(self.clusters):
                cluster.mat.change_memmap(True)
                del self.icoss[0]
                del self.inews[0]
                # np.fill_diagonal(self.icoss[i], 0)
                # for j in range(self.icoss[i].shape[0]):
                #     self.icoss[i][j,:j+1] = 0
                # self.icoss[i] = sparse.csc_matrix(self.icoss[i])
                # self.inews[i] = sparse.csc_matrix(self.inews[i])
            self.import_similarity()

    def all_ind(self,
                classprop,
                clus_i):
        """Computes all diversity indicators for a specific article

        The diversity indicators are hardcoded in this function.
        Their names are identified in self.indicators

        Parameters
        ----------
        classprop: dict of {int: float}
            Dictionary where the key is the integer identifying the
            class/cluster, and the value is the proportion of the element
            belonging to that class.
            It should sum to 1.

        clus_i: int
            int identifying the type of clustering used, corresponding
            to their order in self.clusters

        Returns
        -------
        tuple of (float)*9
            The 9 diversity indicator for the element, in order:
            - Variety
            - Gini coefficient
            - Average disparity
            - Simpson
            - Shannon
            - Rao-Stirling
            - Leinster-Cobbold (q=1)
            - Leinster-Cobbold (q=2)
            - New Expanded Gini
        """
        if len(classprop) == 0:
            return (-1, -1, -1, -1, -1, -1, -1, -1, -1)

        classes = sorted(list(classprop.items()), key=lambda x: -x[1])
        values = np.array([c[1] for c in classes])
        iclasses = np.array([c[0] for c in classes])
        diff = [classes[i-1][1] - classes[i][1]
                for i in range(1, len(classes))] + [classes[-1][1]]
        n = len(classprop)

        # Variety
        variety = n

        # Gini
        n_total = len(self.clusters[clus_i].mat.ref_dict)
        gini = 0
        for i in range(n):
            gini += (2*(i+n_total-n)-n_total+1)*values[-(i+1)]
        gini = 1-gini/n_total

        # Simpson
        simpson = 1-(values**2).sum()

        # Shannon
        shannon = -(values*np.log(values)).sum()

        # Disparity indicators
        # (average disparity, rao-stirling, leinster-cobbold with q=1 and q=2)
        cos_sims = self.icoss[clus_i][iclasses.reshape(-1, 1),
                                      iclasses.reshape(1, -1)]
        if len(classprop) == 1:
            avgdis = 1
        else:
            avgdis = 1 - (cos_sims.sum() - n)/(n*(n-1))
        rao = values@(1-cos_sims)@values
        lcdiv1 = np.power(values@cos_sims, -values).prod()
        lcdiv2 = 1/(1-rao)

        # New expanded gini
        newdiv = 0
        sub_newdiv = 0
        for i in range(len(classprop)):
            m_ind = iclasses[:i+1]
            sub_newdiv += (self.inews[clus_i][m_ind, m_ind[-1]].sum() +
                           self.inews[clus_i][m_ind[-1], m_ind].sum() -
                           self.inews[clus_i][m_ind[-1], m_ind[-1]])
            newdiv += ((i+1) * diff[i] * sub_newdiv)

        return (variety,
                gini,
                avgdis,
                simpson,
                shannon,
                rao,
                lcdiv1,
                lcdiv2,
                newdiv)

    def compute_spearman(self):
        """Computes spearman coefficient between all indicators

        The coefficient is calculated on the diversity for all articles
        """
        print(f'{datetime.now()}\tImporting spearman correlation data')
        time.sleep(0.5)

        model_files = os.listdir(self.model_folder)
        if ('spearman.npy' in model_files and
                'spearman_pvalue.npy' in model_files):
            self.spearman = np.load(self.model_folder /
                                    'spearman.npy')
            self.spearman_pvalue = np.load(self.model_folder /
                                           'spearman_pvalue.npy')
            return

        dim = len(self.indicators) * len(self.clusters)
        self.spearman = np.empty((dim, dim), dtype='f8')
        self.spearman_pvalue = np.empty((dim, dim), dtype='f8')

        # Lasts about 4h of computation
        with tqdm(total=dim*(dim-1)//2) as pbar:
            for i_clus, i_ind, j_clus, j_ind in \
                    product(range(len(self.clusters)),
                            range(len(self.indicators)),
                            range(len(self.clusters)),
                            range(len(self.indicators))):
                i_mat = i_clus*len(self.indicators) + i_ind
                j_mat = j_clus*len(self.indicators) + j_ind
                if i_mat == j_mat:
                    self.spearman[i_mat, j_mat] = 1.
                    self.spearman_pvalue[i_mat, j_mat] = 0.
                elif i_mat > j_mat:
                    data = np.array([self.artdatas[i_clus][:, i_ind],
                                     self.artdatas[j_clus][:, j_ind]])
                    data = data[:, ~np.logical_or(data[0] < 0, data[1] < 0)]
                    sp, pv = spearmanr(data[0], data[1])
                    self.spearman[i_mat, j_mat] = sp
                    self.spearman_pvalue[i_mat, j_mat] = pv
                    self.spearman[j_mat, i_mat] = sp
                    self.spearman_pvalue[j_mat, i_mat] = pv
                    pbar.update()

        np.save(self.model_folder / 'spearman.npy',
                self.spearman)
        np.save(self.model_folder / 'spearman_pvalue.npy',
                self.spearman_pvalue)

    def plotpca(self,
                clus_i=None,
                ind_idxs=None,
                d=2):
        """Runs principal component analysis on spearman coefficients.

        Saves the plot in the local code respository as .png

        Parameters
        ----------
        clus_i: int, optional
            Iindex of the cluster to keep (in the list self.clusters),
            if you only want to print a single cluster.
            If None, prints all clusters
            Default is None

        ind_idxs: list of int, optional
            Index of the indicators to keep in the plot.
            (in the list self.indicators)
            If None, prints all indicators.
            Default is None

        d: int, optional
            Number of dimensions to keep in the PCA
        """

        fig = plt.figure(constrained_layout=True)
        widths = [1]
        heights = [5, 1]
        if clus_i is None:
            heights += [1]
        spec = fig.add_gridspec(ncols=len(widths),
                                nrows=len(heights),
                                width_ratios=widths,
                                height_ratios=heights)

        # Using scikit learn library instead
        # from sklearn.decomposition import PCA
        # p = PCA(n_components=2)
        # p.fit_transform(mat)

        # Home-made PCA
        if ind_idxs is None:
            ind_idxs = range(len(self.indicators))
        if clus_i is not None:
            idx_keep = [len(self.indicators)*clus_i + i
                        for i in ind_idxs]
        else:
            idx_keep = [len(ind_idxs)*clus_i + i
                        for clus_i in range(len(self.clusters))
                        for i in ind_idxs]

        mat = self.spearman[idx_keep][:, idx_keep]
        cov_mat = np.cov(mat)
        eig_val, eig_vec = np.linalg.eigh(cov_mat)
        eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i])
                     for i in range(len(eig_val))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        matrix_w = np.empty((d, len(idx_keep)))

        # pvars is the cumulative sum of the percentage of the variance
        # represented with regards to the number of components used
        pvars = np.empty(len(idx_keep))
        for i in range(len(eig_pairs)):
            pvars[i] = eig_pairs[i][0]/eig_val.sum()
        pvars = pvars.cumsum()

        # pca_vec is the representation of the vectors in reduced dimension
        for i in range(d):
            matrix_w[i] = eig_pairs[i][1]
        sp_cen = (mat-mat.mean(axis=0)).T
        pca_vec = np.dot(matrix_w, sp_cen)

        # PCA plot
        ax_pca = fig.add_subplot(spec[0, 0])

        ind_colors = [cm.tab10(i/10) for i in ind_idxs]
        clus_markers = ['o', 'x', '^', '+', 's', '*', '1', 'D', 'X', '_']
        clus_full = [0, 1, 0, 1, 0, 0, 1, 0, 1, 1]

        if clus_i is None:
            clusters = self.clusters
        else:
            clusters = [self.clusters[clus_i]]
        for ci in range(len(clusters)):
            fillcolors = [(r, g, b, clus_full[ci])
                          for (r, g, b, a) in ind_colors]
            edgecolors = [(r, g, b, 1-clus_full[ci])
                          for (r, g, b, a) in ind_colors]
            ax_pca.scatter(pca_vec[0][ci*len(ind_idxs):
                                      (ci+1)*len(ind_idxs)],
                           pca_vec[1][ci*len(ind_idxs):
                                      (ci+1)*len(ind_idxs)],
                           s=100,
                           c=fillcolors,
                           edgecolors=edgecolors,
                           marker=clus_markers[ci])
        added_title = ''
        if clus_i is not None:
            added_title = f'({clusters[0].name}) '
        ax_pca.set_title('PCA of indicators Spearman similarities ' +
                         added_title +
                         f'({100*pvars[d-1]:.1f}% of variance)')

        # Legend plot
        # Indicators
        ax_ind = fig.add_subplot(spec[1, 0])
        ax_ind.set_axis_off()
        ax_ind.axis([0, 1, 0, 1])
        for i, ind_idx in enumerate(ind_idxs):
            ind_name = self.indicators[ind_idx]
            ax_ind.scatter((i+0.5)/len(ind_idxs),
                           0.75,
                           color=ind_colors[i],
                           s=100,
                           marker='.')
            ax_ind.text((i+0.5)/len(ind_idxs),
                        0,
                        ind_name[0].upper()+ind_name[1:],
                        horizontalalignment='center',
                        color=ind_colors[i],
                        fontsize=9)
        added_title = ''
        if clus_i is None:
            added_title = ' and clusters'
        ax_ind.set_title('Legend: indicators' + added_title)

        # Clusters
        if clus_i is None:
            ax_clus = fig.add_subplot(spec[2, 0])
            ax_clus.set_axis_off()
            ax_clus.axis([0, 1, 0, 1])
            for i, clus in enumerate(self.clusters):
                fc = [(0, 0, 0, 0), (0, 0, 0, 1)][clus_full[i]]
                ec = [(0, 0, 0, 1), (0, 0, 0, 0)][clus_full[i]]
                ax_clus.scatter((i+0.5)/len(self.clusters),
                                0.75,
                                color=fc,
                                edgecolors=ec,
                                s=100,
                                marker=clus_markers[i])
                ax_clus.text((i+0.5)/len(self.clusters),
                             0,
                             clus.name,
                             horizontalalignment='center',
                             color='k',
                             fontsize=10)
        added_title = '_all'
        if clus_i is not None:
            added_title = '_'+clusters[0].name
        plt.savefig('PCA'+added_title,
                    dpi=600,
                    bbox_inches='tight')
        # plt.show()

    def import_all(self):
        """Imports every model files, or computes them if needed be
        """
        self.import_data()
        self.import_irefs()
        self.import_similarity()
        self.import_articles()
        self.compute_spearman()


if __name__ == '__main__':
    cits_path = '/Volumes/Rafiki/pca/citations.txt'
    clusters = [cluster_class('CWTS',
                              path='/Volumes/Rafiki/pca/clusters_CWTS.txt'),
                cluster_class('JOU',
                              path='/Volumes/Rafiki/pca/clusters_JOU.txt'),
                cluster_class('NSF',
                              path='/Volumes/Rafiki/pca/clusters_NSF.txt'),
                cluster_class('WOS',
                              path='/Volumes/Rafiki/pca/clusters_WOS.txt',
                              char_del="'")]
    diversity = interdisc(clusters,
                          cits_path=cits_path)
    diversity.import_all()

    ind_idxs = [0, 1, 2, 3, 4, 5, 6, 8]
    diversity.plotpca(ind_idxs=ind_idxs)
    diversity.plotpca(clus_i=0, ind_idxs=ind_idxs)
    diversity.plotpca(clus_i=1, ind_idxs=ind_idxs)
    diversity.plotpca(clus_i=2, ind_idxs=ind_idxs)
    diversity.plotpca(clus_i=3, ind_idxs=ind_idxs)

    """
    # Example of shared memory multiprocessing code to implement
    import multiprocessing as mp

    class cl():
        def __init__(self,
                     N):
            global M
            M = mp.Array('d', np.random.rand(N*N), lock=False)
            self.N = N

        def test(self,
                 i):
            s = 0
            for j in M[self.N*i: self.N*(i+1)]:
                s += j
            return s, i

        def multi(self,
                  cz):
            W = np.memmap((Path('models') / 'adel.npy'),
                          dtype='f8',
                          mode='w+',
                          shape=(self.N))
            n_pool = mp.cpu_count()
            it = range(self.N)
            with mp.Pool(n_pool) as pool, tqdm(total=len(it)) as pbar:
                for s, i in pool.imap_unordered(self.test, it, cz):
                    pbar.update()
                    W[i] = s

        def serial(self):
            W = np.memmap((Path('models') / 'adel.npy'),
                          dtype='f8',
                          mode='w+',
                          shape=(self.N))
            it = range(self.N)
            for i in tqdm(it):
                W[i] = self.test(i)[0]

    CL = cl(5000)
    CL.serial()
    CL.multi(1)
    CL.multi(10)
    CL.multi(100)
    """



    """
    # Test new indicator functions
    id_art = 51848625
    id_refs = np.array(I.citations.ref(id_art))
    I.clusters[3].mat.change_memmap(False)
    I.icoss[3] = I.icoss[3].toarray()
    I.icoss[3] = I.icoss[3] + I.icoss[3].T
    np.fill_diagonal(I.icoss[3], 1)
    I.inews[3] = I.inews[3].toarray()

    classprop = {}
    for id_ref in id_refs:
        ref_clus = I.clusters[3].mat.ref(id_ref)
        if ref_clus.size:
            val = 1/len(ref_clus)/len(id_refs)
        for i_clus in ref_clus:
            if i_clus not in classprop:
                classprop[i_clus] = val
            else:
                classprop[i_clus] += val

    res1 = I.all_ind(classprop, 3)
    res2 = (I.variety(classprop, 3),
            I.gini(classprop, 3),
            I.avgdis(classprop, 3),
            I.simpson(classprop, 3),
            I.shannon(classprop, 3),
            I.raostirling(classprop, 3),
            I.lcdiv(classprop, 3, 1),
            I.lcdiv(classprop, 3, 2),
            I.newdiv(classprop, 3))
    for i in range(len(res1)):
        print(res1[i]-res2[i], res1[i], res2[i])
    """
    



    """
    # Debugging test
    ftable = np.empty(flen-1, dtype={'names': ['id_citant','id_cite'], 'formats': ['u4','u4']})
    time.sleep(0.5)
    with open(data_path, 'r', encoding='utf-8') as f:
        _ = f.readline()
        for i, lin in tqdm(enumerate(f), total=flen-1):
            if i == flen-1:
                break
            ls = lin.replace('\n', '').split('\t')
            ftable[i-1] = (int(ls[1]),int(ls[0]))
    time.sleep(0.5)
    print('TEST')
    for id_class in tqdm(np.unique(ftable['id_citant'])):
        A = np.sort(citations.ref(id_class))
        B = np.sort(ftable['id_cite'][ftable['id_citant'] == id_class])
        if (A!=B).any():
            print(id_class)
            print(A)
            print(B)
            break
        pass
    """
