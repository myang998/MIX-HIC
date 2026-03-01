
# Read information from the hic header

import struct
import io
import os
import hicstraw


def csr_contact_matrix(norm, hicfile, chr1loc, chr2loc, unit,
                       binsize, data_type='oe', clamp=True, rescale=False):
    '''
    Extract the contact matrix from .hic in CSR sparse format
    data_type: oe, observed
    '''

    from scipy.sparse import csr_matrix, triu
    import numpy as np
    # import straw
    #
    # hic_file = hicstraw.HiCFile(hicfile)
    #
    # print(hic_file.getChromosomes())
    # print(hic_file.getGenomeID())
    # print(hic_file.getResolutions())

    tri_list = hicstraw.straw(data_type, norm, hicfile, chr1loc, chr2loc, unit, binsize)
    row, col, value = [], [], []
    for r in tri_list:
        x = int(r.binX) // int(binsize)
        row.append(x)
        y = int(r.binY) // int(binsize)
        col.append(y)
        v = r.counts
        value.append(v)


    N = max(col) + 1
    # re-scale KR matrix to ICE-matrix range
    M = csr_matrix((value, (row, col)), shape=(N, N))

    # 统计降采样前的 counts
    X_raw_upper_triangle_sparse = triu(M)
    counts_before_downsample = X_raw_upper_triangle_sparse.sum()
    print(f"总交互数: {int(counts_before_downsample)}")

    if clamp:
        M_coo = M.tocoo()
        M_coo.data = np.minimum(M_coo.data, 255)
        max_val = np.max(M_coo.data)
        if max_val > 0:  # 避免除零
            M_coo.data = M_coo.data / max_val
        M = M_coo.tocsr()
    M = M + triu(M, k=1).T
    return M


def get_hic_chromosomes(hicfile, res):

    hic_info = read_hic_header(hicfile)
    chromosomes = []
    # handle with inconsistency between .hic header and matrix data
    for c, Len in hic_info['chromsizes'].items():

        # loc = '{0}:{1}:{2}'.format(c, 0, min(Len, 100000))
        # _ = hicstraw.straw('NONE', hicfile, loc, loc, 'BP', res)
        chromosomes.append(c)


    return chromosomes


def find_chrom_pre(chromlabels):

    ini = chromlabels[0]
    if ini.startswith('chr'):
        return 'chr'

    else:
        return ''


def readcstr(f):
    buf = ""
    while True:
        b = f.read(1)
        b = b.decode('utf-8', 'backslashreplace')
        if b is None or b == '\0':
            return str(buf)
        else:
            buf = buf + b


def read_hic_header(hicfile):

    if not os.path.exists(hicfile):
        return None  # probably a cool URI

    req = open(hicfile, 'rb')
    # print(req.read(3))
    magic_string = struct.unpack('<3s', req.read(3))[0]
    # print(magic_string)
    req.read(1)
    if (magic_string != b"HIC"):
        return None  # this is not a valid .hic file

    info = {}
    version = struct.unpack('<i', req.read(4))[0]
    info['version'] = str(version)

    masterindex = struct.unpack('<q', req.read(8))[0]
    info['Master index'] = str(masterindex)

    genome = ""
    c = req.read(1).decode("utf-8")
    while (c != '\0'):
        genome += c
        c = req.read(1).decode("utf-8")
    info['Genome ID'] = str(genome)

    nattributes = struct.unpack('<i', req.read(4))[0]
    attrs = {}
    for i in range(nattributes):
        key = readcstr(req)
        value = readcstr(req)
        attrs[key] = value
    info['Attributes'] = attrs

    nChrs = struct.unpack('<i', req.read(4))[0]
    chromsizes = {}
    for i in range(nChrs):
        name = readcstr(req)
        length = struct.unpack('<i', req.read(4))[0]
        if name != 'ALL':
            chromsizes[name] = length

    info['chromsizes'] = chromsizes

    info['Base pair-delimited resolutions'] = []
    nBpRes = struct.unpack('<i', req.read(4))[0]
    for i in range(nBpRes):
        res = struct.unpack('<i', req.read(4))[0]
        info['Base pair-delimited resolutions'].append(res)

    info['Fragment-delimited resolutions'] = []
    nFrag = struct.unpack('<i', req.read(4))[0]
    for i in range(nFrag):
        res = struct.unpack('<i', req.read(4))[0]
        info['Fragment-delimited resolutions'].append(res)

    return info
