from pysparse.sparse import spmatrix
from pysparse.precon import precon
from pysparse.itsolvers import krylov
import numpy
n = 300
L = poisson2d_sym_blk(n)
b = numpy.ones(n*n)
x = numpy.empty(n*n)
info, iter, relres = krylov.pcg(L.to_sss(), b, x, 1e-12, 2000)

