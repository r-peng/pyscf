import numpy as np
from pyscf import lib
einsum = lib.einsum

def compute_energy(f0, eri, d, l): # eri in physicists notation
    e  = 2.0 * einsum('pr,rp',f0,d)
    e += 2.0 * einsum('pqrs,rp,sq',eri,d,d)
    e -=       einsum('pqsr,rp,sq',eri,d,d)
    e += 2.0 * einsum('pqrs,rspq',eri,l)
    e -=       einsum('pqsr,rspq',eri,l)
    return e

def energy(f0, eri, t1, t2):
    aov, boo, bvv = compute_irred(t1, t2, order=4)
    d = propagate1(aov, boo, bvv)
    l = propagate2(t2, d, maxiter=500)
    return compute_energy(f0, eri, d, l)

