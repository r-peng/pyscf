import numpy as np
from pyscf import lib
einsum = lib.einsum

def compute_energy(f0, eri, d, l): # eri in physicists notation
    e  = einsum('pr,rp',f0[0],d[0])
    e += einsum('pr,rp',f0[1],d[1])
    e += 0.5 * einsum('pqrs,rp,sq',eri[0],d[0],d[0])
    e += 0.5 * einsum('pqrs,rp,sq',eri[2],d[1],d[1])
    e +=       einsum('pqrs,rp,sq',eri[1],d[0],d[1])
    e += 0.25 * einsum('pqrs,rspq',eri[0],l[0])
    e += 0.25 * einsum('pqrs,rspq',eri[2],l[2])
    e +=        einsum('pqrs,rspq',eri[1],l[1])
    return e

def energy(f0, eri, t1, t2):
    aov, boo, bvv = compute_irred(t1, t2, order=4)
    d = propagate1(aov, boo, bvv)
    l = propagate2(t2, d, maxiter=500)
    return compute_energy(f0, eri, d, l)

