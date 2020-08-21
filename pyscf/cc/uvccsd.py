import numpy as np
from pyscf import lib
einsum = lib.einsum

def compute_energy(f0, eri, d, l): # eri in physicists notation
    e  = einsum('pr,rp',f0,d[0])
    e += einsum('pr,rp',f0,d[1])
    e += 0.5 * einsum('pqrs,rp,sq',eri[0],d,d)
    e -= 0.5 * einsum('pqsr,rp,sq',eri[2],d,d)
    e += 0.5 * einsum('pqrs,rspq',eri,l)
    return e

