from pyscf import gto, scf, cc, lib
#from pyscf.cc import td_roccd_ft as ft
#from pyscf.cc import td_roccd_utils as utils
from kelvin import ccsd, td_ccsd, scf_system
import numpy as np
import math
np.set_printoptions(precision=4,suppress=True)

mol = gto.Mole()
#mol.atom = [['H', (0,0,0)], 
#            ['H', (1,0,0)]] 
#mol.basis = '6-31g'
r = 0.9640
theta = math.radians(109.5)
x = r * math.cos(theta)
y = r * math.sin(theta)
mol.atom = [['O', (0,0,0)], 
            ['H', (r,0,0)],
            ['H', (x,y,0)]] 
mol.symmetry = False
mol.verbose = 4
mol.build()
nmo = mol.nao_nr()

mf = scf.RHF(mol)
mf.kernel()

beta = 1.0
mu = 0.0
ngrid = 11
sys = scf_system.scf_system(mf, 1/beta, mu, orbtype='g')
#sys = scf_system.scf_system(mf, 1/beta, mu, orbtype='r')

# differential version
prop = {'tprop':'rk4', 'lprop':'rk4'}
mycc_ = td_ccsd.TDCCSD(sys, prop=prop, T=1/beta, mu=mu, iprint=1, singles=False, ngrid=ngrid, athresh=0.0, saveT=True, saveL=True)
mycc_._rccsd()
mycc_._rccsd_lambda()
print('nmo: ', nmo)
print('T2[0],[beta],[beta/2]:', np.linalg.norm(mycc_.T2[0]), np.linalg.norm(mycc_.T2[-1]), np.linalg.norm(mycc_.T2[int(ngrid/2)+1]))
print('L2[0],[beta],[beta/2]:', np.linalg.norm(mycc_.L2[0]), np.linalg.norm(mycc_.L2[-1]), np.linalg.norm(mycc_.L2[int(ngrid/2)+1]))
tab_ = mycc_.T2[int(ngrid/2)+1].copy() 
lab_ = mycc_.L2[int(ngrid/2)+1].copy()
# differential version

# integral version
mycc = ccsd.ccsd(sys, T=1/beta, mu=mu, iprint=1, singles=False, ngrid=ngrid, athresh=0.0)
Ecctot, Ecc = mycc.run()
print('Etot: {}, Ecorr: {}'.format(Ecctot, Ecc))
ng, nv, no = mycc.T1.shape
L1 = np.zeros((ng, no, nv), mycc.T1.dtype)
L2 = np.zeros((ng, no, no, nv, nv), mycc.T2.dtype)
#L1 = np.flip(mycc.T1.transpose(0,2,1), 0)
#L2 = np.flip(mycc.T2.transpose(0,3,4,1,2), 0)
mycc._ft_ccsd_lambda(L1=L1,L2=L2)
print('T2[0],[beta],[beta/2]:', np.linalg.norm(mycc.T2[0]), np.linalg.norm(mycc.T2[-1]), np.linalg.norm(mycc.T2[int(ngrid/2)+1]))
print('L2[0],[beta],[beta/2]:', np.linalg.norm(mycc.L2[0]), np.linalg.norm(mycc.L2[-1]), np.linalg.norm(mycc.L2[int(ngrid/2)+1]))
tab = mycc.T2[int(ngrid/2),:nmo,nmo:,:nmo,nmo:].copy()
lab = mycc.L2[int(ngrid/2),:nmo,nmo:,:nmo,nmo:].copy()
# integral version

print('T2: ', np.linalg.norm(tab-tab_))
print('L2: ', np.linalg.norm(lab-lab_))
print('T2 symm: ', np.linalg.norm(tab -tab.transpose(1,0,3,2)))
print('L2 symm: ', np.linalg.norm(tab_-tab_.transpose(1,0,3,2)))
print('L2 symm: ', np.linalg.norm(lab -lab.transpose(1,0,3,2)))
print('T2 symm: ', np.linalg.norm(lab_-lab_.transpose(1,0,3,2)))

#w = 1.0 
#td = 4.0
#step = 1e-5 
#tf = step * 100 
#f0 = np.array((1.0,1.0,1.0))*1e-4
#
#eris_mol = ft.ERIs_mol(mf, f0, w, td, beta, mu)
#ft.kernel(eris_mol, tab, lab, tf, step, RK=4)
#eris_p = ft.ERIs_p(mf, f0, w, td, beta, mu)
#ft.kernel(eris_p, tab, lab, tf, step)
