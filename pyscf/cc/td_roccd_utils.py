import numpy as np
import math, scipy
from pyscf import lib, ao2mo
einsum = lib.einsum

def update_amps(t, l, eris, time=None, X=None):
    eris.make_tensors(time)
    oovv = eris.oovv
    oooo = eris.oooo
    vvvv = eris.vvvv
    ovvo = eris.ovvo
    ovov = eris.ovov
    oovv_ = oovv - oovv.transpose(0,1,3,2)
    ovvo_ = ovvo - ovov.transpose(0,1,3,2)
    t_ = t - t.transpose(0,1,3,2)
    l_ = l - l.transpose(0,1,3,2)

    if X is None:
        Xoo = np.zeros_like(eris.foo)
        Xvv = np.zeros_like(eris.fvv)
    else: 
        Xoo, Xvv = X

    Foo  = eris.foo.copy() - 1j*Xoo
    Foo += 0.5 * einsum('klcd,cdjl->kj',oovv_,t_)
    Foo +=       einsum('klcd,cdjl->kj',oovv ,t )
    Fvv  = eris.fvv.copy() - 1j*Xvv
    Fvv -= 0.5 * einsum('klcd,bdkl->bc',oovv_,t_)
    Fvv -=       einsum('klcd,bdkl->bc',oovv ,t )

    T  = oovv.transpose(2,3,0,1).conj().copy()
    T += einsum('bc,acij->abij',Fvv,t)
    T += einsum('ac,cbij->abij',Fvv,t)
    T -= einsum('kj,abik->abij',Foo,t)
    T -= einsum('ki,abkj->abij',Foo,t)

    L  = oovv.copy()
    L += einsum('cb,ijac->ijab',Fvv,l)
    L += einsum('ca,ijcb->ijab',Fvv,l)
    L -= einsum('jk,ikab->ijab',Foo,l)
    L -= einsum('ik,kjab->ijab',Foo,l)

    loooo  = oooo.copy()
    loooo += einsum('klcd,cdij->klij',oovv,t)
    lvvvv  = vvvv.copy()
    lvvvv += einsum('klab,cdkl->cdab',oovv,t)

    T += einsum('abcd,cdij->abij',vvvv ,t)
    T += einsum('klij,abkl->abij',loooo,t)

    L += einsum('cdab,ijcd->ijab',lvvvv,l)
    L += einsum('ijkl,klab->ijab',loooo,l)

    rovvo_  = einsum('klcd,bdjl->kbcj',oovv_,t_)
    rovvo_ += einsum('klcd,bdjl->kbcj',oovv ,t )
    rovvo   = einsum('lkdc,bdjl->kbcj',oovv ,t_)
    rovvo  += einsum('klcd,bdjl->kbcj',oovv_,t )
    rovov   = einsum('kldc,dbil->kbic',oovv ,t )

    tmp  = einsum('kbcj,acik->abij',ovvo +0.5*rovvo ,t_)
    tmp += einsum('kbcj,acik->abij',ovvo_+0.5*rovvo_,t )
    tmp -= einsum('kbic,ackj->abij',ovov -0.5*rovov ,t )
    tmp += tmp.transpose(1,0,3,2)
    T += tmp.copy()

    tmp  = einsum('jcbk,ikac->ijab',ovvo +rovvo ,l_)
    tmp += einsum('jcbk,ikac->ijab',ovvo_+rovvo_,l )
    tmp -= einsum('ickb,kjac->ijab',ovov -rovov ,l )
    tmp += tmp.transpose(1,0,3,2)
    L += tmp.copy()

    Foo  = 0.5 * einsum('ilcd,cdkl->ik',l_,t_)
    Foo +=       einsum('ilcd,cdkl->ik',l ,t )
    Fvv  = 0.5 * einsum('klad,cdkl->ca',l_,t_)
    Fvv +=       einsum('klad,cdkl->ca',l ,t )

    L -= einsum('ik,kjab->ijab',Foo,oovv)
    L -= einsum('jk,ikab->ijab',Foo,oovv)
    L -= einsum('ca,ijcb->ijab',Fvv,oovv)
    L -= einsum('cb,ijac->ijab',Fvv,oovv)

    oovv_ = ovvo_ = t_ = l_ = None
    loooo = lvvvv = rovvo = rovov = rovvo_ = tmp = None
    Foo = Fvv = None
    return -1j*T, 1j*L

def compute_rho1(t, l): # normal ordered, asymmetric
    t_ = t - t.transpose(0,1,3,2)
    l_ = l - l.transpose(0,1,3,2)
    dvv  = 0.5 * einsum('ikac,bcik->ba',l_,t_)
    dvv +=       einsum('ikac,bcik->ba',l ,t )
    doo  = 0.5 * einsum('jkac,acik->ji',l_,t_)
    doo +=       einsum('jkac,acik->ji',l ,t )
    doo *= - 1.0
    return doo, dvv

def compute_drho1(t, l, dt, dl):
    t_ = t - t.transpose(0,1,3,2)
    l_ = l - l.transpose(0,1,3,2)
    dt_ = dt - dt.transpose(0,1,3,2)
    dl_ = dl - dl.transpose(0,1,3,2)
    dvv  = 0.5 * einsum('ikac,bcik->ba',dl_,t_)
    dvv +=       einsum('ikac,bcik->ba',dl ,t )
    dvv += 0.5 * einsum('ikac,bcik->ba',l_,dt_)
    dvv +=       einsum('ikac,bcik->ba',l ,dt )
    doo  = 0.5 * einsum('jkac,acik->ji',dl_,t_)
    doo +=       einsum('jkac,acik->ji',dl ,t )
    doo += 0.5 * einsum('jkac,acik->ji',l_,dt_)
    doo +=       einsum('jkac,acik->ji',l ,dt )
    doo *= - 1.0
    t_ = l_ = dt_ = dl_ = None
    return doo, dvv

def compute_rho12(t, l): # normal ordered, asymmetric
    t_ = t - t.transpose(0,1,3,2)
    l_ = l - l.transpose(0,1,3,2)
    dvv  = 0.5 * einsum('ikac,bcik->ba',l_,t_)
    dvv +=       einsum('ikac,bcik->ba',l ,t )
    doo  = 0.5 * einsum('jkac,acik->ji',l_,t_)
    doo +=       einsum('jkac,acik->ji',l ,t )
    doo *= - 1.0

    dovvo  = einsum('jkbc,acik->jabi',l_,t )
    dovvo += einsum('jkbc,acik->jabi',l ,t_)
    dovov  = - einsum('jkcb,caik->jaib',l,t)

    dvvvv = einsum('ijab,cdij->cdab',l,t)
    doooo = einsum('klab,abij->klij',l,t)

    dvvoo  = t.copy()
    dvvoo += einsum('ladi,bdjl->abij',dovvo,t )
    dvvoo -= einsum('laid,bdjl->abij',dovov,t )
    dvvoo += einsum('ladi,bdjl->abij',dovvo,t_)
    dvvoo -= einsum('lajd,bdli->abij',dovov,t )
    dvvoo += einsum('klij,abkl->abij',doooo,t ) 
    dvvoo -= einsum('ac,cbij->abij',dvv,t)
    dvvoo -= einsum('bc,acij->abij',dvv,t)
    dvvoo += einsum('ki,abkj->abij',doo,t)
    dvvoo += einsum('kj,abik->abij',doo,t)
    t_ = l_ = None
    return (doo, dvv), (doooo, l.copy(), dvvoo, dovvo, dovov, dvvvv)

def compute_rdm1(t, l, dt=None, dl=None):
    if dt is None:
        doo, dvv = compute_rho1(t, l)
        no = doo.shape[0]
        doo += np.eye(no)
    else:
        doo, dvv = compute_drho1(t, l, dt, dl)

    doo += doo.T.conj()
    dvv += dvv.T.conj()
    doo *= 0.5
    dvv *= 0.5
    return doo, dvv

def compute_rdm12(t, l):
    d1, d2 = compute_rho12(t, l)
    doo, dvv = d1 
    doooo, doovv, dvvoo, dovvo, dovov, dvvvv = d2

    no = doo.shape[0]
    Ioo = np.eye(no)
    doooo += einsum('ki,lj->klij',Ioo,doo)
    doooo += einsum('lj,ki->klij',Ioo,doo)
    doooo += einsum('ki,lj->klij',Ioo,Ioo)
    dovov += einsum('ji,ab->jaib',Ioo,dvv)
    doo += Ioo 

    doo += doo.T.conj()
    dvv += dvv.T.conj()
    doooo += doooo.transpose(2,3,0,1).conj()
    doovv += dvvoo.transpose(2,3,0,1).conj()
    dovvo += dovvo.transpose(3,2,1,0).conj()
    dovov += dovov.transpose(2,3,0,1).conj()
    dvvvv += dvvvv.transpose(2,3,0,1).conj()
    doo *= 0.5
    dvv *= 0.5
    doooo *= 0.5
    doovv *= 0.5
    dovvo *= 0.5
    dovov *= 0.5
    dvvvv *= 0.5
    return (doo, dvv), (doooo, doovv, dovvo, dovov, dvvvv)

def compute_X(d1, d2, eris, time=None):
    Aovvo = compute_Aovvo(d1)
    _, fov, fvo, _ = compute_comm(d1, d2, eris, time, full=False)  
    Bov = fvo.T - fov.conj()
    no, nv = fov.shape
    
    Bov = Bov.reshape(no*nv)
    Aovvo = Aovvo.reshape(no*nv,no*nv)
    Rvo = np.dot(np.linalg.inv(Aovvo),Bov)
    Rvo = Rvo.reshape(nv,no)
    R = np.block([[np.zeros((no,no)),Rvo.T.conj()],
                   [Rvo,np.zeros((nv,nv))]])
    if eris.picture == 'I':
        R += np.block([[eris.Roo,np.zeros((no,nv))],
                       [np.zeros((nv,no)),eris.Rvv]])
    Bov = Aovvo = fvo = fvo = None
    return 1j*R, None

def compute_Aovvo(d1):
    doo, dvv = d1
    no, nv = doo.shape[0], dvv.shape[0]
    Aovvo  = einsum('ab,ji->iabj',np.eye(nv),doo)
    Aovvo -= einsum('ij,ab->iabj',np.eye(no),dvv)
    return Aovvo

def compute_comm(d1, d2, eris, time=None, full=True):
    doo, dvv = d1 
    doooo, doovv, dovvo, dovov, dvvvv =  d2
    doooo_ = doooo - doooo.transpose(0,1,3,2)
    doovv_ = doovv - doovv.transpose(0,1,3,2)
    dovvo_ = dovvo - dovov.transpose(0,1,3,2)
    dvvvv_ = dvvvv - dvvvv.transpose(0,1,3,2)

    eris.make_tensors(time)
    ovvv = eris.ovvv
    ooov = eris.ooov
    ovvv_ = ovvv - ovvv.transpose(0,1,3,2)
    ooov_ = ooov - ooov.transpose(1,0,2,3)

    fvo  = einsum('ab,ib->ai',dvv,eris.hov.conj())
    fvo += 0.5 * einsum('abcd,ibcd->ai',dvvvv_,ovvv_.conj())
    fvo += 0.5 * einsum('klab,klib->ai',doovv_.conj(),ooov_)
    fvo +=       einsum('jabk,ijkb->ai',dovvo_,ooov_.conj())
    fvo += einsum('abcd,ibcd->ai',dvvvv,ovvv.conj())
    fvo += einsum('klab,klib->ai',doovv.conj(),ooov)
    fvo += einsum('jabk,ijkb->ai',dovvo,ooov.conj())
    fvo += einsum('jakb,jikb->ai',dovov,ooov.conj())

    fov  = einsum('ij,ja->ia',doo,eris.hov)
    fov += 0.5 * einsum('ijkl,lkja->ia',doooo_,ooov_)
    fov += 0.5 * einsum('jidc,jadc->ia',doovv_,ovvv_.conj())
    fov +=       einsum('ibcj,jcba->ia',dovvo_,ovvv_)
    fov += einsum('ijkl,lkja->ia',doooo,ooov)
    fov += einsum('jidc,jadc->ia',doovv,ovvv.conj())
    fov += einsum('ibcj,jcba->ia',dovvo,ovvv)
    fov += einsum('ibjc,jcab->ia',dovov,ovvv)

    foo = fvv = None
    if full:
        oovv = eris.oovv
        oooo = eris.oooo
        vvvv = eris.vvvv
        ovvo = eris.ovvo
        ovov = eris.ovov
        oooo_ = oooo - oooo.transpose(0,1,3,2)
        oovv_ = oovv - oovv.transpose(0,1,3,2)
        ovvo_ = ovvo - ovov.transpose(0,1,3,2)
        vvvv_ = vvvv - vvvv.transpose(0,1,3,2)

        fvv  = einsum('ac,cb->ab',dvv,eris.hvv)
        fvv += 0.5 * einsum('aecd,cdbe->ab',dvvvv_,vvvv_)
        fvv += 0.5 * einsum('ijae,ijbe->ab',doovv_.conj(),oovv_)
        fvv +=       einsum('iacj,jcbi->ab',dovvo_,ovvo_)
        fvv += einsum('aecd,cdbe->ab',dvvvv,vvvv)
        fvv += einsum('ijae,ijbe->ab',doovv.conj(),oovv)
        fvv += einsum('iacj,jcbi->ab',dovvo,ovvo)
        fvv += einsum('iajc,jcib->ab',dovov,ovov)
        
        foo  = einsum('ik,kj->ij',doo,eris.hoo)
        foo += 0.5 * einsum('imkl,kljm->ij',doooo_,oooo_)
        foo += 0.5 * einsum('imab,jmab->ij',doovv_,oovv_.conj())
        foo +=       einsum('iabk,kbaj->ij',dovvo_,ovvo_)
        foo += einsum('imkl,kljm->ij',doooo,oooo)
        foo += einsum('imab,jmab->ij',doovv,oovv.conj())
        foo += einsum('iabk,kbaj->ij',dovvo,ovvo)
        foo += einsum('iakb,kbja->ij',dovov,ovov)
        oooo_ = oovv_ = ovvo_ = vvvv_ = None

    doooo_ = doovv_ = dovvo_ = dvvvv_ = None
    ovvv_ = ooov_ = None
    return foo,fov,fvo,fvv

def compute_der1(d1, dd1, C, X):
    # analytical 1st time derivative of <U^{-1}p+qU>
    dd1 = rotate1(dd1, C.T.conj())
    dC = - np.dot(X, C)
    tmp  = einsum('sr,rp,sq->qp',d1,dC,C.conj())
    tmp += einsum('sr,rp,sq->qp',d1,C,dC.conj())
    return dd1 + tmp

def compute_energy(d1, d2, eris, time=None):
    eris.make_tensors(time)
    doo, dvv = d1 
    doooo, doovv, dovvo, dovov, dvvvv =  d2
    doooo_ = doooo - doooo.transpose(0,1,3,2)
    doovv_ = doovv - doovv.transpose(0,1,3,2)
    dovvo_ = dovvo - dovov.transpose(0,1,3,2)
    dvvvv_ = dvvvv - dvvvv.transpose(0,1,3,2)

    oooo = eris.oooo
    oovv = eris.oovv
    ovvo = eris.ovvo
    ovov = eris.ovov
    vvvv = eris.vvvv
    oooo_ = oooo - oooo.transpose(0,1,3,2)
    oovv_ = oovv - oovv.transpose(0,1,3,2)
    ovvo_ = ovvo - ovov.transpose(0,1,3,2)
    vvvv_ = vvvv - vvvv.transpose(0,1,3,2)
 
    e  = einsum('ij,ji',eris.hoo,doo) 
    e += einsum('ab,ba',eris.hvv,dvv)
    e += 0.25 * einsum('ijkl,klij',oooo_,doooo_) 
    e += 0.25 * einsum('abcd,cdab',vvvv_,dvvvv_) 
    e += einsum('jabi,ibaj',ovvo_,dovvo_)
    tmp  = 0.25 * einsum('ijab,ijab',oovv_,doovv_.conj())
    tmp += tmp.conj()
    e += tmp
    e *= 2.0

    e += einsum('ijkl,klij',oooo,doooo) 
    e += einsum('abcd,cdab',vvvv,dvvvv) 
    e += 2.0 * einsum('jabi,ibaj',ovvo,dovvo)
    e += 2.0 * einsum('jaib,ibja',ovov,dovov)
    tmp  = einsum('ijab,ijab',oovv,doovv.conj())
    tmp += tmp.conj()
    e += tmp
    doooo_ = doovv_ = dovvo_ = dvvvv_ = None
    oooo_ = oovv_ = ovvo_ = vvvv_ = None
    return e.real

def rotate1(h, C):
    return einsum('up,vq,...pq->...uv',C,C.conj(),h) 
def rotate2(eri_, C):
    eri = einsum('up,vq,pqrs->uvrs',C,C,eri_)
    return einsum('xr,ys,uvrs->uvxy',C.conj(),C.conj(),eri)

def fac_mol(w, td, time):
    if time > td:
        return 0.0 
    else:
        evlp = math.sin(math.pi*time/td)**2
        osc = 1.0
#        evlp = 1.0
#        osc = math.sin(w*time)
        return evlp * osc

def fac_sol(sigma, w, td, time):
    t0 = 0.5 * td
    dt = time - t0
    if time > td:
        return 0.0 
    else:
        evlp = np.exp(-0.5*(dt/sigma)**2)
        osc = math.cos(w*dt)
        return evlp * osc

def phase_hubbard(A0, sigma, w, td, time):
    return A0*fac_sol(sigma, w, td, time)

def mo_ints_mol(mf, z=np.zeros(3)): # field strength folded into z
    nmo = mf.mol.nao_nr()
    h0 = mf.get_hcore()
    h0 = einsum('uv,up,vq->pq',h0,mf.mo_coeff,mf.mo_coeff)

    eri = mf.mol.intor('int2e_sph')
    eri = ao2mo.full(eri, mf.mo_coeff, compact=False)
    eri = eri.reshape(nmo,nmo,nmo,nmo).transpose(0,2,1,3)

    mu = mf.mol.intor('int1e_r')
    mu = einsum('xuv,up,vq->xpq',mu,mf.mo_coeff,mf.mo_coeff)
    h1 = einsum('xpq,x->pq',mu,z)

    charges = mf.mol.atom_charges()
    coords  = mf.mol.atom_coords()
    nucl_dip = einsum('i,ix->x', charges, coords)
    return h0, h1, eri, mu, nucl_dip

def mo_ints_cell(mf, z=np.zeros(3)): # A0 folded into z
    nmo = mf.mol.nao_nr()
    h0 = mf.get_hcore()
    h0 = einsum('uv,up,vq->pq',h0,mf.mo_coeff,mf.mo_coeff)

    p = mf.cell.pbc_intor('int1e_ipovlp',hermi=0,comp=3)
    p = -1j*p.conj().transpose(0,2,1)
    p = einsum('xuv,up,vq->xpq',p,mf.mo_coeff,mf.mo_coeff) 
    h1 = einsum('xpq,x->pq',p,z)

    eri = mf.with_df.ao2mo(mf.mo_coeff, mf.kpt, compact=False)
    eri = eri.reshape(nmo,nmo,nmo,nmo).transpose(0,2,1,3)
    return h0, h1, eri, p

def update_RK(t, l, C, eris, time, h, RK, orb=True):
    F = None
    eris.rotate(C)
    d1, d2 = compute_rdm12(t, l)
    e = compute_energy(d1, d2, eris, time=None)
    if orb: 
        X1, _ = compute_X(d1, d2, eris, time) 
    else:
        X1 = np.zeros_like(C, dtype=complex)
    dt1, dl1 = update_amps(t, l, eris, time)
    if RK == 1:
        F = compute_comm(d1, d2, eris, time) # F_{qp} = <[H_U,p+q]>
        F = np.block([[F[0],F[1]],[F[2],F[3]]])
        F -= F.T.conj()
        F = rotate1(F, C.T.conj())
        return dt1, dl1, X1, e, F
    if RK == 4:
        t_ = t + dt1*h*0.5
        l_ = l + dl1*h*0.5
        if orb: 
            C_ = np.dot(scipy.linalg.expm(-X1*h*0.5), C)
            eris.rotate(C_)
            d1, d2 = compute_rdm12(t_, l_)
            X2, _ = compute_X(d1, d2, eris, time+h*0.5)
        dt2, dl2 = update_amps(t_, l_, eris, time+h*0.5)

        t_ = t + dt2*h*0.5
        l_ = l + dl2*h*0.5
        if orb: 
            C_ = np.dot(scipy.linalg.expm(-X2*h*0.5), C)
            eris.rotate(C_)
            d1, d2 = compute_rdm12(t_, l_)
            X3, _ = compute_X(d1, d2, eris, time+h*0.5)
        dt3, dl3 = update_amps(t_, l_, eris, time+h*0.5)
    
        t_ = t + dt3*h
        l_ = l + dl3*h
        if orb:
            C_ = np.dot(scipy.linalg.expm(-X3*h), C)
            eris.rotate(C_)
            d1, d2 = compute_rdm12(t_, l_)
            X4, _ = compute_X(d1, d2, eris, time+h)
        dt4, dl4 = update_amps(t_, l_, eris, time+h)
    
        dt = (dt1 + 2.0*dt2 + 2.0*dt3 + dt4)/6.0
        dl = (dl1 + 2.0*dl2 + 2.0*dl3 + dl4)/6.0
        if orb: 
            X = (X1 + 2.0*X2 + 2.0*X3 + X4)/6.0
            X1 = X2 = X3 = X4 = None
        else:
            X = np.zeros_like(C, dtype=complex)
        dt1 = dt2 = dt3 = dt4 = None
        dl1 = dl2 = dl3 = dl4 = None
        d1 = d2 = t_ = l_ = C_ = None
        return dt, dl, X, e, F

def compute_sqrt_fd(mo_energy, beta, mu):
    fd = (mo_energy - mu) * beta
    fd = np.exp(fd) + 1.0
    fd = np.reciprocal(fd) # n_p
    fd_ = 1.0 - fd # 1-n_p
    return np.sqrt(fd), np.sqrt(fd_)

def make_bogoliubov1(h, fd, fd_):
    no = len(fd)
    hb = np.zeros((no*2,)*2, dtype=complex)
    hb[:no,:no] = einsum('pq,p,q->pq',h,fd ,fd )
    hb[no:,no:] = einsum('pq,p,q->pq',h,fd_,fd_)
    hb[:no,no:] = einsum('pq,p,q->pq',h,fd ,fd_)
    hb[no:,:no] = einsum('pq,p,q->pq',h,fd_,fd)
    return hb

def make_bogoliubov2(eri, fd, fd_):
    no = len(fd)
    erib = np.zeros((no*2,)*4, dtype=complex)
    erib[:no,:no,:no,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd ,fd ,fd )
    erib[no:,:no,:no,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd ,fd ,fd )
    erib[:no,no:,:no,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd_,fd ,fd )
    erib[:no,:no,no:,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd ,fd_,fd )
    erib[:no,:no,:no,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd ,fd ,fd_)
    erib[:no,:no,no:,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd ,fd_,fd_)
    erib[no:,no:,:no,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd_,fd ,fd )
    erib[:no,no:,no:,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd_,fd_,fd )
    erib[:no,no:,:no,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd_,fd ,fd_)
    erib[no:,:no,:no,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd ,fd ,fd_)
    erib[no:,:no,no:,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd ,fd_,fd )
    erib[:no,no:,no:,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd_,fd_,fd_)
    erib[no:,:no,no:,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd ,fd_,fd_)
    erib[no:,no:,:no,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd_,fd ,fd_)
    erib[no:,no:,no:,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd_,fd_,fd )
    erib[no:,no:,no:,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd_,fd_,fd_)
    return erib

def make_Roo(mo_energy, fd):
    Roo = np.diag(mo_energy)
    Roo -= einsum('pq,p,q->pq',np.diag(mo_energy),fd, fd ) 
#    Rvv = np.diag(mf.mo_energy)
#    Rvv -= einsum('pq,p,q->pq',np.diag(mf.mo_energy),fd_,fd_)
    return Roo 

def compute_phys1(d1, fd, fd_): 
    # physical rdm1 and commutator 
    # d1 in fixed Bogoliubov basis
    no = len(fd)
    d1_  = einsum('pq,p,q->pq',d1[:no,:no],fd ,fd ) 
    d1_ += einsum('pq,p,q->pq',d1[:no,no:],fd ,fd_) 
    d1_ += einsum('pq,p,q->pq',d1[no:,:no],fd_,fd ) 
    d1_ += einsum('pq,p,q->pq',d1[no:,no:],fd_,fd_) 
    return d1_

