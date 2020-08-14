from pyscf.fci import direct_spin1, direct_uhf

def hop_uhf(hao, eri_ao, mo_coeff, nelec):
    moa, mob = mo_coeff
    nmoa = moa.shape[0]
    ha = einsum('uv,up,vq->pq',hao,moa.conj(),moa)
    hb = einsum('uv,up,vq->pq',hao,mob.conj(),mob)
    
    eri_aa = einsum('uvxy,up,vr->prxy',eri_ao,moa.conj(),moa)
    eri_aa = einsum('prxy,xq,ys->prqs',eri_aa,moa.conj(),moa)
    eri_bb = einsum('uvxy,up,vr->prxy',eri_ao,mob.conj(),mob)
    eri_bb = einsum('prxy,xq,ys->prqs',eri_bb,mob.conj(),mob)
    eri_ab = einsum('uvxy,up,vr->prxy',eri_ao,moa.conj(),moa)
    eri_ab = einsum('prxy,xq,ys->prqs',eri_ab,mob.conj(),mob)

    h1e = (ha, hb)
    eri = (eri_aa, eri_ab, eri_bb)
    h2e = direct_uhf.absorb(h1e, eri, nmoa, nelec, 0.5)
    def _hop(c):
        return direct_uhf.contract_2e(h2e, c, nmoa, nelec)
    return _hop

def hop_rhf(hao, eri_ao, mo_coeff, nelec):
    nmo = moa.shape[0]
    h1e = einsum('uv,up,vq->pq',hao,mo_coeff.conj(),mo_coeff)
    eri = einsum('uvxy,up,vr->prxy',eri_ao,mo_coeff.conj(),mo_coeff)
    eri = einsum('prxy,xq,ys->prqs',eri   ,mo_coeff.conj(),mo_coeff)
    h2e = direct_spin1.absorb(h1e, eri, nmo, nelec, 0.5)
    def _hop(c):
        return direct_spin1.contract_2e(h2e, c, nmo, nelec)
    return _hop

def update_ci(c, hao, eri_ao, mo_coeff, nelec, uhf):
    if uhf:
        _hop = hop_uhf(hao, eri_ao, mo_coeff, nelec)
    else:
        _hop = hop_rhf(hao, eri_ao, mo_coeff, nelec)
    return -1j * _hop(c)

def update_RK4(c, hao, eri_ao, mo_coeff, nelec, step, uhf=False, RK4=True):
    dc1 = update_ci(c, hao, eri_ao, mo_coeff, nelec, uhf)
    if not RK4:
        return dc1
    else: 
        dc2 = update_ci(c+dc1*step*0.5, hao, eri_ao, mo_coeff, nelec, uhf)
        dc3 = update_ci(c+dc2*step*0.5, hao, eri_ao, mo_coeff, nelec, uhf)
        dc4 = update_ci(c+dc3*step    , hao, eri_ao, mo_coeff, nelec, uhf)
        return (dc1+2.0*dc2+2.0*dc3+dc4)/6.0

def kernel_rt_test(mf, c, w, f0, tp, tf, step, uhf=False, RK4=True):
    nao = mf.mol.nao_nr()
    mu_ao = mf.mol.intor('int1e_r')
    hao  = mu_ao[0,:,:] * f0[0]
    hao += mu_ao[1,:,:] * f0[1]
    hao += mu_ao[2,:,:] * f0[2]

    td = 2 * int(tp/step)
    maxiter = int(tf/step)
    no, _, nv, _ = l.shape

    Aao = np.random.rand(nao,nao)
#    Aao += Aao.T
    Amo0 = ao2mo(Aao, (mo0,mo0)) # in stationary HF basis
    d1_old, d2 = compute_rdms(t, l)
    d0_old = einsum('qp,vq,up->vu',d1_old,U,U.conj()) # in stationary HF basis
    Amo = ao2mo(Aao, mo_coeff)
    A_old = einsum('pq,qp',Amo,d1_old)
    A0_old = einsum('pq,qp',Amo0,d0_old)
    tr = abs(np.trace(d1_old)-no)
    for i in range(maxiter):
        eris.ao2mo(mo_coeff)
        if i <= td:
            evlp = math.sin(math.pi*i/td)**2
#            osc = math.cos(w*(i*step-tp)) 
            osc = math.sin(w*i*step) 
            eris.h += ao2mo(hao, mo_coeff) * osc * evlp
        Amo = ao2mo(Aao, mo_coeff)
        dt, dl, X, C, d1, d2 = update_RK4(t, l, X, eris, step, RK4=RK4, RK4_X=RK4_X)
        t += step * dt
        l += step * dl
        d0 = einsum('qp,vq,up->vu',d1,U,U.conj()) # in stationary HF basis
        A = einsum('pq,qp',Amo,d1)
        A0 = einsum('pq,qp',Amo0,d0)
        dd1, d1_old = d1-d1_old, d1.copy()
        dd0, d0_old = d0-d0_old, d0.copy()
        dA, A_old = A-A_old, A
        dA0, A0_old = A0-A0_old, A0
        LHS = dd1/step
        LHS0 = dd0/step
        C0 = einsum('qp,vq,up->vu',C,U,U.conj()) # in stationary HF basis
        HA = einsum('pq,qp',Amo,C)
        HA0 = einsum('pq,qp',Amo0,C0)
        tmp  = einsum('rp,qr->qp',X,d1)
        tmp -= einsum('qr,rp->qp',X,d1)
        RHS = C + tmp
#        LHS_  = einsum('vu,up,vq->qp',LHS0,U,U.conj())
#        dU = np.dot(U, X)
#        tmp_  = einsum('vu,up,vq->qp',d0,dU,U.conj())
#        tmp_ += einsum('vu,up,vq->qp',d0,U,dU.conj())
#        LHS_ += tmp_.copy()
#        diff = LHS - LHS_
#        print(np.linalg.norm(diff))
        error = LHS-RHS
        if orb:
            U = np.dot(U, scipy.linalg.expm(step*X))
            mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
            print('time: {:.4f}, d1: {}, d1(ov/vo): {}, d0: {}, A: {}, A0: {}, A.imag: {}, normX: {}'.format(
                   i*step, np.linalg.norm(error), 
                   np.linalg.norm(error[:no,no:])+np.linalg.norm(error[no:,:no]), 
                   np.linalg.norm(LHS0-C0), 
                   abs(dA/step-HA), abs(dA0/step-HA0), A_old.imag, np.linalg.norm(X)))
        else:
            print('time: {:.4f}, d1: {}, d1(ov/vo): {}, d0: {}, A: {}, A0: {}, A.imag: {}'.format(
                   i*step, np.linalg.norm(error), 
                   np.linalg.norm(error[:no,no:])+np.linalg.norm(error[no:,:no]), 
                   np.linalg.norm(LHS0-C0), 
                   abs(dA/step-HA), abs(dA0/step-HA0), A_old.imag))
        if np.linalg.norm(error) > 1.0:
            print('diverging error!')
            break
        tr += abs(np.trace(d1_old)-no)
    print('check trace: {}'.format(tr))

def kernel_rt(mf, t, l, U, w, f0, tp, tf, step, RK4=True, RK4_X=False):
    nao = mf.mol.nao_nr()
    mu_ao = mf.mol.intor('int1e_r')
    hao  = mu_ao[0,:,:] * f0[0]
    hao += mu_ao[1,:,:] * f0[1]
    hao += mu_ao[2,:,:] * f0[2]

    td = 2 * int(tp/step)
    maxiter = int(tf/step)
    no, _, nv, _ = l.shape
    mo0 = mf.mo_coeff.copy()
    U = np.array(U, dtype=complex)
    X = np.zeros((no+nv,)*2,dtype=complex)
    mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
    eris = ERIs(mf)
    mux = np.zeros(maxiter+1,dtype=complex)  
    muy = np.zeros(maxiter+1,dtype=complex)  
    muz = np.zeros(maxiter+1,dtype=complex)  
    Hmux = np.zeros(maxiter,dtype=complex)  
    Hmuy = np.zeros(maxiter,dtype=complex)  
    Hmuz = np.zeros(maxiter,dtype=complex)  
    E = np.zeros(maxiter,dtype=complex)

    d1, d2 = compute_rdms(t, l)
    mux_mo = ao2mo(mu_ao[0,:,:], mo_coeff)
    muy_mo = ao2mo(mu_ao[1,:,:], mo_coeff)
    muz_mo = ao2mo(mu_ao[2,:,:], mo_coeff)
    mux[0] = einsum('pq,qp',mux_mo,d1)
    muy[0] = einsum('pq,qp',muy_mo,d1)
    muz[0] = einsum('pq,qp',muz_mo,d1)

    for i in range(maxiter):
        eris.ao2mo(mo_coeff)
        if i <= td:
            evlp = math.sin(math.pi*i/td)**2
#            osc = math.cos(w*(i*step-tp)) 
            osc = math.sin(w*i*step) 
            eris.h += ao2mo(hao, mo_coeff) * evlp * osc
        mux_mo = ao2mo(mu_ao[0,:,:], mo_coeff)
        muy_mo = ao2mo(mu_ao[1,:,:], mo_coeff)
        muz_mo = ao2mo(mu_ao[2,:,:], mo_coeff)
        dt, dl, X, C, d1, d2 = update_RK4(t, l, X, eris, step, RK4=RK4, RK4_X=RK4_X)
        t += step * dt
        l += step * dl
        mux[i+1] = einsum('pq,qp',mux_mo,d1)
        muy[i+1] = einsum('pq,qp',muy_mo,d1)
        muz[i+1] = einsum('pq,qp',muz_mo,d1)
        e  = einsum('pq,qp',eris.h,d1) 
        e += 0.25 * einsum('pqrs,rspq',eris.eri,d2)
        E[i] = e
        Hmux[i] = einsum('pq,qp',mux_mo,C)
        Hmuy[i] = einsum('pq,qp',muy_mo,C)
        Hmuz[i] = einsum('pq,qp',muz_mo,C)
        U = np.dot(U, scipy.linalg.expm(step*X))
        mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
        error  = (mux[i+1]-mux[i])/step - Hmux[i] 
        error += (muy[i+1]-muy[i])/step - Hmuy[i] 
        error += (muz[i+1]-muz[i])/step - Hmuz[i]
        imag  = mux[i+1].imag + muy[i+1].imag + muz[i+1].imag 
        print('time: {:.4f}, ehrenfest: {}, imag: {}, E.imag: {},'.format(i*step, abs(error), imag, E[i].imag))
#        print('mux: {}, muy: {}, muz: {}'.format(mux[i+1].real,muy[i+1].real,muz[i+1].real))
    return mux, muy, muz, E

