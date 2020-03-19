import numpy

# CCS
def simple_update_s(Hoo, Hvv, Hov, s):
    ds = dt*(- Hov - numpy.dot(s,Hvv) + numpy.dot(Hoo,s) + numpy.linalg.multi_dot([s,Hov.T,s]))
    return s + ds

def mobius_update_s(Hoo, Hvv, Hov, s, dt):
    a = Hoo
    b = - Hov
    c = - Hov.T
    d = Hvv
    alpha = numpy.eye(a.shape[0]) + dt*a
    beta = dt*b
    gamma = dt*c
    delta = numpy.eye(d.shape[0]) + dt*d
    num = numpy.dot(alpha,s) + beta
    denom = numpy.dot(gamma,s) + delta
    return numpy.dot(num, numpy.linalg.inv(denom))

def mp2_guess(Hoo, Hvv, Hov):
    eo, ev = Hoo.diagonal(), Hvv.diagonal()
    eov = numpy.einsum('i,a->ia', eo, -ev)
    return Hov / eov

def energy(Hoo, Hov, s):
    return sum(Hoo.diagonal()) + numpy.einsum('ia,ia', Hov, s)

no = 4 
nv = 6
dt = 0.1
tmax = 20

nmo = no + nv
H = numpy.random.rand(nmo, nmo)
H = H + H.T

w, v = numpy.linalg.eigh(H)
e0 = sum(w[:no])
print('Exact ground state energy: {}'.format(e0))

Hov = H[:no,no:]
Hoo = H[:no,:no]
Hvv = H[no:, no:]

#s_simple_0 = numpy.zeros((no, nv))
#s_simple_mp2 = mp2_guess(Hoo, Hvv, Hov)
#s_simple_rand = numpy.random.rand(no, nv)
#s_mobius_0 = numpy.zeros((no, nv))
#s_mobius_mp2 = mp2_guess(Hoo, Hvv, Hov)
#s_mobius_rand = numpy.random.rand(no, nv)
s_simple = numpy.zeros((no, nv))
s_mobius = numpy.zeros((no, nv))
ts = numpy.arange(0.0, tmax, dt)

for t in ts:
#    s_simple_0 = simple_update_s(Hoo, Hvv, Hov, s_simple_0) 
#    s_simple_mp2 = simple_update_s(Hoo, Hvv, Hov, s_simple_mp2) 
#    s_simple_rand = simple_update_s(Hoo, Hvv, Hov, s_simple_rand) 
#    s_mobius_0 = mobius_update_s(Hoo, Hvv, Hov, s_mobius_0, dt)
#    s_mobius_mp2 = mobius_update_s(Hoo, Hvv, Hov, s_mobius_mp2, dt)
#    s_mobius_rand = mobius_update_s(Hoo, Hvv, Hov, s_mobius_rand, dt)
    s_simple = simple_update_s(Hoo, Hvv, Hov, s_simple) 
    s_mobius = mobius_update_s(Hoo, Hvv, Hov, s_mobius, dt)
    e_simple = energy(Hoo, Hov, s_simple)
    e_mobius = energy(Hoo, Hov, s_mobius)
    print('t={} dE_simple={} dE_mobius={}'.format(
          t, e_simple - e0, e_mobius - e0))
    

