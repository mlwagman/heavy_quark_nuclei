#!/usr/bin/env python

"""
This example shows how to work with the Hydrogen radial wavefunctions.
"""
import math as m
import numpy as np
from sympy import simplify, summation, sqrt, Eq, Integral, oo, pprint, symbols, Symbol, log, exp, diff, Sum, factorial, IndexedBase, Function, cos, sin, atan, acot, pi, atan2, trigsimp
from sympy.physics.hydrogen import R_nl, Psi_nlm
from itertools import permutations

# Defining difference spherical coords
nCoord = 2;
rr = np.full((nCoord,nCoord), fill_value = '',dtype=object)
for i in range(nCoord):
    for j in range(nCoord):
        rr[i][j] = Symbol('rr[{},{}]'.format(i, j))
tt = np.full((nCoord,nCoord), fill_value = '',dtype=object)
for i in range(nCoord):
    for j in range(nCoord):
        tt[i][j] = Symbol('tt[{},{}]'.format(i, j))
pp = np.full((nCoord,nCoord), fill_value = '',dtype=object)
for i in range(nCoord):
    for j in range(nCoord):
        pp[i][j] = Symbol('pp[{},{}]'.format(i, j))

# Define spherical coords
r = symbols('r0:%d'%nCoord, positive=True);
t = symbols('t0:%d'%nCoord);
p = symbols('p0:%d'%nCoord);

# Define cartesian coords
x = symbols('x0:%d'%nCoord);
y = symbols('y0:%d'%nCoord);
z = symbols('z0:%d'%nCoord);

# Define color vector
v = symbols('v0:%d'%nCoord);

#   Potential coupling and a0
a, Z = symbols("a Z", positive=True)
B = symbols("B")

#print(sum(rr))

def laPlaceSpher(f,r,t,p):
    dfdr = diff(f,r)
    #print(f'dfdr = {dfdr}')
    rpart = 1/r**2*diff(r**2*(diff(f,r)),r)
    #print(f'rpart = {rpart}')
    tpart = 1/(r**2*sin(t))*diff(sin(t)*(diff(f,t)),t)
    #print(f'tpart = {tpart}')
    ppart = 1/(r**2*sin(t)**2)*diff(diff(f,p),p)
    #print(f'ppart = {ppart}')
    nabla = rpart + tpart + ppart
    #print(f'sum = {nabla}')
    return nabla
    #return 1/r**2*diff(r**2*(diff(f,r)),r) + 1/(r**2*sin(t))*diff(sin(t)*(diff(f,t)),t) + 1/(r**2*sin(t)**2)*diff(diff(f,p),p)
print(laPlaceSpher(r[0]*exp(r[1])*t[1],r[1],t[1],p[1]))

def rInv(i,j):
    return 1/sqrt(r[i]**2 + r[j]**2 - 2*r[j]*r[i]*sin(t[i])*sin(t[j])*cos(p[i]-p[j]) - 2*r[j]*r[i]*cos(t[i])*cos(t[j]))
    #return 1/sqrt(ri**2-rj**2)
#print(rInv(r[1],r[2]))


def Potential(rr,B,nCoord):
    V0 = -B*sum(o for i, a in enumerate(rr) for j, o  in enumerate(a) if i!=j and j>=i);
    for i in range(nCoord):
        for j in range(nCoord):
            if i!=j and j>=i:
                V0 = V0.subs(rr[i][j],rInv(i,j))
    return V0
#print(Potential(rr,B,nCoord))

# Define HWF quantum numbers
n, l, m, phi, theta, Z = symbols("n l m phi theta Z")

# Spher (i,j) to Cart(i),Cart(j) to Spher(i),Spher(j)

def rrSpher(i,j,r,t,p):
    return  sqrt(r[j]*(r[j]-2*r[i]*(sin(t[i])*sin(t[j])*cos(p[i]-p[j])+cos(t[i])*cos(t[j]))) +r[i]**2);
def ttSpher(i,j,r,t,p):
    # experimentally this works assuming atan2(y, x) = atan(y / x) + phase
    return atan2(sqrt((cos(p[i])*r[i]*sin(t[i]) - cos(p[j])*r[j]*sin(t[j]))**2 + (r[i]*sin(t[i])*sin(p[i]) - r[j]*sin(t[j])*sin(p[j]))**2), (cos(t[i])*r[i] - cos(t[j])*r[j]));
def ppSpher(i,j,r,t,p):
    return  atan2((r[i]*sin(t[i])*sin(p[i]) - r[j]*sin(t[j])*sin(p[j])), (cos(p[i])*r[i]*sin(t[i]) - cos(p[j])*r[j]*sin(t[j])));


jj = Symbol('jj', integer=True)
#print(rrSpher(1,2,r,t,p))

#  Define chi(r_i) where psi(r1,..,rn)=chi(r1)*...*chi(rn)
def Chi(k, nCoord, n, l, m, Z, r, t, p, v, col):
     Chi =  0
     for j in range(0,nCoord):
         if k!=j and j>=k:
             Chi = Chi + v[col]*1/(nCoord-1)*Psi_nlm(n, l, m, rrSpher(k,j,r,t,p), ppSpher(k,j,r,t,p), ttSpher(k,j,r,t,p), Z)
         elif k!=j and k>=j:
             Chi = Chi + v[col]*1/(nCoord-1)*Psi_nlm(n, l, m, rrSpher(j,k,r,t,p), ppSpher(j,k,r,t,p), ttSpher(j,k,r,t,p), Z)
         else:
             Chi = Chi
     return Chi

def Chi_no_v(k, nCoord, n, l, m, Z, r, t, p):
     Chi =  0
     for j in range(0,nCoord):
         if k!=j and j>=k:
             Chi = Chi + 1/(nCoord-1)*Psi_nlm(n, l, m, rrSpher(k,j,r,t,p), ppSpher(k,j,r,t,p), ttSpher(k,j,r,t,p), Z)
         elif k!=j and k>=j:
             Chi = Chi + 1/(nCoord-1)*Psi_nlm(n, l, m, rrSpher(j,k,r,t,p), ppSpher(j,k,r,t,p), ttSpher(j,k,r,t,p), Z)
         else:
             Chi = Chi
     return Chi

# test chi
#print(Chi(1, 2, 1, 0, 0, 1, r, t, p, v, 1).subs(r[0],0))

#print(Potential(rr,B,nCoord))

# test laplacian of chi
#print(laPlaceSpher(Chi(1, nCoord, 1, 0, 0, 1, r, t, p, v, 1),r[1],t[1],p[1]).subs(r[0],0))

# test potential
#eq=rr[1][1]**2+rr[2][2]*2;
#print(diff(eq,rr[1,1]))
#print(rr[1][2].subs(rr[1][2],r[1]+r[2]))
print(Potential(rr,B,nCoord))


#  Define psi(r1,..,rn)=chi(r1)*...*chi(rn)
def psi(nCoord, n, l, m, Z, r, t, p, v):
    psi =  1
    for j in range(0,nCoord-1):
        psi = Chi(j, nCoord, n, l, m, Z, r, t, p, v, j)*psi
    return psi

def psi_no_v(nCoord, n, l, m, Z, r, t, p):
    psi =  1
    for k in range(0,nCoord-1):
        psi = Chi_no_v(k, nCoord, n, l, m, Z, r, t, p)*psi
    return psi



#  Define Psi(r1,..,rn)=1/n!*Sum(perms of psi(r1,..,rn)) (not sure how..)
#def Psi(k, nCoord, n, l, m, Z, r, t, p, v, col):
#    psi =  1
#    for k in range(0,nCoord):
#        psi=Chi(k, nCoord, n, l, m, Z, r, t, p, v, col)*psi
#    return psi

def main():

    Hammy = -1*laPlaceSpher(Chi(1, nCoord, 1, 0, 0, 1/a, r, t, p, v, 1),r[0],t[0],p[0]).subs(r[1],0) + (Potential(rr,B,nCoord)*Chi(1, nCoord, 1, 0, 0, 1/a, r, t, p, v, 1)).subs(r[1],0)
    #Hammy = laPlaceSpher(Chi(1, nCoord, 1, 0, 0, Z, r, t, p, v, 1),r[0],t[0],p[0]).subs(r[1],0)

    #print(simplify(Hammy.subs(v[1],1).subs(a,-2/B)))
    print(simplify(Hammy.subs(v[1],1).subs(a,1/B)))

    print("\n")

    print("\nn l m = 1 0 0 hydrogen atom")
    nn = 1
    ll = 0
    mm = 0
    wvfn = Chi(1, nCoord, nn, ll, mm, 1/a, r, t, p, v, 1)
    #wvfn = Chi(1, nCoord, nn, ll, mm, 1, r, t, p, v, 1)
    print(f'psi = {simplify(wvfn.subs(r[1],0).subs(v[1],1))}')
    print(f'psi = {simplify(wvfn.subs(r[1],0).subs(v[1],1).subs(a,1/B).subs(B,1))}')

    Hammy = -1/2*laPlaceSpher(wvfn,r[0],t[0],p[0]) + (Potential(rr,B,nCoord)*wvfn)
    #Hammy = laPlaceSpher(wvfn,r[0],t[0],p[0]) + (Potential(rr,B,nCoord)*wvfn)
    #Enl = simplify((Hammy / wvfn).subs(r[1],0).subs(v[1],1).subs(a,-2/B))
    #Enl = simplify((Hammy / wvfn).subs(r[1],0).subs(v[1],1).subs(a,2/B))
    Enl = simplify((Hammy / wvfn).subs(r[1],0).subs(v[1],1).subs(a,1/B))
    print(f"E(n={nn}, l={ll}) = {Enl}")

    print("\nn l m = 1 0 0 positronium")
    V = (Potential(rr,B,nCoord)*wvfn)
    #print(f'V = {V}')
    K1 = -1/2*laPlaceSpher(wvfn,r[0],t[0],p[0])
    #print(f'K1 = {K1}')
    K2 = -1/2*laPlaceSpher(wvfn,r[1],t[1],p[1])
    #print(f'K2 = {K2}')
    Hammy = K1 + K2 + V
    #print(f'H = {Hammy}')
    #print(f'presimp = {(Hammy / wvfn).subs(v[1],1).subs(a,-2/B).subs(r[0],1).subs(r[1],1).subs(t[0],1).subs(t[1],1).subs(p[0],1).subs(p[1],1).subs(B,1)}')
    #Enl = simplify((Hammy / wvfn).subs(v[1],1).subs(a,-2/B).subs(r[0],1).subs(r[1],1).subs(t[0],pi/2).subs(t[1],pi/2).subs(p[0],1).subs(p[1],-1).subs(B,1))
    #Enl = simplify((Hammy / wvfn).subs(v[1],1).subs(a,2/B).subs(r[0],1).subs(r[1],1).subs(t[0],pi/2).subs(t[1],pi/2).subs(p[0],1).subs(p[1],-1).subs(B,1))
    Enl = simplify((Hammy / wvfn).subs(v[1],1).subs(a,2/B).subs(r[0],1).subs(r[1],1).subs(t[0],pi/2).subs(t[1],pi/2).subs(p[0],1).subs(p[1],-1).subs(B,1))
    print(f"E(n={nn}, l={ll}) = {Enl}")

    print("\nn l m = 2 0 0 hydrogen atom")
    nn = 2
    ll = 0
    mm = 0
    wvfn = Chi(1, nCoord, nn, ll, mm, 1/a, r, t, p, v, 1)
    #Hammy = laPlaceSpher(wvfn,r[0],t[0],p[0]) + (Potential(rr,B,nCoord)*wvfn)
    #Enl = simplify((Hammy / wvfn).subs(r[1],0).subs(v[1],1).subs(a,-2/B))
    Hammy = -1/2*laPlaceSpher(wvfn,r[0],t[0],p[0]) + (Potential(rr,B,nCoord)*wvfn)
    Enl = simplify((Hammy / wvfn).subs(r[1],0).subs(v[1],1).subs(a,1/B))
    print(f"E(n={nn}, l={ll}) = {Enl}")

    print("\nn l m = 2 1 0 hydrogen atom")
    nn = 2
    ll = 1
    mm = 0
    wvfn = Chi(1, nCoord, nn, ll, mm, 1/a, r, t, p, v, 1)
    Hammy = -1/2*laPlaceSpher(wvfn,r[0],t[0],p[0]) + (Potential(rr,B,nCoord)*wvfn)
    Enl = simplify((Hammy / wvfn).subs(r[1],0).subs(v[1],1).subs(a,1/B))
    print(f"E(n={nn}, l={ll}) = {Enl}")

    print("\nn l m = 2 1 1 hydrogen atom")
    nn = 2
    ll = 1
    mm = 1
    wvfn = Chi(1, nCoord, nn, ll, mm, 1/a, r, t, p, v, 1)
    Hammy = -1/2*laPlaceSpher(wvfn,r[0],t[0],p[0]) + (Potential(rr,B,nCoord)*wvfn)
    Enl = trigsimp(simplify((Hammy / wvfn).subs(r[1],0).subs(v[1],1).subs(a,1/B)))
    print(f"E(n={nn}, l={ll}) = {Enl}")

    print("\nn l m = 2 1 -1 hydrogen atom")
    nn = 2
    ll = 1
    mm = -1
    wvfn = Chi(1, nCoord, nn, ll, mm, 1/a, r, t, p, v, 1)
    Hammy = -1/2*laPlaceSpher(wvfn,r[0],t[0],p[0]) + (Potential(rr,B,nCoord)*wvfn)
    Enl = trigsimp(simplify((Hammy / wvfn).subs(r[1],0).subs(v[1],1).subs(a,1/B)))
    print(f"E(n={nn}, l={ll}) = {Enl}")


    #print(list(permutations(range(3))))

#
    #print("R_{21}:")
    #pprint(R_nl(2, 1, a, r[0]))
    #print("R_{60}:")
    #pprint(R_nl(6, 0, a, r[0]))

    #print("Normalization:")
    #i = Integral(R_nl(1, 0, 1, r[0])**2 * r[0]**2, (r[0], 0, oo))
    #pprint(Eq(i, i.doit()))
    #i = Integral(R_nl(2, 0, 1, r[0])**2 * r[0]**2, (r[0], 0, oo))
    #pprint(Eq(i, i.doit()))
    #i = Integral(R_nl(2, 1, 1, r[0])**2 * r[0]**2, (r[0], 0, oo))
    #pprint(Eq(i, i.doit()))

if __name__ == "__main__":
    main()
