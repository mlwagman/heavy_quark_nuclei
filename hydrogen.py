#!/usr/bin/env python

"""
This example shows how to work with the Hydrogen radial wavefunctions.
"""
import math as m
import numpy as np
from sympy import sqrt, Eq, Integral, oo, pprint, symbols, Symbol, log, exp, diff, Sum, factorial, IndexedBase, Function, cos, sin, atan, acot, pi
from sympy.physics.hydrogen import R_nl, Psi_nlm


def main():

# Defining difference spherical coords
    nCoord = 6;
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
    r = symbols('r0:%d'%nCoord);
    t = symbols('t0:%d'%nCoord);
    p = symbols('p0:%d'%nCoord);

# Define cartesian coords
    x = symbols('x0:%d'%nCoord);
    y = symbols('y0:%d'%nCoord);
    z = symbols('z0:%d'%nCoord);

#   Potential coupling and a0
    a, B = symbols("a B")
    print(sum(rr))

    def laPlaceSpher(f,r,t,p):
        return 1/r**2*diff(r**2*(diff(f,r)),r)+1/(r**2*sin(t))*diff(sin(t)*(diff(f,t)),t)+1/(r**2*sin(t)**2)*diff(diff(f,p),p)
    print(laPlaceSpher(r**2*sin(t)+cos(p)*sin(t),r,t,p))

    def rInv(ri,rj):
        return 1/sqrt(ri**2-rj**2)
    print(rInv(r[1],r[2]))


    def Potential(rr,B,nCoord):
        V0 = -B*sum(o for i, a in enumerate(rr) for j, o  in enumerate(a) if i!=j and j>=i);
        for i in range(nCoord):
            for j in range(nCoord):
                if i!=j and j>=i:
                    V0 = V0.subs(rr[i][j],rInv(r[i],r[j]))
        return V0
    print(Potential(rr,B,nCoord))

    # Define HWF quantum numbers
    n, l, m, r, phi, theta, Z = symbols("n l m r phi theta Z")

    # Spher (i,j) to Cart(i),Cart(j) to Spher(i),Spher(j)

    def rrSpher(i,j,r,t,p):
        return  sqrt(r[j]*(r[j]-2*r[i]*(sin(t[i])*sin(t[j])*cos(p[i]-p[j])+cos(t[i])*cos(t[j]))) +r[i]**2);
    print(rrSpher(1,2,r,t,p))
    def ppSpher(i,j,r,t,p):
        return acot((cos(t[i])*r[i] - cos(t[j])*r[j])/sqrt((cos(p[i])*r[i]*sin(t[i]) - cos(p[j])*r[j]*sin(t[j]))**2 + (r[i]*sin(t[i])*sin(p[i]) - r[j]*sin(t[j])*sin(p[j]))**2));
    print(rrSpher(1,2,r,t,p))
    def ttSpher(i,j,r,t,p):
        return  atan((r[i]*sin(t[i])*sin(p[i]) - r[j]*sin(t[j])*sin(p[j]))/(cos(p[i])*r[i]*sin(t[i]) - cos(p[j])*r[j]*sin(t[j])));
    print(ttSpher(1,2,r,t,p))

     def Chi(k, nCoord, l, m, Z, rr, tt, pp, r, t, p, x, y, z, col):
         if k!=j and j>=k:
             return Chi = v[col]*Sum(1/(nCoord-1)*sum(Psi_nlm(n, l, m, rrSpher(k,j,r,t,p), ppSpher(k,j,r,t,p), ttSpher(k,j,r,t,p), Z),(j,k+1,nCoord-1))
         elif k!=j and k>=j:
             return Chi = v[col]*Sum(1/(nCoord-1)*sum(Psi_nlm(n, l, m, rrSpher(j,k,r,t,p), ppSpher(j,k,r,t,p), ttSpher(j,k,r,t,p), Z),(j,0,k-1))
         else:
             return None
 #rrTocart = [(rr[i][j],sqrt((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i]-z[j])**2)) for i, j in zip(range(nCoord), range(nCoord)) if i!=j and j>=i]
 #        ChiCart=Chi;
 #        for i in range(nCoord):
 #            for j in range(nCoord):
 #                if i!=j and j>=i:
 #                    ChiCart = (ChiCart.subs(rr[i][j],rInv(r[i],r[j]))).subs(rr[i][j],rInv(r[i],r[j]))
 #        return V0
     print(Chi(1, 2, 1, 1, 1, rr, tt, pp, r, t, p, x, y, z, 1))


    #eq=r[1]**2+r[2]*2;
    #print(diff(eq,r[1]))
    eq=rr[1][1]**2+rr[2][2]*2;
    print(diff(eq,rr[1,1]))
    print(rr[1][2].subs(rr[1][2],r[1]+r[2]))
    print(Potential(rr,B,nCoord))

#
    print("R_{21}:")
    pprint(R_nl(2, 1, a, r[0]))
    print("R_{60}:")
    pprint(R_nl(6, 0, a, r[0]))

    print("Normalization:")
    i = Integral(R_nl(1, 0, 1, r[0])**2 * r[0]**2, (r[0], 0, oo))
    pprint(Eq(i, i.doit()))
    i = Integral(R_nl(2, 0, 1, r[0])**2 * r[0]**2, (r[0], 0, oo))
    pprint(Eq(i, i.doit()))
    i = Integral(R_nl(2, 1, 1, r[0])**2 * r[0]**2, (r[0], 0, oo))
    pprint(Eq(i, i.doit()))

if __name__ == '__main__':
    main()
