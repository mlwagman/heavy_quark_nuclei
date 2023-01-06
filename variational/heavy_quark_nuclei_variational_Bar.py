# Wagman Feb 2022

if __name__ == '__main__':


    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import colors
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import time
    import csv
    import h5py
    import tqdm.auto as tqdm
    import os, sys
    import argparse
    import copy
    import scipy
    import mpmath


    from hydrogen_Bar import *
    from config import *



    parser = argparse.ArgumentParser()
    parser.add_argument('--N_walkers', default=200, type=int)
    parser.add_argument('--N_train', default=5000, type=int)
    parser.add_argument('--N_exp', default=1, type=int)
    parser.add_argument('--OLO', default="LO", type=str)
    parser.add_argument('--log10_learn_rate', default=3, type=float)
    parser.add_argument('--alpha', default=0.2, type=float)
    parser.add_argument('--c_loss', default=0, type=float)
    parser.add_argument('--v_loss', default=0, type=float)
    parser.add_argument('--output', default="./wvfn", type=str)
    globals().update(vars(parser.parse_args()))



    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 14})

    N_coord = nCoord

    Nc=3
    nf=4

    CF = (Nc**2 - 1)/(2*Nc)
    VB = alpha*CF
    if N_coord > 2:
        VB = alpha*CF/(Nc-1)


    beta0 = 11/3*Nc - 2/3*nf
    beta1 = 34/3*Nc**2 - 20/3*Nc*nf/2 - 2*CF*nf
    beta2 = 2857/54*Nc**3 + CF**2*nf-205/9*Nc*CF*nf/2-1415/27*Nc**2*nf/2+44/9*CF*(nf/2)**2+158/27*Nc*(nf/2)**2
    aa1 = 31/9*Nc-10/9*nf
    zeta3 = scipy.special.zeta(3)
    zeta5 = scipy.special.zeta(5)
    zeta51 = 1/2 + 1/3 + 1/7 + 1/51 + 1/4284
    zeta6 = scipy.special.zeta(6)
    aa2 = ( 4343/162 + 6*np.pi**2 - np.pi**4/4 + 22/3*zeta3 )*Nc**2 - ( 1798/81 + 56/3*zeta3 )*Nc*nf/2 - ( 55/3 - 16*zeta3  )*CF*nf/2 + (10/9*nf)**2
    dFF = (18-Nc**2+Nc**4)/(96*Nc**2)
    dFA = Nc*(Nc**2+6)/48
    alpha4 = float(mpmath.polylog(4,1/2))*0+(-np.log(2))**4/(4*3*2*1)
    ss6 = zeta51+zeta6
    aa30 = dFA*( np.pi**2*( 7432/9-4736*alpha4+np.log(2)*(14752/3-3472*zeta3)-6616*zeta3/3)  +  np.pi**4*(-156+560*np.log(2)/3+496*np.log(2)**2/3)+1511*np.pi**6/45)  + Nc**3*(385645/2916 + np.pi**2*( -953/54 +584/3*alpha4 +175/2*zeta3 + np.log(2)*(-922/9+217*zeta3/3) ) +584*zeta3/3 + np.pi**4*( 1349/270-20*np.log(2)/9-40*np.log(2)**2/9 ) -1927/6*zeta5 -143/2*zeta3**2-4621/3024*np.pi**6+144*ss6  )
    aa31 = dFF*( np.pi**2*(1264/9-976*zeta3/3+np.log(2)*(64+672*zeta3)) + np.pi**4*(-184/3+32/3*np.log(2)-32*np.log(2)**2) +10/3*np.pi**6 ) + CF**2/2*(286/9+296/3*zeta3-160*zeta5)+Nc*CF/2*(-71281/162+264*zeta3+80*zeta5)+Nc**2/2*(-58747/486+np.pi**2*(17/27-32*alpha4+np.log(2)*(-4/3-14*zeta3)-19/3*zeta3)-356*zeta3+np.pi**4*(-157/54-5*np.log(2)/9+np.log(2)**2)+1091*zeta5/6+57/2*zeta3**2+761*np.pi**6/2520-48*ss6)
    aa32 = Nc/4*(12541/243+368/3*zeta3+64*np.pi**4/135)+CF/4*(14002/81-416*zeta3/3)
    aa33 = -(20/9)**3*1/8
    aa3 = aa30+aa31*nf+aa32*nf**2+aa33*nf**3

    N_skip = 10
    N_refresh_metropolis = 1
    patience_factor = 10

    print(f'precomputing wavefunctions')
    psi_time = time.time()
    #psitab = []
    #psitab.append(psi_no_v(N_coord, r, t, p, C, A))

    if N_exp == 1:
        psitab = psi_no_v(N_coord, r, t, p, C, A)
    else:
        psitab = psi_no_v_3(N_coord, r, t, p, C, A)
    print(f"precomputed wavefunctions in {time.time() - psi_time} sec")

    print(f'precomputing wavefunction Laplacians')
    nabla_psi_time = time.time()
    #nabla_psitab = []
    #nabla_psitab.append(nabla_psi_no_v(N_coord, r, t, p, C, A))
    if N_exp == 1:
        nabla_psitab = nabla_psi_no_v(N_coord, r, t, p, C, A)
    else:
        nabla_psitab = nabla_psi_no_v_3(N_coord, r, t, p, C, A)

    print(f"precomputed wavefunction Laplacians in {time.time() - nabla_psi_time} sec")

    def total_Psi_nlm(Rs, A_n, C_n, psi_fn):
        N_walkers = Rs.shape[0]
        assert Rs.shape == (N_walkers, N_coord, 3)
        Psi_nlm_s = torch.zeros((N_walkers), dtype=torch.complex64)
        # convert to spherical
        x = Rs[:,:,0]
        y = Rs[:,:,1]
        z = Rs[:,:,2]
        r_n = torch.sqrt(x**2 + y**2 + z**2)
        t_n = torch.atan2(torch.sqrt(x**2 + y**2), z)
        p_n = torch.atan2(y, x)
        # evaluate wavefunction
        for i in range(N_walkers):
           Psi_nlm_s[i] = psi_fn(C_n, A_n, r_n[i], t_n[i], p_n[i])
        return Psi_nlm_s

    def nabla_total_Psi_nlm(Rs, A_n, C_n, nabla_psi_fn):
        nabla_psi_time = time.time()
        N_walkers = Rs.shape[0]
        assert Rs.shape == (N_walkers, N_coord, 3)
        nabla_Psi_nlm_s = torch.zeros((N_walkers), dtype=torch.complex64)
        # convert to spherical
        x = Rs[:,:,0]
        y = Rs[:,:,1]
        z = Rs[:,:,2]
        r_n = torch.sqrt(x**2 + y**2 + z**2)
        t_n = torch.atan2(torch.sqrt(x**2 + y**2), z)
        p_n = torch.atan2(y, x)
        # evaluate wavefunction
        for i in range(N_walkers):
            nabla_Psi_nlm_s[i] = nabla_psi_fn(C_n, A_n, r_n[i], t_n[i], p_n[i])
        print(f"calculated nabla in {time.time() - nabla_psi_time} sec")
        return nabla_Psi_nlm_s


    def potential_no_Psi_nlm(Rs, A_n, C_n, psi_fn):
        N_walkers = Rs.shape[0]
        assert Rs.shape == (N_walkers, N_coord, 3)
        V_Psi_nlm_s = torch.zeros((N_walkers), dtype=torch.complex64)
        wvfn = total_Psi_nlm(Rs, A_n, C_n, psi_fn)
        for i in range(N_walkers):
            x = Rs[i,:,0]
            y = Rs[i,:,1]
            z = Rs[i,:,2]
            # evaluate potential
            V = 0
            for a in range(N_coord):
                for b in range(N_coord):
                    if b > a:
                        rabs = np.sqrt( (x[a]-x[b])**2 + (y[a]-y[b])**2 + (z[a]-z[b])**2 )
                        mu=4*alpha
                        rprime = rabs*np.exp(np.euler_gamma)*mu
                        if OLO == "LO":
                         V += -VB/rabs*(1)
                        elif OLO == "NLO":
                         V += -VB/rabs*(1+ alpha/(4*np.pi)*(2*beta0*np.log(rprime)+aa1))
                        elif OLO == "mNLO":
                         V += -1*VB/rabs*(1 + alpha/(4*np.pi)*(2*beta0*np.log(rprime)+aa1)) -1*CF*Nc*alpha**2/(N_coord-1)/(rabs**2)
                        elif OLO == "NNLO":
                             if N_coord > 2:
                                 V += -1*VB/rabs*(1 + alpha/(4*np.pi)*(2*beta0*np.log(rprime)+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*np.log(rprime)**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*np.log(rprime)+ aa2 + Nc*(Nc-2)/2*((np.pi)**4-12*(np.pi)**2) ) )
                             else:
                                 V += -1*VB/rabs*(1 + alpha/(4*np.pi)*(2*beta0*np.log(rprime)+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*np.log(rprime)**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*np.log(rprime)+ aa2 ) )
            V_Psi_nlm_s[i] = V
        return V_Psi_nlm_s

    def potential_total_Psi_nlm(Rs, A_n, C_n, psi_fn):
        N_walkers = Rs.shape[0]
        assert Rs.shape == (N_walkers, N_coord, 3)
        V_Psi_nlm_s = torch.zeros((N_walkers), dtype=torch.complex64)
        wvfn = total_Psi_nlm(Rs, A_n, C_n, psi_fn)
        for i in range(N_walkers):
            x = Rs[i,:,0]
            y = Rs[i,:,1]
            z = Rs[i,:,2]
            # evaluate potential
            V = 0
            for a in range(N_coord):
                for b in range(N_coord):
                    if b > a:
                        rabs = np.sqrt( (x[a]-x[b])**2 + (y[a]-y[b])**2 + (z[a]-z[b])**2 )
                        mu=4*alpha
                        rprime = rabs*np.exp(np.euler_gamma)*mu
                        if OLO == "LO":
                         V += -VB/rabs*(1)
                        elif OLO == "NLO":
                         V += -VB/rabs*(1+ alpha/(4*np.pi)*(2*beta0*np.log(rprime)+aa1))
                        elif OLO == "mNLO":
                         V += -1*VB/rabs*(1 + alpha/(4*np.pi)*(2*beta0*np.log(rprime)+aa1)) -1*CF*Nc*alpha**2/(N_coord-1)/(rabs**2)
                        elif OLO == "NNLO":
                             if N_coord > 2:
                                 V += -1*VB/rabs*(1 + alpha/(4*np.pi)*(2*beta0*np.log(rprime)+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*np.log(rprime)**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*np.log(rprime)+ aa2 + Nc*(Nc-2)/2*((np.pi)**4-12*(np.pi)**2) ) )
                             else:
                                 V += -1*VB/rabs*(1 + alpha/(4*np.pi)*(2*beta0*np.log(rprime)+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*np.log(rprime)**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*np.log(rprime)+ aa2 ) )
            V_Psi_nlm_s[i] = V * wvfn[i]
        return V_Psi_nlm_s

    def K_Psi_nlm(Rs, A, C, nabla_psi_fn):
        K_psi = -1/2*nabla_total_Psi_nlm(Rs, A, C, nabla_psi_fn)
        return K_psi


    def V_Psi_nlm(Rs, A, C, psi_fn):
        V_psi = potential_total_Psi_nlm(Rs, A, C, psi_fn)
        return V_psi
    atan2
    def hammy_Psi_nlm(Rs, A, C, psi_fn, nabla_psi_fn):
        K_psi = K_Psi_nlm(Rs, A, C, nabla_psi_fn)
        V_psi = V_Psi_nlm(Rs, A, C, psi_fn)
        H_psi = K_psi + V_psi
        return H_psi


    ####################################################################################
    #######################################################################################
    def draw_coordinates(shape, *, eps=1.0, axis=1):
        dR = eps/np.sqrt(2) * torch.normal(torch.ones(shape))
        # subtract mean to keep center of mass fixed
        dR -= torch.mean(dR, axis=axis, keepdims=True)
        return dR

    def metropolis_coordinate_ensemble(this_psi, *, n_therm, N_walkers, n_skip, eps):
        # array of walkers to be generated
        Rs = torch.zeros((N_walkers, N_coord, 3))
        psi2s = torch.zeros((N_walkers))
        this_walker = 0
        # store acceptance ratio
        acc = 0
        # initial condition to start metropolis
        R = torch.normal(torch.ones((1,N_coord,3)))
        # set center of mass position to 0
        R -= torch.mean(R, axis=1, keepdims=True)
        # metropolis updates
        print("Running Metropolis")
        for i in tqdm.tqdm(range(-n_therm, N_walkers*n_skip)):
            # update
            dR = draw_coordinates(R.shape, eps=eps, axis=1)
            new_R = R + dR
            # accept/reject based on |psi(R)|^2
            abspsi = torch.abs(this_psi(R))
            p_R = abspsi**2
            abspsi_new = torch.abs(this_psi(new_R))
            p_new_R = abspsi_new**2
            if (torch.rand(1) < (p_new_R / p_R) and not torch.isnan(p_new_R) and p_new_R > 0 and p_new_R < 1 ):
                R = new_R #accept
                p_R = p_new_R
                if i >= 0:
                    acc += 1
            # store walker every skip updates
            if i >= 0 and (i+1) % n_skip == 0:
                Rs[this_walker,:,:] = R
                psi2s[this_walker] = p_R
                this_walker += 1
                #print(f'iteration {i+1}')
                #print(f'|psi(R)|^2 = {p_R}')
                #print(f'Total acc frac = {acc / (i+1)}')
        print(f'Total acc frac = {acc / (i+1)}')
        # return coordinates R and respective |psi(R)|^2
        return Rs, psi2s

    class wvfn(nn.Module):
        # Psi(R) = sum_{n,l,m} c_{n,l,m} psi(n, l, m, R)
        def __init__(self):
            super(wvfn, self).__init__()
            # register Bohr radius a and c_{n,l,m,k,j} as pytorch paramters
            self.A = nn.Parameter(2/VB*torch.ones(nExp, dtype=torch.double))
            self.C = nn.Parameter(torch.cat((
                torch.ones((1), dtype=torch.complex64),
                0*torch.ones((nExp-1), dtype=torch.complex64))))
            #self.A = nn.Parameter(2/VB*torch.ones(1, dtype=torch.double))
            #self.C = nn.Parameter(torch.ones(1, dtype=torch.complex64))
        # For N_coord>1 C and A have Length N_coord not 1
        def psi(self, Rs):
            A_n=self.A
            C_n=self.C
            psi = total_Psi_nlm(Rs, A_n, C_n, psitab)
            return psi
        def psi2(self, Rs):
            return torch.pow(torch.abs(self.psi(Rs)), 2)
        def laplacian(self, Rs):
            A_n=self.A
            C_n=self.C
            return nabla_total_Psi_nlm(Rs, A_n, C_n, nabla_psitab)

        def coulPot(self, Rs):
            A_n=self.A
            C_n=self.C
            return potential_no_Psi_nlm(Rs, A_n, C_n, psitab)

        def hammy(self, Rs):
            A_n=self.A
            C_n=self.C
            H_psi = hammy_Psi_nlm(Rs, A_n, C_n, psitab, nabla_psitab)
            psistar = torch.conj(total_Psi_nlm(Rs, A_n, C_n, psitab))
            return psistar*H_psi
        def forward(self, Rs):
            A_n=self.A
            C_n=self.C
            H_psi = hammy_Psi_nlm(Rs, A_n, C_n, psitab, nabla_psitab)
            psistar = torch.conj(total_Psi_nlm(Rs, A_n, C_n, psitab))
            return psistar*H_psi / VB**2, torch.pow(torch.abs(psistar), 2)


    def loss_function(wvfn, Rs):
        # <psi|H|psi> / <psi|psi>
        hammy, psi2s = wvfn(Rs)
        N_walkers = len(hammy)
        E_trial = torch.mean(torch.real(hammy/psi2s))
        noise_E_trial = torch.sqrt(torch.var(torch.real(hammy/psi2s)))/np.sqrt(N_walkers)
        print(f'<psi|H|psi>/<psi|psi> = {E_trial} +/- {noise_E_trial}')
        #loss = E_trial + np.sqrt(N_walkers)*noise_E_trial
        loss = c_loss*E_trial + v_loss*np.sqrt(N_walkers)*noise_E_trial
        return loss

    def fast_loss_function(wvfn, Rs, psi2s0):
        # <psi|H|psi> / <psi|psi>
        # quickly by reweighting using a Monte Carlo distribution proportional to
        # |psi_0|^2 instead of generating one proportional to |psi|^2
        hammy, psi2s = wvfn(Rs)
        N_walkers = len(hammy)
        E_trial = torch.mean(torch.real(hammy/psi2s0)) / torch.mean(psi2s/psi2s0)
        noise_E_trial = torch.abs( torch.mean(torch.real(hammy/psi2s0)) / torch.mean(psi2s/psi2s0) ) * torch.sqrt( torch.var(torch.real(hammy/psi2s0))/torch.mean( torch.real(hammy/psi2s0) )**2 + torch.var(psi2s/psi2s0)/torch.mean( psi2s/psi2s0 )**2 ) / np.sqrt(N_walkers)
        print(f'1/V^2 <psi|H|psi>/<psi|psi> = {E_trial} +/- {noise_E_trial}')
        loss = E_trial + np.sqrt(N_walkers)*noise_E_trial
        return loss

    def train_variational_wvfn(wvfn):
        print("\nStarting training")
        optimizer.zero_grad()
        # train net
        train_time = time.time()
        max_reduces = 0
        reduces = 0
        best_loss = 1e10
        best_iter = 0
        best_wvfn_state = copy.deepcopy(wvfn.state_dict())
        for n in tqdm.tqdm(range(N_train)):
            sep_time = time.time()
            epsilon=1.0/np.sqrt(VB)
            if n % N_refresh_metropolis == 0:
                print("\nRefreshing walkers")
                Rs, psi2s = metropolis_coordinate_ensemble(wvfn.psi, n_therm=500, N_walkers=N_walkers, n_skip=N_skip, eps=epsilon)
                loss = loss_function(wvfn, Rs)
                print("\nCalculating gradients")
                loss.backward()
                print("\nAdvancing optimizer")
                optimizer.step()
                scheduler.step(loss)
                if loss < best_loss:
                    best_iter = n
                    best_loss = loss
                    best_wvfn_state = copy.deepcopy(wvfn.state_dict())
            else:
                loss = fast_loss_function(wvfn, Rs, psi2s)
                print("\nCalculating gradients")
                loss.backward(retain_graph=True)
                print("\nAdvancing optimizer")
                optimizer.step()
                scheduler.step(loss)
                if loss < best_loss:
                    best_iter = n
                    best_loss = loss
                    best_wvfn_state = copy.deepcopy(wvfn.state_dict())
            tqdm.tqdm.write(f"\nTraining step {n}")
            tqdm.tqdm.write(f"loss function curently {loss}")
            for name, param in wvfn.named_parameters():
                if param.requires_grad:
                    print(name, param.data)
            lr = optimizer.param_groups[0]['lr']
            print(f"learn rate = {lr}")
            print(f"bad epochs = {scheduler.num_bad_epochs}")
            if (lr / 10**(-log10_learn_rate)) < 10**(-1*(max_reduces+training_round+.5)):
                print(f"reduced learn rate {max_reduces} times, quitting")
                break
        print(f"completed {N_train} steps of training in {time.time() - train_time} sec")
        print(f"best iteration {best_iter}")
        print(f"best loss function {best_loss} \n\n")
        wvfn.load_state_dict(best_wvfn_state)
        return best_loss, wvfn

    def diagnostics():
        print("Running positronium diagnostics")

        A_n=torch.ones((3));
        C_n=torch.ones((3));
        B_n=VB
        A_n[:]=2/B_n
        C_n[:]=1.0 #(1/A_n)**(3/2)


        def psi0(Rs):
            return total_Psi_nlm(Rs, A_n, C_n, psitab)

        Rs, psi2s0 = metropolis_coordinate_ensemble(psi0, n_therm=500, N_walkers=N_walkers, n_skip=N_skip, eps=A_n[0]/N_coord**2)

        print(Rs.shape)
        print(A_n)
        print(B_n)
        print(C_n)

        print(f"R = {Rs[0]}")
        print(f"psi0 = {psi0(Rs)[0]}")
        print(f"|psi0|^2 = {np.conjugate(psi0(Rs)[0])*psi0(Rs)[0]}")
        hammy_ME = np.conjugate(psi0(Rs))*hammy_Psi_nlm(Rs, A_n, C_n, psitab, nabla_psitab)
        print(f"|psi|^2 = ", psi2s0[0])
        print(f"<psi|H|psi>/|psi|^2 = {hammy_ME[0]/psi2s0[0]}")
        V_ME = np.conjugate(psi0(Rs))*V_Psi_nlm(Rs, A_n, C_n, psitab)
        print(f"<psi|V|psi>/|psi|^2 = {V_ME[0]/psi2s0[0]}")
        K_ME = np.conjugate(psi0(Rs))*K_Psi_nlm(Rs, A_n, C_n, nabla_psitab)
        print(f"<psi|K|psi>/|psi|^2 = {K_ME[0]/psi2s0[0]}")

        print(f"R = {Rs[1]}")
        print(f"psi0 = {psi0(Rs)[1]}")
        print(f"|psi0|^2 = {np.conjugate(psi0(Rs)[1])*psi0(Rs)[1]}")
        hammy_ME = np.conjugate(psi0(Rs))*hammy_Psi_nlm(Rs, A_n, C_n, psitab, nabla_psitab)
        print(f"|psi|^2 = ", psi2s0[1])
        print(f"<psi|H|psi>/|psi|^2 = {hammy_ME[1]/psi2s0[1]}")
        V_ME = np.conjugate(psi0(Rs))*V_Psi_nlm(Rs,A_n, C_n, psitab)
        print(f"<psi|V|psi>/|psi|^2 = {V_ME[1]/psi2s0[1]}")
        K_ME = np.conjugate(psi0(Rs))*K_Psi_nlm(Rs, A_n, C_n, nabla_psitab)
        print(f"<psi|K|psi>/|psi|^2 = {K_ME[1]/psi2s0[1]}")


        print(f'|psi|^2 = {psi2s0}')

        E0 = hammy_ME / psi2s0

        print(f'\nEvery element should be E0=-1/4, {E0} \n')
        print(f'<psi|H|psi>/<psi|psi> = {torch.mean(hammy_ME/psi2s0)} +/- {torch.sqrt(torch.var(hammy_ME/psi2s0))/np.sqrt(N_walkers)} = -1/4?')
        print(f'<psi|V|psi>/<psi|psi> = {torch.mean(V_ME/psi2s0)} +/- {torch.sqrt(torch.var(V_ME/psi2s0))/np.sqrt(N_walkers)}')
        print(f'<psi|K|psi>/<psi|psi> = {torch.mean(K_ME/psi2s0)} +/- {torch.sqrt(torch.var(K_ME/psi2s0))/np.sqrt(N_walkers)}')

        print("\n")

    filename = output + "exp" + str(N_exp) + "_Ncoord" + str(N_coord) + "_cutoff" + str(cutoff) + "_order" + str(OLO) + "_alpha" + str(alpha) + "_c_loss" + str(c_loss) + "_v_loss"  + f"_potential{VB:.3f}.wvfn"
    print("saving wvfn results to "+filename+"\n")
    if os.path.exists(filename):
        print("Error - remove existing wavefunction, torch save doesn't overwrite\n\n")
        sys.exit()



        diagnostics()

    # initialize wvfn
    trial_wvfn = wvfn()
    for name, param in trial_wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    # initialize optimizer
    optimizer = optim.Adam(trial_wvfn.parameters(), lr=10**(-log10_learn_rate))
    N_patience = patience_factor*N_refresh_metropolis
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=N_patience, threshold=0.00001, threshold_mode='abs', verbose=True)

    # train
    training_round = 0
    best_loss, trial_wvfn = train_variational_wvfn(trial_wvfn)

    epsilon=1.0/np.sqrt(VB)

    # print results
    print(f'Wavefunction results:')
    for name, param in trial_wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    Rs, psi2s = metropolis_coordinate_ensemble(trial_wvfn.psi, n_therm=500, N_walkers=N_walkers, n_skip=N_skip, eps=epsilon)
    hammy, psi2s = trial_wvfn(Rs)
    E_trial = torch.mean(hammy/psi2s)
    noise_E_trial = torch.sqrt(torch.var(hammy/psi2s))/np.sqrt(N_walkers)
    print(f'\n\n1/V^2 <psi|H|psi>/<psi|psi> = {E_trial} +/- {noise_E_trial}')

    print(f"\n\n Round two!")
    # initialize optimizer
    optimizer = optim.Adam(trial_wvfn.parameters(), lr=10**(-log10_learn_rate-1))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=N_patience, threshold=0.00001, threshold_mode='abs', verbose=True)

    # train
    training_round += 1
    best_loss, trial_wvfn = train_variational_wvfn(trial_wvfn)

    # print results
    print(f'Wavefunction results:')
    for name, param in trial_wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    Rs, psi2s = metropolis_coordinate_ensemble(trial_wvfn.psi, n_therm=500, N_walkers=N_walkers, n_skip=N_skip, eps=epsilon)
    hammy, psi2s = trial_wvfn(Rs)
    E_trial = torch.mean(hammy/psi2s)
    noise_E_trial = torch.sqrt(torch.var(hammy/psi2s))/np.sqrt(N_walkers)
    print(f'\n\n1/V^2 <psi|H|psi>/<psi|psi> = {E_trial} +/- {noise_E_trial}')

    print(f"\n\n Round three!")
    # initialize optimizer
    optimizer = optim.Adam(trial_wvfn.parameters(), lr=10**(-log10_learn_rate-2))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=N_patience, threshold=0.00001, threshold_mode='abs', verbose=True)

    # train
    training_round += 1
    best_loss, trial_wvfn = train_variational_wvfn(trial_wvfn)

    # print results
    print(f'Wavefunction results:')
    for name, param in trial_wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    Rs, psi2s = metropolis_coordinate_ensemble(trial_wvfn.psi, n_therm=500, N_walkers=N_walkers, n_skip=N_skip, eps=epsilon)
    hammy, psi2s = trial_wvfn(Rs)
    E_trial = torch.mean(hammy/psi2s)
    noise_E_trial = torch.sqrt(torch.var(hammy/psi2s))/np.sqrt(N_walkers)
    print(f'\n\n1/V^2 <psi|H|psi>/<psi|psi> = {E_trial} +/- {noise_E_trial}')

    # save best wvfn
    print(f'\n\nSaving best wavefunction to '+filename)
    torch.save(trial_wvfn.state_dict(), filename)

    new_wvfn = wvfn()
    new_dict = torch.load(filename)
    new_wvfn.load_state_dict(new_dict)
    print(f'\nVerifying saved wavefunction results:')
    for name, param in new_wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    Rs, psi2s = metropolis_coordinate_ensemble(new_wvfn.psi, n_therm=500, N_walkers=N_walkers, n_skip=N_skip, eps=epsilon)
    hammy, psi2s = new_wvfn(Rs)
    E_trial = torch.mean(hammy/psi2s)
    noise_E_trial = torch.sqrt(torch.var(hammy/psi2s))/np.sqrt(N_walkers)
    print(f'\n\n1/V^2 <psi|H|psi>/<psi|psi> = {E_trial} +/- {noise_E_trial} \n\n')

    filenamec = "exp" + str(N_exp) + "_Ncoord" + str(N_coord) + "_cutoff" + str(cutoff) + "_order" + str(OLO) + "_alpha" + str(alpha) + "_c_loss" + str(c_loss) + "_v_loss" + str(v_loss)
    nms = [[E_trial], [noise_E_trial]]
    f = open(filenamec+'.csv', 'w')
    with f:
      writer = csv.writer(f)
      writer.writerows(nms)
