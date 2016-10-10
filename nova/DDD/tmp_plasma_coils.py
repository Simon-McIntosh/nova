    '''    
    def plasma_coils(self,N=5,dL=0.25):
        alpha = np.linspace(1/(2*N),1-1/(2*N),N)
        I,fact = np.zeros(N),1
        ffprim = interp1(np.linspace(0,1,len(self.eq['ffprim'])),
                         self.eq['ffprim'])
        for j in range(2):
            for i,a in enumerate(alpha):
                r,z = self.get_boundary(alpha=a)
                L = self.length(r,z,norm=False)
                #I[i] = L[-1]*fact*2*sf.eq['cpasma']/N*(1-a**2)
                #I[i] = L[-1]*fact*sf.eq['cpasma']/N
                I[i] = -L[-1]*fact*self.eq['cpasma']*ffprim(a)
            fact = -self.eq['cpasma']/np.sum(I) 
        for i,a in enumerate(alpha): 
            r,z = self.get_boundary(alpha=a)
            L = self.length(r,z,norm=False)
            Lp = np.arange(0,L[-1],dL)[:-1]
            rp = interp1(L,r)(Lp)
            zp = interp1(L,z)(Lp)
            Ic = I[i]/len(Lp)
            for j,(rpc,zpc) in enumerate(zip(rp,zp)):
                self.coil['plasma_'+str(i)+'_'+str(j)] = \
                {'I':Ic,'r':rpc,'z':zpc,'dr':dL,'dz':dL}
    '''  
    def coil_core_old(self):
        for name in self.coil.keys():
            r,z = self.coil[name]['r'],self.coil[name]['z']
            dr,dz = self.coil[name]['dr'],self.coil[name]['dz']
            I = self.coil[name]['I']
            if self.ingrid(r,z):
                io = np.argmin(np.abs(r-dr/2-self.r))
                i1 = np.argmin(np.abs(r+dr/2-self.r))
                jo = np.argmin(np.abs(z-dz/2-self.z))
                j1 = np.argmin(np.abs(z+dz/2-self.z))
                irange = np.arange(io,i1+1)
                jrange = np.arange(jo,j1+1)
                ni,nj = len(irange),len(jrange)
                n = ni*nj
                dI = I/n
                print(n,dI,self.dA,self.r[irange])
                b = np.zeros((ni,nj))
                for j in range(nj):
                    b[:,j] += -mu_o*dI*self.r[irange]/(self.dA) 
                for i in range(ni):
                    b[i,:] = np.mean(b[i,:])*np.ones(nj)
                for j,jr in enumerate(jrange):    
                    self.b[self.indx(irange,jr)] += b[:,j]
                    
                    
                        r = 0
        for i in range(self.Nplasma):
            indx = self.plasma_index[i]
            i,j = self.ij(indx)   
            r += self.r[i]
        r_bar = r/self.Nplasma
        print(r_bar)

        for i in range(self.Nplasma):
            indx = self.plasma_index[i]
            i,j = self.ij(indx)
            self.b[indx] *= (scale_plasma*self.r[i]/r_bar)
            
            
            ****
            
            '''
sf.sol()
sf.get_legs()
targets = conf.targets
sol = solCalc(sf,flip=1,targets=conf.targets)
#L2D,L3D = sol.connection('outer',0)
        
R,Z = sf.legs['outer']['R'][0],sf.legs['outer']['Z'][0]
dRsol = np.diff(R)
dZsol = np.diff(Z)
L2D = np.append(0,np.cumsum(np.sqrt(dRsol**2+dZsol**2)))
      
i = np.argmin(abs(L2D-2.3))  
   
'''   

'''
sol = solCalc(sf,flip=1,targets=conf.targets)  
Ro,Zo = rb.targets['outer']['Rsol'][-1],rb.targets['outer']['Zsol'][-1]
print(Ro,Zo,sf.Xpsi)
Xi = sol.expansion([Ro],[Zo])
theta = sol.strike_point(Xi,rb.targets['outer']['graze'])

print(theta*180/np.pi)

pl.plot(Ro,Zo,'o',markersize=3)


graze = np.arcsin(np.sin(83.8488627351*np.pi/180)*(Xi[-1]**2+1)**-0.5)    
            
print(graze*180/np.pi)
'''


#inv.move_coil(5,point=(6.67,8.98))
#inv.move_coil(6,point=(14.10,8.35))
#inv.move_coil(7,point=(17.17,3.90))
#inv.move_coil(8,point=(17.51,-1.29))
#inv.move_coil(9,point=(14.31,-9.25))
#inv.move_coil(10,point=(9.47,-12.02))
#inv.add_coil(15.82,-7.51,1,1)
#inv.add_coil(4.5,-10.5,1,1)


#inv.move_coil(10,delta=(10,0.1,0))

#inv.move_coil(12,delta=(12,3,-1))
#inv.move_coil(10,delta=(10,1,0))
#inv.move_coil(9,delta=(10,2,0.5))
#inv.remove_coil([9])

'''
inv.rb.get_legs()
Np = [3,6]
for leg,npt in zip(inv.rb.targets.keys(),Np):
    R,Z = inv.rb.targets[leg]['Rsol'],inv.rb.targets[leg]['Zsol']
    L = sf.length(R,Z)
    Lp = np.linspace(1/(2*npt),1-1/(2*npt),npt)
    Rp,Zp = interp1(L,R)(Lp),interp1(L,Z)(Lp)
    for r,z in zip(Rp,Zp):
        inv.add_alpha(1,factor=1,point=(r,z))
'''


    def current_adjust_old(self,I,*args):
        if not hasattr(self,'T'):  # background update
            self.get_weight()
            self.set_background()
        self.I = 1e6*I.reshape(len(I),1)
        self.update_current(self.I)
        self.gain_weight()
        self.get_LSresidual()
        if args[0] == 'RMS':
            return args[1]-self.LSresidual
        else:
            return self.Itotal*1e-8
            
    def current_opp_old(self,LStarget=0.05,Ilim=[None,None],disp=True):
        constraint = {'type':'ineq','fun':self.current_adjust,
                      'args':['RMS',LStarget]}
        self.solve(BG=True)
        Io = self.I*1e-8
        bounds = np.array([Ilim]*len(Io))  # MA bounds
        result = op.minimize(self.current_adjust,Io,bounds=bounds,
                           tol=1e-6,args=('I'),constraints=constraint,
                           options={'disp':disp,'maxiter':2e4},  # ,'eps':1e-4
                           method='SLSQP')
        self.I = result.x
        return result.success
        
        
        
    def position_sead(self,Ncoil=6,Isum=100,**kwargs):
        self.LSresidual = 0
        self.get_weight()
        self.set_background()
        self.grid_PF(Ncoil=Ncoil)
        
        constraint = {'type':'ineq','fun':self.position_adjust,
                      'args':['Isum',Isum]}
        bounds = np.array([[0,1]]*Ncoil)
        if 'Lo' in kwargs.keys():
            Lcoilo = kwargs['Lo']
        else:
            Lcoilo = self.Lo
        '''
        self.Lopp = op.minimize(self.position_adjust,Lcoilo,bounds=bounds,
                           tol=2e-3,constraints=constraint,
                           args=('LS'),
                           options={'disp':True,'maxiter':1e2},#,'eps':1e-3
                           method='SLSQP').x
        
        method = 'Nelder-Mead'
        options = {'disp':True,'maxiter':1e2,'xtol':0.01}
        self.Lopp = op.minimize(self.position_adjust,Lcoilo,args=('LS'),
                           options=options,method=method).x
        '''                   
        method = 'COBYLA'
        self.Lopp = op.minimize(self.position_adjust,Lcoilo,
                           tol=2e-3,  # constraints=constraint,
                           args=('LS'),
                           options={'disp':True,'maxiter':1e2},#,'eps':1e-3
                           method=method).x
        return self.Lopp
        
    def position_opp(self,Ncoil=6,LStarget=0.05,Ilim=[None,None],**kwargs):
        self.LSresidual = 0
        self.get_weight()
        self.set_background()
        self.grid_PF(Ncoil=Ncoil)
        
        constraint = {'type':'ineq','fun':self.position_adjust,
                      'args':['RMS',LStarget,Ilim]}
        bounds = np.array([[0,1]]*Ncoil)
        if 'Lo' in kwargs.keys():
            Lcoilo = kwargs['Lo']
        else:
            Lcoilo = self.Lo

        self.Lopp = op.minimize(self.position_adjust,Lcoilo,bounds=bounds,
                           tol=2e-3,constraints=constraint,
                           args=('I',LStarget,Ilim),
                           options={'disp':True,'maxiter':1e2},  # ,'eps':1e-4
                           method='SLSQP').x
        print(self.Lopp)

        '''
        Inew = op.minimize(self.update_current,Io,
                                options={'disp':True,'maxiter':10,'eps':1e5},
                                method='SLSQP',constraints=constraint).x
        ll = -75e5*np.ones([len(self.adj_coils),1])
        ul = +75e5*np.ones([len(self.adj_coils),1])    
        from pymls import bounded_lsq as blsq
        I = blsq(G,Tpsi,ll,ul)
        self.update_current(I)
        '''
        '''
        constraint = {'type':'ineq','fun':self.psi_err}
        #bounds = [(0,None),(1,None),(1,None),  # (0.7,None),(0.1,None)
        #              (self.TFbound['ro_min'],None),(None,None)]  # 
        Io = np.copy(I).reshape((1,len(I)))
        Inew = op.minimize(self.update_current,Io,
                                options={'disp':True,'maxiter':10,'eps':1e5},
                                method='SLSQP',constraints=constraint).x
        print(Inew)
        self.update_current(Inew)
        '''
        
            '''
    def psi_err(self,Lcoil,*args):
        ineq = 0
        LStarget = args[0]
        self.coil_adjust(Lcoil,[LStarget])
        LSineq = LStarget-self.LSresidual
        if LSineq < 0:  # limit residual
            ineq += LSineq
        #for i in range(len(Lcoil)-1):  # order coils
        #    if Lcoil[i] > Lcoil[i+1]:
        #        ineq += Lcoil[i+1]-Lcoil[i]
        return ineq
        '''
        
            def position_adjust(self,Lcoil,*args):
        for name,L in zip(self.coil['active'].keys(),Lcoil):
            if L<0: L=0
            if L>1: L=1
            r,z = self.TFoutR(L),self.TFoutZ(L)
            ref = int(name.split('Coil')[-1])
            self.move_coil(ref,point=(r,z))
            #dr,dz = self.rb.Cshift(name,'out',0)
            #self.move_coil(ref,delta=(ref,dr,dz))  # move outside TF
        self.update_coils()
        self.get_weight()
        self.solve()
        if args[0] != 'Isum' and args[0] != 'LS':
            Itotal,LStarget = self.Itotal,self.LSresidual
            if self.LSresidual < args[1]-1e-3:
                LStarget = args[1]
            else:
                LStarget = self.LSresidual
            success = self.current_opp(LStarget=args[1]-1e-3,
                                          Ilim=args[2],disp=True)
            if not success: 
                self.Itotal,self.LSresidual = Itotal,LStarget
        '''
        elif args[0] == 'Isum':
            Itotal,LStarget = self.Itotal,self.LSresidual
            success = self.current_opp(LStarget=LStarget,disp=False)
            if not success: 
                self.Itotal = Itotal
            self.LSresidual = LStarget
        '''
        if args[0] == 'RMS':
            val = args[1]-self.LSresidual
            print(args[1],self.LSresidual)
        elif args[0] == 'I':
            val = self.Itotal*1e-6   
        elif args[0] == 'LS':
            val = self.LSresidual
            #print('LS',val)
        elif args[0] == 'Isum':
            val = args[1]-self.Itotal*1e-6
            print('Isum',val,self.LSresidual)
        return val

                def update_current_new(self,I):
        self.Itotal = 0
        for j,name in enumerate(self.coil['active'].keys()):
            if np.sum(abs(I))>0:
                Nfilament = self.eq.coil[name+'_0']['Nf']
                Icoil = I[j][0]*Nfilament  # update sf  I[j][0]
                self.sf.coil[name]['I'] = Icoil
                dA = Icoil/12.5e6  # apply current density limit
                scale = dA/(self.sf.coil[name]['dr']*self.sf.coil[name]['dz'])
                self.sf.coil[name]['dr'] *= scale
                self.sf.coil[name]['dz'] *= scale
            
            #self.eq.coils(delta=self.eq.delta)  # update eq (re-grid)  
            #self.update_coils()  # update inv
                rc,zc = self.sf.coil[name]['r'],self.sf.coil[name]['z']
                for i in range(Nfilament):
                    sub_name = name+'_{:1.0f}'.format(i)
                    self.eq.coil[sub_name]['I'] = I[j][0]  # update eq
                    self.eq.coil[sub_name]['dr'] *= scale
                    self.eq.coil[sub_name]['dz'] *= scale
                    r = self.eq.coil[sub_name]['r']
                    z = self.eq.coil[sub_name]['z']
                    dr,dz = r-rc,z-zc
                    self.eq.coil[sub_name]['r'] = rc+scale*dr
                    self.eq.coil[sub_name]['z'] = zc+scale*dz
                    self.coil['active'][name]['I'][i] = I[j][0]  # update inv
            self.Itotal += abs(self.sf.coil[name]['I'])  # sum current
            
            
                        '''
            for sign,leg in zip([-1,1],self.targets.keys()):
                gap = []
                ends = [0,-1][::-1*sign]
                psi = self.psi_fw
                if sign == -1: psi = Phi_target[1]#-0.4*psi
                pl.figure()  # dummy figure
                CS = pl.contour(self.sf.r, self.sf.z, self.sf.psi.T,levels=psi)
                pl.close()
                for p in CS.collections[0].get_paths():
                    r,z = p.vertices[:,0],p.vertices[:,1]
                    gap.append(np.min((self.targets['inner']['R'][ends[-1]]-r)**2+
                              (self.targets['inner']['Z'][ends[-1]]-z)**2))
                select = np.argmin(gap)
                p = CS.collections[0].get_paths()[select]
                r,z = p.vertices[:,0],p.vertices[:,1] 
                index = np.zeros(2)
                index[0] = np.argmin((self.targets['inner']['R'][ends[1]]-r)**2+
                         (self.targets['inner']['Z'][ends[1]]-z)**2)
                index[1] = np.argmin((self.targets['outer']['R'][ends[0]]-r)**2+
                         (self.targets['outer']['Z'][ends[0]]-z)**2)
                if index[0]>index[1]:
                    index = index[::-1]
                r,z = r[index[0]:index[1]+1],z[index[0]:index[1]+1]
                if sign*r[0]<sign*r[-1]:
                    r,z = r[::-1],z[::-1]
                Rb = np.append(Rb,self.targets[leg]['R'][1:])
                Zb = np.append(Zb,self.targets[leg]['Z'][1:])    
                Rb,Zb = np.append(Rb,r[1:]),np.append(Zb,z[1:])
            '''
            
                def current_adjust(self,I,*args):
        if not hasattr(self,'T'):  # background update
            self.get_weight()
            self.set_background()
        self.I = I.reshape(len(I),1)/args[1]
        self.update_current(self.I)
        self.set_foreground()
        self.gain_weight()
        self.get_LSresidual()
        mix = args[0]
        if args[2] == 'Itotal':
            val = (1-mix)*self.Itotal/50e6 + mix*self.LSresidual/0.05
        else:
            val = (1-mix)*self.Imax/20e6 + mix*self.LSresidual/0.05
        '''    
        result = op.minimize(self.current_adjust,self.Io,tol=1e-5,
                           args=(mix,Inorm,'Itotal'),
                           options={'disp':disp,'maxiter':1e3},  # 
                           method='SLSQP')  # 'SLSQP'
        '''
        return val
        
        
        
    def find_position(self,mix,Lo):
        result = op.minimize(self.find_current,Lo,tol=1e-2,
                           args=(mix),  # tol=1e-1,,'maxiter':5e2
                           options={'disp':True},  
                           method='Nelder-Mead')  # 'Nelder-Mead'
        return result.x,result.fun
        
        
                if not hasattr(self,'Xpsi'):
            self.get_Xpsi()
        if not hasattr(self,'Mpsi'):
            self.get_Mpsi()
            
            
                '''    
    def midplane_contour(self,C,closed=False):
        p = C.collections[0].get_paths()  # plasma contour
        N = len(p)
        r,z = np.array([]),np.array([])
        for i in range(N):
            v = p[i].vertices
            zo = v[:,1]
            if closed:
                ends = np.sqrt((v[0,0]-v[-1,0])**2+(v[0,1]-v[-1,1])**2) < 1
            else:
                ends = True
            if (zo>self.Mpoint[1]).any() and \
            (zo<self.Mpoint[1]).any() and ends:
                v = p[i].vertices
                r = np.append(r,v[:,0])
                z = np.append(z,v[:,1])
        return (r,z)

        pl.figure()
        cs = pl.contour(self.r,self.z,psi_b.T,levels=[psi_edge],colors='k')
        pl.close()
        r,z = self.midplane_contour(cs,closed=True)
        return r,z
    '''
    
    
            '''
        if not hasattr(self, 'Xpsi') or xo is not None:
            self.get_Xpsi(xo=xo)
            
        self.get_Mpsi()
        '''
        
        '''
        psi = alpha*(self.Xpsi-self.Mpsi)+self.Mpsi
        pl.figure()  # dummy figure
        C=pl.contour(self.r,self.z,self.psi.T,levels=[psi],colors='b')
        pl.close()
        r,z = self.midplane_contour(C,closed=True)
        '''
        
        
            '''
    def feildlines(self,psi_line):
        alpha = np.array([1,0.5])
        lw = 0.75
        if not Plasma: norm = 0
        if color == 'k': alpha *= 0.25
        for psi in psi_line:
            r,z = psi[0],psi[1]
            if self.inPlasma(r,z):
                pl.plot(r,z,linetype,linewidth=1.25*lw,
                        color=norm*np.array([1,1,1]),alpha=alpha[0])  
            else:
                pl.plot(r,z,linetype,linewidth=lw,color=color,alpha=alpha[1])

    def feildlines(self,levels,norm,Plasma=False,color='k',
                   pcolor='w',linetype='-'):
        alpha,lw = np.array([1,0.5]),0.75
        #if not Plasma: norm = 0
        if color == 'k': alpha *= 0.25
        c = cntr.Cntr(self.r,self.z,self.psi.T-self.Xpsi)
        for level in levels:
            psi_line = c.trace(level,level,0)
            psi_line = psi_line[:len(psi_line)//2]
            for psi in psi_line:
                r,z = psi[0],psi[1]
                pl.plot(r,z,linetype,linewidth=lw,color=color,alpha=alpha[1])
    '''
    
    
            
        '''
            if Plasma:
                norm = 0
            else:
                norm = level/(self.Mpsi-self.Xpsi)
        '''
            
        '''
        pl.figure()  # dummy figure
        CS = pl.contour(self.r, self.z, self.psi.T-self.Xpsi, 
                        levels=levels)  # 
        pl.close()
        self.cs = CS  # store contour levels
                       
        for cs,norm in zip(CS.collections,levels/(self.Mpsi-self.Xpsi)):
            self.plot_cs(cs,norm,Plasma=Plasma,color=color,linetype=linetype)
        return CS.levels
        '''
        
                for i in range(len(cs.collections)):
            for j in range(len(cs.collections[i].get_paths())):
                p = cs.collections[i].get_paths()[j]
                v = p.vertices
                R = v[:,0][:]
                Z = v[:,1][:]
                
                
                        #pl.figure()
        #nhist,bins,patches = pl.hist(self.tleg,bins=50)
        #pl.close()
                
                
                            #L2Dspace = np.linspace(L2D[0],L2D[-1],len(L2D))  # uniform interpolation
            #Rsol = interp1(L2D,Rsol,kind='linear')(L2Dspace)
            #Zsol = interp1(L2D,Zsol,kind='linear')(L2Dspace)
                
                
                
'''
for swing in np.linspace(-20,80,5):
    pl.figure()
    pl.axis('equal')
    pl.axis('off')


    inv.swing_fix(swing)
    inv.solve() 
    
    inv.update_coils(plot=True)
    sf.plot_coils(Color,coils=sf.coil,label=False,plasma=False,current=True) 
    sf.plot_coils(Color,coils=eq.coil,label=False,plasma=False) 
 
    eq.run()
    
    sf.contour()
    eq.plasma()
    #eq.plotb()
    #sf.eqwrite(config='SXex')
    pl.plot(sf.rbdry,sf.zbdry,'--')
    inv.plot_fix()
'''

'''
#inv.rb.sol.plot()
#sf.sol()
#Rsol,Zsol = inv.rb.sol.legs('outer')

from eqConfig import Config
conf = Config('SXex')

conf.TF(sf)
rb = RB(conf,sf,Np=100)
rb.divertor_outline(True,plot=True,debug=False)


rb.FWfill(dt=conf.tfw,loop=True,alpha=0.7,color=next(Color),s=2e-3)
rb.fill(dt=conf.BB[::-1],alpha=0.7,ref_o=0.3,dref=0.2,
        referance='length',color=next(Color))
rb.fill(dt=conf.tBBsupport,alpha=0.7,color=next(Color))
rb.BBsheild_fill(dt=conf.sheild,ref_o=0.35*np.pi,dref=0.2*np.pi,offset=1/10*np.pi,
                 alpha=0.7,color=next(Color))
rb.VVfill(dt=conf.VV,ref_o=0.25*np.pi,dref=0.25*np.pi,offset=0.5/10*np.pi,
          alpha=0.7,loop=True,color=next(Color))  # ref_o=0.385




'''
'''
print('L3D',inv.rb.sol.connection('outer',0)[-1][-1])
print('R',Rsol[-1])
print('R/X',Rsol[-1]/sf.Xpoint[0])
print('Itotal',inv.Itotal*1e-6,'MA')
print('R',rb.targets['outer']['Rsol'][-1],'Z',
      rb.targets['outer']['Zsol'][-1])
'''

'''
conf.TFopp = 'V'
rb.set_TFbound()  # TF boundary conditions
rb.TFbound['ro_min'] -= 0.5
#rb.plot_TFbounds()          
rb.TFopp(True,objF=conf.TFopp)  # L==length, V==volume
rb.TFfill()
'''

'''
pl.figure(figsize=(3.14,3.14*12/16))
pl.semilogy(I,LS,'o-',markersize=2.5)
pl.xlabel(r'current sum $I$ MA')
pl.ylabel('error m')

pl.figure(figsize=(3.14,3.14*12/16))
pl.plot(np.log10(Mix),Fun,'o-',markersize=2.5)

pl.figure(figsize=(3.14,3.14*12/16))
pl.plot(np.log10(Mix),I,'o-',markersize=2.5)

pl.figure(figsize=(3.14,3.14*12/16))
pl.plot(np.log10(Mix),LS,'o-',markersize=2.5)
'''

'''
for i in range(sf.nr):
    print(i,'of',sf.nr)
    for j in range(sf.nz):
        point = (sf.r2d[i,j],sf.z2d[i,j])
        feild = cc.Bfeild(eq.coil,eq.plasma_coil,point)
        sf.Br[i,j],sf.Bz[i,j] = feild[0],feild[1]
'''
print('Br',sf.Br.max(),'Bz',sf.Bz.max())
sf.Bsf()


            '''
            Nfilament = self.eq.coil[name+'_0']['Nf']
            F = np.zeros((2,Nfilament))
            for i,sub_name in enumerate(self.coil['active'][name]['sub_name']):
                F[:,i] = cc.Fcoil(self.eq.coil,self.eq.plasma_coil,sub_name)
            '''
            
                        F[:,0] = cc.Fcoil(self.sf.coil,self.eq.plasma_coil,name)
            '''
            R,Z = coil[name]['r'],coil[name]['z']
            Fr,Fz = coil[name]['Fr'],coil[name]['Fz']
            for r,z,fr,fz in zip(R,Z,Fr,Fz):
                pl.arrow(r,z,5*fr/self.F_max,5*fz/self.F_max,linewidth=1)
            '''
            
            
            
        i = np.argmin((Rin[-1]-Rloop)**2+(Zin[-1]-Zloop)**2)
        dl = np.sqrt((Rin[-1]-Rloop[i])**2+(Zin[-1]-Zloop[i])**2)
        #if dl < np.sqrt((Rin[-1]-Rin[-2])**2+(Zin[-1]-Zin[-2])**2):
        #    Rin,Zin = np.append(Rin,Rloop[i]),np.append(Zin,Zloop[i])