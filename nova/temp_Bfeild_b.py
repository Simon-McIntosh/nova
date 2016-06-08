    # temp
    def Bfeild_b(self):
        from cross_coil import Bcoil_b
        self.Br_b = np.zeros((self.nz,self.nr))
        self.Bz_b = np.zeros((self.nz,self.nr))
        for i in range(self.nz):
            for j in range(self.nr):
                B = Bcoil_b(self.coil, [self.rm[i,j],0,self.zm[i,j]])
                self.Br_b[i,j] = B[0]
                self.Bz_b[i,j] = B[2]
                
    def Bcontor_b(self, axis, Nstd=1.5, color='b'):
        var = 'B'+axis+'_b'
        if not hasattr(self, var):
            self.Bfeild_b()  
        B = getattr(self, var)
        
        level = [np.mean(B)-Nstd*np.std(B), 
                 np.mean(B)+Nstd*np.std(B)]
        level = [-5, 5]
        pl.contour(self.rm, self.zm, B, 
                   levels=np.linspace(level[0],level[1],30), colors=color) 
    # temp 