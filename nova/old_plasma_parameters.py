    def min_r(self,xo):
        zo = xo[0]
        ro = xo[1]  # major radius
        A = xo[2]  # aspect ratio
        k = xo[3]  # elongation
        delta = xo[4]  # triangularity
    
        theta = np.linspace(0,2*np.pi,100)
        R,Z = self.elipsoid(theta,ro,zo,A,k,delta)
        err = 0
        for i in range(np.shape(self.Rp)[0]):
            err += np.min((R-self.Rp[i])**2+(Z-self.Zp[i])**2)
        return err
        
    def dot_r(self,xo):
        zo = xo[0]
        ro = xo[1]  # major radius
        A = xo[2]  # aspect ratio
        k = xo[3]  # elongation
        delta = xo[4]  # triangularity
    
        theta = np.linspace(0,2*np.pi,100)
        R,Z = self.elipsoid(theta,ro,zo,A,k,delta)
        nR,nZ = self.normal(R,Z)
        dsum = 0
        for r,z in zip(self.Rp,self.Zp):
            i = np.argmin((r-R)**2+(z-Z)**2)
            dr = [R[i]-r,Z[i]-z]  
            dn = [nR[i],nZ[i]]
            dot = np.dot(dr,dn)
            if dot < 0:
                dsum -= (dr[0]**2+dr[1]**2)
        return dsum