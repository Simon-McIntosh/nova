from pylab import * 
import math

'''def getForce(x,y):
    B = Bcoil(mu_o, coil, [x,0,y])
    B_mag = sum(B*B)**0.5
    return B[0]/B_mag,B[2]/B_mag

def equations(trv):
    x  = trv[0]; y  = trv[1]; vx = trv[2]; vy = trv[3];
    ax,ay = getForce(x,y)
    flow = np.array([ vx, vy, ax, ay ])
    return flow

def SimpleStep( x, dt, flow ):
    x += dt*flow(x)

def verletStep1( x, dt, flow ):
    ax,ay = getForce(x[0],x[1])
    vx   = x[2] + dt*ax; vy   = x[3] + dt*ay; 
    x[0]+= vx*dt;        x[1]+= vy*dt;
    x[2] = vx;        x[3] = vy;
    
def RK4_step(x, dt, flow):    # replaces x(t) by x(t + dt)
    k1 = dt * flow(x);     
    x_temp = x + k1 / 2;   k2 = dt * flow(x_temp)
    x_temp = x + k2 / 2;   k3 = dt * flow(x_temp)
    x_temp = x + k3    ;   k4 = dt * flow(x_temp)
    x += (k1 + 2*k2 + 2*k3 + k4) / 6
    
def RK4_adaptive_step(x, dt, flow, accuracy=1e-6):  # from Numerical Recipes
    SAFETY = 0.9; PGROW = -0.2; PSHRINK = -0.25;  ERRCON = 1.89E-4; TINY = 1.0E-30
    scale = flow(x)
    scale = abs(x) + abs(scale * dt) + TINY
    while True:
        x_half = x.copy();  RK4_step(x_half, dt/2, flow); RK4_step(x_half, dt/2, flow)
        x_full = x.copy();  RK4_step(x_full, dt  , flow)
        Delta = (x_half - x_full)
        error = max( abs(Delta[:] / scale[:]) ) / accuracy
        if error <= 1:
            break;
        dt_temp = SAFETY * dt * error**PSHRINK
        if dt >= 0:
            dt = max(dt_temp, 0.1 * dt)
        else:
            dt = min(dt_temp, 0.1 * dt)
        if abs(dt) == 0.0:
            raise OverflowError("step size underflow")
    if error > ERRCON:
        dt *= SAFETY * error**PGROW
    else:
        dt *= 5
    x[:] = x_half[:] + Delta[:] / 15
    return dt 

def integrate( trv0, dt, F, t_max, method='RK4', accuracy=1e-6 ):
    global ForceEvals
    ForceEvals = 0
    trv = trv0.copy()
    step = 0
    t = 0
    print("integrating with method: ",method," ... ")
    while True:
        if method=='RK4adaptive':
            dt = RK4_adaptive_step(trv, dt, equations, accuracy)
        elif method=='RK4':
            RK4_step(trv, dt, equations)
        elif method=='Euler':
            SimpleStep(trv, dt, equations)
        elif method=='Verlet':
            verletStep1(trv, dt, equations)
        F[:,step] = trv[:]
        step += 1
        t+=dt
        print(trv)
        if t > t_max:
            break
    print(" step = ", step)    
    print('F', len(F))
    
    
# ============ MAIN PROGRAM BODY =========================
dt       = 5
accuracy = 0.0001

trv0 = np.array([ 8, 3.535,    0, 0 ])             

def testMethod( trv0, dt, fT, n, method, style ):
    print(" ")
    F = np.zeros((4,n));
    integrate(trv0, dt, F, fT, method, accuracy);
    pl.plot(F[0],F[1], style ,label=method+" "+str(fT)); 

    print(len(F[0]))
testMethod( trv0, dt, 15, 200  , 'RK4', '-' )
#testMethod( trv0, dt, 0.1, 10  , 'RK4adaptive', 'o-' )
#testMethod( trv0, dt/4, 0.1, 10, 'Verlet', '-' )
#testMethod( trv0, dt/160, 0.00001, 1000, 'Euler', '-x' )


'''

G_m1_plus_m2 = 4 * math.pi**2

ForceEvals = 0

def getForce(x,y):
    global ForceEvals
    ForceEvals += 1
    r = math.sqrt( x**2 + y**2 )
    A = - G_m1_plus_m2 / r**3
    return x*A,y*A

def equations(trv):
    x  = trv[0]; y  = trv[1]; vx = trv[2]; vy = trv[3];
    ax,ay = getForce(x,y)
    flow = array([ vx, vy, ax, ay ])
    return flow

def SimpleStep( x, dt, flow ):
    x += dt*flow(x)

def verletStep1( x, dt, flow ):
    ax,ay = getForce(x[0],x[1])
    vx   = x[2] + dt*ax; vy   = x[3] + dt*ay; 
    x[0]+= vx*dt;        x[1]+= vy*dt;
    x[2] = vx;        x[3] = vy;

def RK4_step(x, dt, flow):    # replaces x(t) by x(t + dt)
    k1 = dt * flow(x);     
    x_temp = x + k1 / 2;   k2 = dt * flow(x_temp)
    x_temp = x + k2 / 2;   k3 = dt * flow(x_temp)
    x_temp = x + k3    ;   k4 = dt * flow(x_temp)
    x += (k1 + 2*k2 + 2*k3 + k4) / 6

def RK4_adaptive_step(x, dt, flow, accuracy=1e-6):  # from Numerical Recipes
    SAFETY = 0.9; PGROW = -0.2; PSHRINK = -0.25;  ERRCON = 1.89E-4; TINY = 1.0E-30
    scale = flow(x)
    scale = abs(x) + abs(scale * dt) + TINY
    while True:
        x_half = x.copy();  RK4_step(x_half, dt/2, flow); RK4_step(x_half, dt/2, flow)
        x_full = x.copy();  RK4_step(x_full, dt  , flow)
        Delta = (x_half - x_full)
        error = max( abs(Delta[:] / scale[:]) ) / accuracy
        if error <= 1:
            break;
        dt_temp = SAFETY * dt * error**PSHRINK
        if dt >= 0:
            dt = max(dt_temp, 0.1 * dt)
        else:
            dt = min(dt_temp, 0.1 * dt)
        if abs(dt) == 0.0:
            raise OverflowError("step size underflow")
    if error > ERRCON:
        dt *= SAFETY * error**PGROW
    else:
        dt *= 5
    x[:] = x_half[:] + Delta[:] / 15
    return dt    

def integrate( trv0, dt, F, t_max, method='RK4', accuracy=1e-6 ):
    global ForceEvals
    ForceEvals = 0
    trv = trv0.copy()
    step = 0
    t = 0
    print("integrating with method: ",method," ... ")
    while True:
        if method=='RK4adaptive':
            dt = RK4_adaptive_step(trv, dt, equations, accuracy)
        elif method=='RK4':
            RK4_step(trv, dt, equations)
        elif method=='Euler':
            SimpleStep(trv, dt, equations)
        elif method=='Verlet':
            verletStep1(trv, dt, equations)
        step += 1
        t+=dt
        F[:,step] = trv[:]
        if t > t_max:
            break
    print(" step = ", step)


# ============ MAIN PROGRAM BODY =========================

r_aphelion   = 1
eccentricity = 0.95
a = r_aphelion / (1 + eccentricity)
T = a**1.5
vy0 = math.sqrt(G_m1_plus_m2 * (2 / r_aphelion - 1 / a))
print(" Semimajor axis a = ", a, " AU")
print(" Period T = ", T, " yr")
print(" v_y(0) = ", vy0, " AU/yr")
dt       = 0.0003
accuracy = 0.0001

#                 x        y     vx  vy
trv0 = array([ r_aphelion, 0,    0, vy0 ])             

def testMethod( trv0, dt, fT, n, method, style ):
    print(" ")
    F = zeros((4,n));
    integrate(trv0, dt, F, T*fT, method, accuracy);
    print("Periods ",fT," ForceEvals ",  ForceEvals)
    plot(F[0],F[1], style ,label=method+" "+str(fT)+"T "+str(  ForceEvals ) ); 

testMethod( trv0, dt, 10, 20000  , 'RK4', '-' )
testMethod( trv0, dt, 10, 10000  , 'RK4adaptive', 'o-' )
testMethod( trv0, dt/4, 10, 100000, 'Verlet', '-' )
#testMethod( trv0, dt/160, 2, 1000000, 'Euler', '-' )

legend();
axis("equal")
savefig("kepler.png")
show();