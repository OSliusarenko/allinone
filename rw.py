import numpy as np
from scipy import special


class pdf_hist(object):
    """A handy class for managing numerical
    histograms for pdfs"""

    def __init__(self, x0, xn, cells, logcells=False):
        """Defines necessary variables and divides the x interval"""
        self.x0 = x0
        self.xn = xn
        self.cells = cells
        self.lc = logcells
        self.stat = 0
        self.normalized = False

        self.x = np.zeros(self.cells, dtype='float')  # averaged x (bin edges)
        self.b = np.zeros(self.cells+1, dtype='float')  # bin edges
        self.y = np.zeros(self.cells, dtype='float')  # distr
        self.cdfy = np.zeros(self.cells, dtype='float')  # CDF

        if self.lc:
            for i in range(len(self.b)):  # defining bins edges
                self.b[i] = self.x0*pow(self.xn/self.x0,
                                        float(i)/float(self.cells))
            for i in range(len(self.x)):  # defining averaged bins (x)
                self.x[i] = np.sqrt(self.b[i]*self.b[i+1])
        else:
            for i in range(len(self.b)):
                self.b[i] = self.x0 + (self.xn - self.x0) * \
                                        float(i) / float(self.cells)
            for i in range(len(self.x)):
                self.x[i] = (self.b[i]+self.b[i+1])/2.

    def what_cell(self, x):
        """Returns the histogram cell number where x should be"""
        if self.lc:
            return int(np.floor(float(self.cells)*np.log10(x/self.x0) /
                                np.log10(self.xn/self.x0)))
        else:
            return int(np.floor(float(self.cells)*(x-self.x0) /
                                (self.xn-self.x0)))

    def inc_y(self, x):
        """Increases by 1 the y cell where x is"""
        self.stat += 1
        if(x >= self.x0 and x < self.xn):
            self.y[self.what_cell(x)] += 1

    def normalize(self):
        """Normalizes by 1 the pdf; shifts x to the center of bin;
        calculates cdf"""
        # normalize distr
        if not self.normalized:
            for i in range(len(self.y)):
                self.y[i] /= float(self.stat)*(self.b[i+1]-self.b[i])
            self.recalculate_cdf()
        else:
            print('pdf already has been normalized once!')

    def recalculate_cdf(self):
        """Calculates cdf"""
        s = 0
        for i in range(len(self.y)):
            s += self.y[i]*(self.b[i+1]-self.b[i])
            self.cdfy[i] = s
            
    def join(self, pdf2):
        """Appends data from another pdf, should have same x"""
        self.y += pdf2.y
        self.stat += pdf2.stat
            
            
class random_path(object):
    """Container for a random trajectory"""
    
    def __init__(self, length=None):
        if length is None:
            self.length = 0
            self.x = []
            self.t = []
        else:
            self.length = length
            self.x = np.zeros(length)
            self.t = np.zeros(length)
            
    def push(self, t, x):
        self.length += 1
        self.x.append(x)
        self.t.append(t)
        
    def xx(self, tt):
        """Provides interpolation of
        the trajectory"""
        if self.t[0]<=tt<=self.t[-1]:
            for i in xrange(len(self.t)-1):
                if tt==self.t[i]:
                    return self.x[i]
                elif self.t[i]<tt<=self.t[i+1]:
                    return tt*(self.x[i] - self.x[i+1])/ \
                              (self.t[i] - self.t[i+1]) + \
                           (self.t[i]*self.x[i+1] - self.t[i+1]*self.x[i])/ \
                              (self.t[i] - self.t[i+1])
        elif tt<self.t[0]:
            return 0
        elif tt>self.t[-1]:
            return None #self.x[-1]
        else: # if nan or something
            return None
        
        
class ctrw_v(object):
    
    def __init__(self,x0=0,t0=0,v=1,wtr=np.random.exponential):
        """'wtr' is waiting time random"""
        self.x0 = x0
        self.t0 = t0
        self.x1 = x0
        self.t1 = t0
        self.v = v
        self.gen = wtr
        
    def step(self):
        if np.random.rand()<0.5:
            self.v *= -1
        dt = self.gen()
        self.t0 = self.t1
        self.x0 = self.x1
        self.t1 += dt
        self.x1 += self.v*dt
        
    def coord(self, t):
        """Provides interpolation of
        the trajectory"""
        # should be t0<t<t1
        return self.x0 + self.v*(t-self.t0)
    
    
class ctrw_wiener(object):
    
    def __init__(self,x0=0,t0=0):
        self.x0 = x0
        self.t0 = t0
        self.x1 = self.x0
        self.t1 = self.t0
        self.v = 0
        
    def step(self):
        
        self.t0 = self.t1
        self.x0 = self.x1
        dt = np.random.exponential()
        dx = np.random.normal()
        self.v = dx/dt
        self.t1 += dt
        self.x1 += dx
        
    def coord(self, t):
        """Provides interpolation of
        the trajectory"""
        # should be t0<t<t1
        return self.x0 + self.v*(t-self.t0)
        
        
class langevin_gaussian_overdamped(object):
    
    def __init__(self,dt,D,x0=0,t0=0):
        self.dt = dt
        self.D  = D
        self.x0 = x0
        self.t0 = t0
        self.x1 = self.x0
        self.t1 = self.t0
        self.v = 0
        
    def step(self, tau):
        """Performs one step in time"""
        self.t0 = self.t1
        self.x0 = self.x1
        dx = tau * np.sqrt(2.*self.dt*self.D) * np.random.normal()
        self.v = dx/self.dt
        self.t1 += self.dt
        self.x1 += dx
        return self.t1, self.x1
        
    def coord(self, t):
        """Provides interpolation of
        the trajectory"""
        # should be t0<t<t1
        return self.x0 + self.v*(t-self.t0)
        

class langevin_gaussian(object):
    
    def __init__(self,D,tau,x0=0,v0=0,t0=0):
        self.D  = D
        self.tau = tau
        self.x0 = x0
        self.v0 = v0
        self.t0 = t0
        self.x1 = self.x0
        self.v1 = self.v0
        self.t1 = self.t0
        
    def step(self, dt):
        """Performs one step in time"""
        self.x0 = self.x1
        self.v0 = self.v1
        self.t0 = self.t1

        dv = -1. / self.tau * self.v0 * dt + \
             np.sqrt(2.*dt*self.D) * np.random.normal()

        self.t1 += dt
        self.v1 += dv
        self.x1 += (self.v0+self.v1) / 2. * dt

        return self.x1, self.v1, self.t1
        
    def x(self, t):
        """Provides interpolation of
        the trajectory"""
        # should be t0<t<t1
        return self.x0 + (self.v1-self.v0)*(t-self.t0)

    def v(self, t):
        """Provides interpolation of
        the trajectory"""
        # should be t0<t<t1. 
        return self.v0 + (self.v1-self.v0)*(t-self.t0) / (self.t1 - self.t0)
        
                
class langevin_overdamped(object):
    
    def __init__(self,dt,D,alpha,x0=0,t0=0):
        self.dt = dt
        self.D  = D
        self.x0 = x0
        self.t0 = t0
        self.x1 = self.x0
        self.t1 = self.t0
        self.alpha = alpha
        self.gen = astable(alpha)
        
    def step(self):
        """Performs one step in time"""
        self.t0 = self.t1
        self.x0 = self.x1
        dx = (2.*self.dt*self.D)**(1./self.alpha) * self.gen.random()
        self.v = dx/self.dt
        self.t1 += self.dt
        self.x1 += dx
        return self.t1, self.x1
        
    def coord(self, t):
        """Provides interpolation of
        the trajectory"""
        # should be t0<t<t1
        return self.x0 + self.v*(t-self.t0)
