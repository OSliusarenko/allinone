
# coding: utf-8

# In[16]:


from __future__ import print_function

#import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pickle
import signal
import sys
import time

from datetime import datetime, timedelta
from rw import pdf_hist
from scipy.stats import gengamma

#get_ipython().magic(u'matplotlib inline')

TRIM_RESULT = 1e24


def init_pool(l):
    global lock
    lock = l


# In[36]:


def calcA(procNum):

    p = dict()
    V = dict()
    for n in ns:
        p[n] = pdf_hist(1e-4, 1e12, 64, True)

    lock.acquire()
    np.random.seed( int( time.time() ) + procNum )
    lock.release()

    errors = 0

    for i in xrange(stat):
        for n in ns:
            V[n] = 0

        if procNum==0:
            if (i * 100) % stat == 0:
                lock.acquire()
                print('{:}% '.format(i * 100 / stat), end='')
                sys.stdout.flush()
                lock.release()

        taus = np.random.exponential(size=N)
        ms = gengamma.rvs(1/c, c, size=N)

        for m, tau in zip(ms, taus):   # Subsidiary trajectories
            u = 0; t = 0
            while(t < tmax):
                try:
                    u += (- u * dt / tau + np.sqrt(2.*dt) *
                          np.random.normal()) / m
                except FloatingPointError:
                    errors += 1
                    u = TRIM_RESULT
                t += dt;
                n = round(t/dt)
                if n in ns:
                    try:
                        V[n] += m * u
                    except FloatingPointError:
                        V[n] = TRIM_RESULT
        for n in ns:
            V[n] /= sum(ms)
            p[n].inc_y(V[n])

    return [p, errors]


def calcB(procNum):

    p = dict()
    V = dict()
    for n in ns:
        p[n] = pdf_hist(1e-4, 1e12, 64, True)

    lock.acquire()
    np.random.seed( int( time.time() ) + procNum )
    lock.release()

    errors = 0

    for i in xrange(stat):
        for n in ns:
            V[n] = 0

        if procNum==0:
            if (i * 100) % stat == 0:
                lock.acquire()
                print('{:}% '.format(i * 100 / stat), end='')
                sys.stdout.flush()
                lock.release()

        taus = np.random.exponential(size=N)
        ms = gengamma.rvs(1/c, c, size=N)

        for m, tau in zip(ms, taus):   # Subsidiary trajectories
            u = 0; t = 0
            while(t < tmax):
                try:
                    u += - u * dt / (tau*m) + np.sqrt(2.*dt) * \
                         np.random.normal()
                except FloatingPointError:
                    errors += 1
                    u = TRIM_RESULT
                t += dt;
                n = round(t/dt)
                if n in ns:
                    try:
                        V[n] += m * u
                    except FloatingPointError:
                        V[n] = TRIM_RESULT

        for n in ns:
            V[n] *= np.sqrt(1./N*sum(map(lambda m: 1/m**2, ms))) / sum(ms)
            p[n].inc_y(V[n])

    return [p, errors]


def calc_gamma(procNum):

    lock.acquire()
    np.random.seed( int( time.time() ) + procNum )
    lock.release()

    errors = 0
    A = np.zeros(int(np.ceil(tmax/dt)+1))
    B = np.zeros(int(len(A)))

    taus = np.random.exponential(size=N)
    ms = gengamma.rvs(1/c, c, size=N)

    for m, tau in zip(ms, taus):   # Subsidiary trajectories
        for i in xrange(stat):
            u0 = 0; t = 0
            while(t < tmax):
                try:
                    u0 += - u0 * dt / (tau*m) + np.sqrt(2.*dt) * \
                         np.random.normal()
                except FloatingPointError:
                    errors += 1
                    u0 = 0
                    break
                t += dt;
                try:
                    A[int(t/dt)] += (u0/(tau*m)) ** 2
                    B[int(t/dt)] += (u0*m) ** 2
                except FloatingPointError:
#                    print('error')
                    errors += 1
                    break
                    A[int(t/dt)] = 0
                    B[int(t/dt)] = 0

    return [A, B, errors]


def calcC(procNum): # rewrite

    p = dict()
    for n in ns:
        p[n] = pdf_hist(1e-4, 1e12, 64, True)

    lock.acquire()
    np.random.seed( int( time.time() ) + procNum )
    lock.release()

    errors = 0

    for i in xrange(stat):

        if procNum==0:
            if (i * 100) % stat == 0:
                lock.acquire()
                print('{:}% '.format(i * 100 / stat), end='')
                sys.stdout.flush()
                lock.release()

        taus = np.random.exponential(size=N)
        ms = gengamma.rvs(1/c, c, size=N)

        V = 0; t = 0
        while(t < tmax):
            t += dt;
            n = round(t/dt)
            try:
                V += - V * dt * gammaeff[n] + np.sqrt(2.*dt) * \
                     np.random.normal()
            except FloatingPointError:
                errors += 1
                V = TRIM_RESULT

            if n in ns:
                p[n].inc_y(V)

    return [p, errors]



# In[50]:

np.seterr(over='raise', invalid='raise')

N = int(1e0)
stat = int(4e1)
c = 0.75

dt = 1e-6; tmax = 1e-1
ns = map(round, np.logspace(-3, np.log10(tmax), 5)/dt)

processors = 1   #multiprocessing.cpu_count()
stat /= processors
print('Using {:} processors'.format(processors))

l = multiprocessing.Lock()
original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
pool = multiprocessing.Pool(processors, initializer=init_pool, initargs=(l,))
signal.signal(signal.SIGINT, original_sigint_handler)


# In[ ]:


# Calculation

t1 = time.time()

try:
    res = pool.map_async(calc_gamma, range(processors))
    print("Waiting for results")
    sys.stdout.flush()
    result = res.get(2678400)
except KeyboardInterrupt:
    print("Caught KeyboardInterrupt, terminating workers")
    sys.stdout.flush()
    pool.terminate()
else:
    t2 = time.time()
    sec = timedelta(seconds=t2-t1)
    d = datetime(2017,1,1) + sec

    print("\nNormal termination. Time needed {:d} days, {:%H:%M:%S}".
          format(d.day-1, d))
    sys.stdout.flush()
    pool.close()
pool.join()

if True:
    # C:

    ###

    A = result[0][0]
    B = result[0][1]
    errs = result[0][2]

    for res in result[1:]:
 #       A += res[0]
#        B += res[1]
        errs += res[2]

    print('{:} errors'.format(errs))

    A[B==0] = 0

    A[B!=0] /= B[B!=0]
    A = np.sqrt(A)

    del B

    # put here usage of gamma(t)



    with open('taueff.pkl', 'wb') as ff:
        pickle.dump(A, ff)

else:
    # A, B:
    pdfs = result[0][0]
    errs = result[0][1]

    for res in result[1:]:
        pdfstmp = res[0]
        errs += res[1]
        for key in pdfstmp:
            pdfs[key].join(pdfstmp[key])

    for key in pdfs:
        pdfs[key].normalize()

    with open('pdfsC.pkl', 'wb') as ff:
        pickle.dump(pdfs, ff)






# In[ ]:




