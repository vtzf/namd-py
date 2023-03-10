import numpy as np
from scipy.integrate import cumtrapz
from scipy.signal import correlate
from scipy.optimize import curve_fit
import Args

def LoadDataNull(step,nsend,Type):
    return np.zeros([step]+nsend,dtype=Type)


# Energy read/write
def ReadE(idx,data):
    line = open(Args.dftdir+'/%04d/EIGENVAL'%(idx)).readlines()
    nband = int(line[5].split()[2])
    energy = np.array([line[8+i].split()[1] for i in range(nband)],dtype=float)

    band_vbm = 0
    idx_s = 0
    idx_e = nband
    while True:
        if ((idx_e-idx_s)<=1):
            band_vbm = idx_e
            break
        band_vbm = int((idx_s+idx_e)/2)
        if (occ[band_vbm]<1e-10):
            idx_e = band_vbm
        else:
            idx_s = band_vbm
    if Args.LHOLE:
        if Args.LRECOMB:
            return np.concatenate((energy,[1],[band_vbm+1]))
        else:
            return np.concatenate((energy,[1],[band_vbm]))
    else:
        if Args.LRECOMB:
            return np.concatenate((energy,[nband],[band_vbm]))
        else:
            return np.concatenate((energy,[nband],[band_vbm+1]))


def gaussian(x,c):
    return np.exp(-x**2/(2*c**2))


def Dephase(energy):
    T = np.arange(Args.nstep-1)*Args.dt
    matrix = np.zeros((energy.shape[1],energy.shape[1]),dtype=float)
    for ii in range(energy.shape[1]):
        for jj in range(ii):
            Et = energy[:,ii]-energy[:,jj]
            Et -= np.average(Et)
            Ct = correlate(Et,Et)[Args.nstep:]/Args.nstep
            Gt = cumtrapz(Ct,dx=Args.dt,initial=0)
            Gt = cumtrapz(Gt,dx=Args.dt,initial=0)
            Dt = np.exp(-Gt/Args.hbar**2)
            popt,pcov = curve_fit(gaussian,T,Dt)
            matrix[ii,jj] = popt[0]
            matrix[jj,ii] = matrix[ii,jj]

    return matrix


# Energy read/write, obtain iband parameters
# save all_en.npy, EIGTXT, INICON
def ReadE1(idx,data):
    line = open(Args.dftdir+'/%04d/EIGENVAL'%(idx)).readlines()
    nband = int(line[5].split()[2])
    e_occ = np.array([line[8+i].split()[1:3] for i in range(nband)],dtype=float)
    energy = e_occ[:,0]
    occ = e_occ[:,1]

    band_vbm = 0
    idx_s = 0
    idx_e = nband
    while True:
        if ((idx_e-idx_s)<=1):
            band_vbm = idx_e
            break
        band_vbm = int((idx_s+idx_e)/2)
        if (occ[band_vbm]<1e-10):
            idx_e = band_vbm
        else:
            idx_s = band_vbm

    if Args.LHOLE:
        band_init = 0
        idx_s = 0
        idx_e = band_vbm
        while True:
            if ((idx_e-idx_s)<=1):
                band_init = idx_s
                break
            band_init = int((idx_s+idx_e)/2)
            if (energy[band_vbm-1]-energy[band_init]>Args.dE):
                idx_s = band_init
            else:
                idx_e = band_init
        if Args.LRECOMB:
            return np.concatenate((energy,[band_init],[band_vbm+1]))
        else:
            return np.concatenate((energy,[band_init],[band_vbm]))
    else:
        band_init = nband
        idx_s = band_vbm
        idx_e = nband
        while True:
            if ((idx_e-idx_s)<=1):
                band_init = idx_e
                break
            band_init = int((idx_s+idx_e)/2)
            if (energy[band_init]-energy[band_vbm]>Args.dE):
                idx_e = band_init
            else:
                idx_s = band_init
        if Args.LRECOMB:
            return np.concatenate((energy,[band_init],[band_vbm]))
        else:
            return np.concatenate((energy,[band_init],[band_vbm+1]))


def SaveE(savename,energy):
    np.save(Args.namddir+savename,energy[:,:-2])
    iband_s = 1
    iband_e = Args.nbands
    if Args.LHOLE:
        iband_s = int(np.min(energy[:,-2]))
        iband_e = int(np.max(energy[:,-1]))
    else:
        iband_e = int(np.max(energy[:,-2]))
        iband_s = int(np.min(energy[:,-1]))

    with open(Args.namddir+'INICON','w') as f:
        for i in range(Args.istart_t-Args.start_t,Args.iend_t+1-Args.start_t):
            f.write('%3d%5d\n'%(i+1,int(energy[i,-2])))
    with open(Args.namddir+'bandrange.dat','w') as f:
        f.write('%d %d\n'%(iband_s,iband_e))
    np.savetxt(Args.namddir+'EIGTXT',energy[:,iband_s-1:iband_e])
    if Args.LSH != 'FSSH':
        matrix = Dephase(energy[:,iband_s-1:iband_e])
        np.savetxt(Args.namddir+"DEPHTIME",matrix)


# elgenvector/overlap read/write
def setWFPrec(rtag):
    if rtag == 45200:
        return np.complex64
    elif rtag == 45210:
        return np.complex128


def orthogon(cic,WFPrec):
    S = np.dot(cic,cic.T.conj())
    Dsqrt = np.zeros_like(S,dtype=WFPrec)
    D,V = np.linalg.eig(S)
    Dsqrt += np.diag(1/np.sqrt(D))
    T = np.dot(np.dot(V,Dsqrt.conj()),V.T.conj())

    return np.dot(T,cic)


def ReadOrbital(Dir,Band_s,Band_e):
    wfc = open(Dir+'/WAVECAR','rb')
    wfc.seek(0)
    recl,nspin,rtag = np.array(np.fromfile(wfc,dtype=float,count=3),dtype=int)
    WFPrec = setWFPrec(rtag)
    wfc.seek(recl)
    nband = int(np.fromfile(wfc,dtype=float,count=2)[1])
    wfc.seek(2*recl)
    nplws = int(np.fromfile(wfc,dtype=float,count=1)[0])
    wav = np.zeros((nband,nplws),dtype=WFPrec)
    for i in range(nband):
        wfc.seek((3+i)*recl)
        wav[i] = np.fromfile(wfc,dtype=WFPrec,count=nplws)
    wav /= np.linalg.norm(wav,axis=1,keepdims=True)
    #wav = orthogon(wav,WFPrec)

    return (wav[Band_s-1:Band_e].T).astype(complex)


# charge density read/write, save all_wht.npy
def CDInfo(idx,data):
    line = open(Args.dftdir+'/%04d/PROCAR'%(idx)).readlines()
    nband = int(line[1].split()[7])
    natom = int(line[1].split()[11])
    All = np.array([[line[5+i*(natom+5)+j+3].split()[-1] \
          for i in range(nband)] for j in range(natom)],dtype=float)
    CAll = np.sum(All,axis=0)
    CPart = np.sum(All[Args.whichA],axis=0)

    return CPart/CAll


def SaveCD(savename,data):
    np.save(Args.namddir+savename,data)


# phase correction, save phase_m.npy
def Phase(vec1,vec2):
    phase = np.zeros((Args.ibands))
    for i in range(Args.ibands):
        phase[i] = np.dot(vec1[:,i].conj(),vec2[:,i]).real

    return np.sign(phase).astype('int32')


#def Phase(vec1,vec2):
#    phase = np.dot(vec1.T.conj(),vec2)
#    phase = np.diagonal(phase).real
#
#    return np.sign(phase).astype('int32')


def PhaseInfo(idx,data):
    if idx == Args.start_t:
        vec1 = ReadOrbital(Args.dftdir+'%04d'%(idx),Args.iband_s,Args.iband_e)
        phase = Phase(vec1,vec1)
    else:
        vec1 = ReadOrbital(Args.dftdir+'%04d'%(idx-1),Args.iband_s,Args.iband_e)
        vec2 = ReadOrbital(Args.dftdir+'%04d'%(idx),Args.iband_s,Args.iband_e)
        phase = Phase(vec1,vec2)

    return phase


def PhaseMatrix(phase):
    phase_m = np.zeros((Args.nstep-1,Args.ibands,Args.ibands),dtype='int32')
    
    p0 = phase[0]
    for i in range(1,Args.nstep):
        p1 = p0*phase[i]
        phase_m[i-1] = np.dot(p0.reshape(-1,1),p1.reshape(1,-1))
        p0[:] = p1

    return phase_m


def SavePhase(savename,phase):
    phase_m = PhaseMatrix(phase)
    np.save(Args.namddir+savename,phase_m)


# NAC with/without phase correction, save NATXT
def LoadDataPhase(step,nsend,Type):
    return np.load(Args.namddir+'/phase_m.npy')


def NAC(vec1,vec2):
    nac = np.dot(vec1.T.conj(),vec2)-np.dot(vec2.T.conj(),vec1)

    return nac.real


def NACInfo(idx,phase):
    vec1 = ReadOrbital(Args.dftdir+'%04d'%(idx),Args.iband_s,Args.iband_e)
    vec2 = ReadOrbital(Args.dftdir+'%04d'%(idx+1),Args.iband_s,Args.iband_e)
    nac = NAC(vec1,vec2)
    return nac


def NACPhase(vec1,vec2,phase):
    nac = np.dot(vec1.T.conj(),vec2)*phase\
          -np.dot(vec2.T.conj(),vec1)*phase.T

    return nac.real


def NACPhaseInfo(idx,phase):
    vec1 = ReadOrbital(Args.dftdir+'%04d'%(idx),Args.iband_s,Args.iband_e)
    vec2 = ReadOrbital(Args.dftdir+'%04d'%(idx+1),Args.iband_s,Args.iband_e)
    nac = NACPhase(vec1,vec2,phase)
    return nac


def SaveNAC(savename,data):
    np.savetxt(Args.namddir+savename,data.reshape(Args.nstep-1,-1))


def SaveCOUP(savename,data):
    energy = np.load(Args.namddir+'/all_en.npy')
    buf = np.zeros((Args.nstep,Args.nbands+1,Args.nbands))
    buf[0,0,0] = Args.nbands*(Args.nbands+1)*8
    buf[0,0,1] = Args.nbands
    buf[0,0,2] = Args.nstep
    buf[0,0,3] = Args.dt
    buf[1:,Args.nbands,:] = energy[0:-1]
    buf[1:,0:Args.nbands,:] = data
    buf.tofile(Args.namddir+savename)
