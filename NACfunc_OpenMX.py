import numpy as np
from scipy.integrate import cumtrapz
from scipy.signal import correlate
from scipy.optimize import curve_fit
import h5py
import json
from glob import glob
from scipy.linalg import eigh
import os
import Args

def LoadDataNull(step,nsend,Type):
    return np.zeros([step]+nsend,dtype=Type)


# Energy read/write
def ReadScfout(Name):
    fp = open(Name,'rb')
    fp.seek(0)
    i_vec = np.fromfile(fp,dtype=np.intc,count=6)
    atomnum = i_vec[0]
    TCpyCell = i_vec[5]
    fp.seek(4+(TCpyCell+1)*4*(8+4),1)
    
    Total_NumOrbs = np.ones((atomnum+1),dtype=np.intc)
    Total_NumOrbs[1:] = np.fromfile(fp,dtype=np.intc,count=atomnum)
    FNAN = np.zeros((atomnum+1),dtype=np.intc)
    FNAN[1:] = np.fromfile(fp,dtype=np.intc,count=atomnum)
    FNAN_sum = np.sum(FNAN[1:])+atomnum
    natn = [[]]
    for ct_AN in range(1,atomnum+1):
        natn.append(np.fromfile(fp,dtype=np.intc,count=FNAN[ct_AN]+1))
    for ct_AN in range(1,atomnum+1):
        fp.seek((FNAN[ct_AN]+1)*4,1) 
    fp.seek((3+3+atomnum)*4*8,1)

    hamil = np.zeros((Args.nbands,Args.nbands),dtype=float)
    for ct_AN in range(1,atomnum+1):
        TNO1 = Total_NumOrbs[ct_AN]
        for h_AN in range(FNAN[ct_AN]+1):
            Gh_AN = natn[ct_AN][h_AN]
            TNO2 = Total_NumOrbs[Gh_AN]
            Hks1 = np.fromfile(fp,dtype=float,count=TNO1*TNO2)
            hamil[Args.orb_idx[ct_AN-1]:Args.orb_idx[ct_AN],\
                  Args.orb_idx[Gh_AN-1]:Args.orb_idx[Gh_AN]] += Hks1.reshape(TNO1,TNO2)
    
    olp = np.zeros((Args.nbands,Args.nbands),dtype=float)
    for ct_AN in range(1,atomnum+1):
        TNO1 = Total_NumOrbs[ct_AN]
        for h_AN in range(FNAN[ct_AN]+1):
            Gh_AN = natn[ct_AN][h_AN]
            TNO2 = Total_NumOrbs[Gh_AN]
            OLP1 = np.fromfile(fp,dtype=float,count=TNO1*TNO2)
            olp[Args.orb_idx[ct_AN-1]:Args.orb_idx[ct_AN],\
                Args.orb_idx[Gh_AN-1]:Args.orb_idx[Gh_AN]] += OLP1.reshape(TNO1,TNO2)
    
    fp.close()
    return hamil*(2*Args.Ry2eV), olp


def ReadH5(Name,mat):
    f=h5py.File(Name,'r')
    Os={}
    for key_str, O_nm in f.items():
        Os[key_str] = O_nm[...]

    h_key = list(Os.keys())
    h_key = np.array([json.loads(x) for x in h_key],dtype=int)
    h_value = list(Os.values())

    for i in range(h_key.shape[0]):
        mat[Args.orb_idx[h_key[i,3]-1]:Args.orb_idx[h_key[i,3]],\
            Args.orb_idx[h_key[i,4]-1]:Args.orb_idx[h_key[i,4]]] += h_value[i]


def ReadMat(Dir):
    mat = np.zeros((Args.nbands,Args.nbands),dtype=float)
    matfile = glob(Dir)
    for i in matfile:
        ReadH5(i,mat)

    return mat


def ReadE(idx,data):
    if not os.path.exists(Args.dftdir+'/%04d/vec.npy'%(idx)):
        if Args.openmxf == 'hdf5':
            hamil = ReadMat(Args.dftdir+'/%04d/'%(idx)+Args.hamildir+'/hamiltonians*')
            olp = ReadMat(Args.dftdir+'/%04d/'%(idx)+Args.olpdir+'/overlaps*')
        else:
            hamil,olp = ReadScfout(Args.dftdir+'/%04d/'%(idx)+Args.scfoutf)
        val,vec = eigh(hamil,olp)
        np.save(Args.dftdir+'/%04d/olp.npy'%(idx),olp)
        np.save(Args.dftdir+'/%04d/vec.npy'%(idx),vec)
        np.save(Args.dftdir+'/%04d/val.npy'%(idx),val)
    else:
        val = np.load(Args.dftdir+'/%04d/val.npy'%(idx))
    if Args.LHOLE:
        if Args.LRECOMB:
            return np.concatenate((val,[1],[Args.band_vbm+1]))
        else:
            return np.concatenate((val,[1],[Args.band_vbm]))
    else:
        if Args.LRECOMB:
            return np.concatenate((val,[nband],[Args.band_vbm]))
        else:
            return np.concatenate((val,[nband],[Args.band_vbm+1]))


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
    if not os.path.exists(Args.dftdir+'/%04d/vec.npy'%(idx)):
        if Args.openmxf == 'hdf5':
            hamil = ReadMat(Args.dftdir+'/%04d/'%(idx)+Args.hamildir+'/hamiltonians*')
            olp = ReadMat(Args.dftdir+'/%04d/'%(idx)+Args.olpdir+'/overlaps*')
        else:
            hamil,olp = ReadScfout(Args.dftdir+'/%04d/'%(idx)+Args.scfoutf)
        energy,vec = eigh(hamil,olp)
        np.save(Args.dftdir+'/%04d/olp.npy'%(idx),olp)
        np.save(Args.dftdir+'/%04d/vec.npy'%(idx),vec)
        np.save(Args.dftdir+'/%04d/val.npy'%(idx),energy)
    else:
        energy = np.load(Args.dftdir+'/%04d/val.npy'%(idx))

    band_vbm = Args.band_vbm
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
def ReadOrbital(Dir,Band_s,Band_e):
    orbital = np.load(Dir+'/vec.npy')

    return orbital[:,Band_s-1:Band_e]


def ReadMatrix(Dir,Name):
    mat = np.load(Dir+'/'+Name)

    return mat


# charge density read/write, save all_wht.npy
def ChargeDensity(vec,olp):
    All = vec*np.dot(olp,vec)
    CAll = np.sum(All,axis=0)
    CPart = np.sum(All[Args.whichO],axis=0)

    return CPart/CAll


def CDInfo(idx,data):
    vec = ReadOrbital(Args.dftdir+'/%04d'%(idx),Args.band_s,Args.band_e)
    olp = ReadMatrix(Args.dftdir+'/%04d'%(idx),'olp.npy')

    CD = ChargeDensity(vec,olp)

    return CD


def SaveCD(savename,data):
    np.save(Args.namddir+savename,data)


# phase correction, save phase_m.npy
def Phase(vec1,vec2,olp1,olp2):
    olp12 = (olp1+olp2)/2
    phase = np.zeros((Args.ibands))
    vec_olp = np.dot(vec1.T,olp12)
    for i in range(Args.ibands):
        phase[i] = np.dot(vec_olp[i],vec2[:,i])

    return np.sign(phase).astype('int32')


#def Phase(vec1,vec2,olp1,olp2):
#    olp12 = (olp1+olp2)/2
#    phase = np.dot(np.dot(vec1.T,olp12),vec2)
#    phase = np.diagonal(phase)
#
#    return np.sign(phase).astype('int32')


def PhaseInfo(idx,data):
    if idx == Args.start_t:
        vec1 = ReadOrbital(Args.dftdir+'/%04d'%(idx),Args.iband_s,Args.iband_e)
        olp1 = ReadMatrix(Args.dftdir+'/%04d'%(idx),'olp.npy')
        phase = Phase(vec1,vec1,olp1,olp1)
    else:
        vec1 = ReadOrbital(Args.dftdir+'/%04d'%(idx-1),Args.iband_s,Args.iband_e)
        vec2 = ReadOrbital(Args.dftdir+'/%04d'%(idx),Args.iband_s,Args.iband_e)
        olp1 = ReadMatrix(Args.dftdir+'/%04d'%(idx-1),'olp.npy')
        olp2 = ReadMatrix(Args.dftdir+'/%04d'%(idx),'olp.npy')
        phase = Phase(vec1,vec2,olp1,olp2)

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


def NAC(vec1,vec2,olp1,olp2):
    olp12 = (olp1+olp2)/2
    nac = np.dot(np.dot(vec1.T,olp12),vec2)\
          -np.dot(np.dot(vec2.T,olp12),vec1)

    return nac


def NACInfo(idx,phase):
    vec1 = ReadOrbital(Args.dftdir+'/%04d'%(idx),Args.iband_s,Args.iband_e)
    vec2 = ReadOrbital(Args.dftdir+'/%04d'%(idx+1),Args.iband_s,Args.iband_e)
    olp1 = ReadMatrix(Args.dftdir+'/%04d'%(idx),'olp.npy')
    olp2 = ReadMatrix(Args.dftdir+'/%04d'%(idx+1),'olp.npy')
    nac = NAC(vec1,vec2,olp1,olp2)
    return nac


def NACPhase(vec1,vec2,olp1,olp2,phase):
    olp12 = (olp1+olp2)/2
    nac = (np.dot(np.dot(vec1.T,olp12),vec2))*phase\
          -(np.dot(np.dot(vec2.T,olp12),vec1))*phase.T

    return nac


def NACPhaseInfo(idx,phase):
    vec1 = ReadOrbital(Args.dftdir+'/%04d'%(idx),Args.iband_s,Args.iband_e)
    vec2 = ReadOrbital(Args.dftdir+'/%04d'%(idx+1),Args.iband_s,Args.iband_e)
    olp1 = ReadMatrix(Args.dftdir+'/%04d'%(idx),'olp.npy')
    olp2 = ReadMatrix(Args.dftdir+'/%04d'%(idx+1),'olp.npy')
    nac = NACPhase(vec1,vec2,olp1,olp2,phase)
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
