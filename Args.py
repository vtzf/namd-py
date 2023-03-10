import numpy as np
import os
from mpi4py import MPI

# NAMD parameter
# manual input start
interface= 'OpenMX' # 'VASP','ABACUS','OpenMX'
dftdir   = '/public/share/zhaojin/tuyy/deeph_test/run/'
# [openmxf,hamildir,olpdir,scfoutf,nelectron] interface='OpenMX' only
openmxf  = 'scfout'   # 'scfout','hdf5'
# only with openmxf = 'hdf5'
hamildir = 'out/' # hamiltonian matrix in hdf5 format
olpdir   = 'output/' # overlap matrix in hdf5 format
# only with openmxf = 'scfout'
scfoutf  = 'openmx.scfout' # scfout file name

nelectron= [12,6] # number of electron in each atomic type

namddir  = '../namd_test/' # output NAMD output file in namddir
dt       = 1      # MD time step (fs)
start_t  = 1      # start MD_* dir
end_t    = 500   # end MD_* dir
istart_t = 1    # isample start MD_* dir
iend_t   = 50   # isample end MD_* dir

LCHARGE  = True   # output atom projected charge density
atom     = [13,26]# atom number of all species (only needed in atomic basis)
orbital  = [14,13]# atomic orbital basis number (only needed in atomic basis)
whichA   = [0,13,14,15] # atom index for projected charge density (starts from 0)

LRANGE   = True   # select range of band, change iband
                  # if not given, LRECOMB specifies energy range
                  # if LRECOMB not given, energy range: [0,nbands]
LHOLE    = True   # Hole/electron transfer
dE       = 2.0    # initial energy from VBM/CBM (eV)

LPHASE   = True   # phase correction

TEMP     = 400    # temperature in Kelvin
NACTIME  = 450   # time for used NAC
NAMDTIME = 450   # time for NAMD run
NELM     = 10     # electron time step (per fs)
NTRAJ    = 5000   # SH trajectories number

LINTERP  = 2      # hamiltonian interpolation algorithm 1,2,3
LTA      = True   # Liouville-Trotter algorithm for small NELM
LSH    = 'FSSH'   # run DISH, FSSH or DCSH

LRECOMB  = False   # consider electron-hole recombination
# manual input end

# constants
Ry2eV = 13.605698065894
hbar = 0.6582119281559802 # reduced Planck constant eV/fs
eV = 1.60217733E-19
Kb_eV = 8.6173857E-5
Kb = Kb_eV*eV
KbT = kbT = Kb_eV * TEMP

# parameters check
comm = MPI.COMM_WORLD
myid = comm.Get_rank()
if myid == 0:
    if not os.path.exists(namddir):
        os.makedirs(namddir)

# preprocess
nsample = iend_t-istart_t+1
nstep = end_t-start_t+1
if interface == 'ABACUS':
    nbands = int((open(dftdir+'MD_%d/LOWF_GAMMA_S1.dat'%(start_t)).readline()).split()[0])
elif interface == 'VASP':
    f = open(dftdir+'%04d/PROCAR'%(start_t))
    f.readline()
    nbands = int((f.readline()).split()[7])
    f.close()
elif interface == 'OpenMX':
    nbands = np.dot(atom,orbital)
    band_vbm = np.dot(atom,nelectron)//2
band_s = 1
band_e = nbands
iband_s = 1
iband_e = nbands
ibands = iband_e-iband_s+1
edt = dt/NELM

if interface == 'ABACUS' or interface == 'OpenMX':
    orb_idx = np.zeros((sum(atom)+1),dtype=int)
    atom_idx = np.cumsum([0]+atom)
    for i in range(len(atom)):
        orb_idx[atom_idx[i]+1:atom_idx[i+1]+1]=orbital[i]
    orb_idx = np.cumsum(orb_idx)
    
    whichO = []
    for i in whichA:
        whichO1 = range(orb_idx[i],orb_idx[i+1])
        for j in whichO1:
            whichO.append(j)

