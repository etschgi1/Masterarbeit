# Project types
- Test project
Gives a limited amount of time to use
- Funded project
Peer-reviewed paper appreciated. Check if Hauser has proof of funding! 
Project Manager will be supervisor - does project application + admin on vsc. + creates user accounts
**Acknowledge VSC on paper**
**Add publication to VSC Homepage (see VSC_intro)**

## Hardware
VSC-4: Intel CPUs, skylake
VSC-5: Intel cascadelake & amd epyc milan

Fat-tree switched (2-level fat-tree on VSC-4 and VSC-5)

## Password change for login here: 
https://service.vsc.ac.at/media/index.html?entityID=https%3A%2F%2Fservice.vsc.tuwien.ac.at%2Fshibboleth&return=https%3A%2F%2Fservice.vsc.ac.at%2FShibboleth.sso%2FLogin%3FSAMLDS%3D1%26target%3Dss%253Amem%253Ac3489ae76e58f3f6751e7676f345ada949ba19d9293d5b1e44e0e31eb27eddde

## SSH - key login only over port 27
ssh -p 27!!!

## Storage
See Storage.pdf
**Global File-Sys**
*$Home* -> 100GB limit (strict) 10^6 inodes default 
current usage: mmlsquota --block-size auto -j home_fs7XXXX home
or: df-h $Home
df -i $Home (for inodes)

*$Data* -> 10 TB 10^6 inodes
same commands as above with $Data

**Local Disk**
480 GB local SSD (VSC-4)
2 TB NVMe (VSC-5) 
accessible under /local 

**TMP**
/tmp 
if we run on only one node! 
e.g. for xyz files - faster loading

## Spack
Deprecated! Will change in the future

## Modules
Needed to load in software & set env vars
see pdf modules for details

## EESSI 
module purge is recommended at start of slurm script.

## Conda
u know what it is
module load miniconda3/latest 
to install! 
also run: 
eval "$(conda shell.bash hook)"

prepare env before slurm job! 
-> *Create sbatch & setup conda there! look in slides 9*

## Slurm 
Slides show basic concepts really good
--partition= XX !different mem / gpu / non gpu
--qos= XX !

**sinfo** to get info about every partition! or *sinfo -o %P*

**If you send procs into background** use *wait* at the end of the slurm-script such that it waits for all bg-procs to terminate and only end the slurm-job thereafter!

--> also look into SLURM_advanced as well!

**ssh into nodes while running**: ? somehow possible
