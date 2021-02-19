# PBCI
## Canonical Correlation Based BCI

The code provides offline aclassification algorithms for the provided benchmark dataset. Imlpemented are:

 1. CCA: simple CCA based classification
 2. FBCCA: Ensamble classifier using frequency subbands to increase contribution by harmonics
 3. Extended CCA: Ensamlbe classifier based approach taking subject specific training data into account
 4. Extended FBCCA: combination of FBCCA and extended CCA
 
Further, the Repository contains shell scripts to run the files on the HPC (Supercomputer) at DTU. These scipts take variables as well, to simplify running with different parameters.

```
bsub -env length=5,subjects=35,tag=filt < submit_cca.sh;
bsub -env length=5,subjects=35,tag=filt < submit_ext_cca.sh;
bsub -env length=5,subjects=35,tag=filt < submit_fbcca.sh;
bsub -env length=5,subjects=35,tag=filt < submit_ext_fbcca.sh;
bstat
```

Additionally, all scripts can be called with command-line commands:
```
run cca.py --length 5 --subjects 35 --tag test
run fbcca.py --length 5 --subjects 35 --tag test
run advanced_cca.py --length 5 --subjects 35 --tag test
run advanced_fbcca.py --length 5 --subjects 35 --tag test

// evaluation script
run evaluation --length 5 --subjects 35 --tag test
```
functions.py contains all functions that are used multiple times and has to be imported at the beginning of all other files.
