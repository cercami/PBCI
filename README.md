# PBCI
## Canonical Correlation Based BCI

The code provides offline aclassification algorithms for the provided benchmark dataset [Wang2017]. Imlpemented are:

 1. CCA: simple CCA based classification [Lin2007]
 2. FBCCA: Ensamble classifier using frequency subbands to increase contribution by harmonics [Chen2015a]
 3. Extended CCA: Ensamlbe classifier based approach taking subject specific training data into account  [Chen2014]
 4. Extended FBCCA: combination of FBCCA and extended CCA [Chen2015b]
 
 ## Command line tools
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

## Folder structure
Files are stored in following structure:
```
PBCI/
|
├── results/ # all the outputs/results from 1.-4.       
│
├── data/ # subject data from benchmar data set
|
├── figures/ # resulting figures from evaluation
|
└── log/ # output from cluster
```

## Resources
[Lin2007] Z.  Lin,  C.  Zhang,  W.  Wu,  and  X.  Gao,  “Frequency  recognitionbased  on  canonical  correlation  analysis  for  SSVEP-Based  BCIs,”IEEE  Transactions  on  Biomedical  Engineering,  vol.  54,  no.  6,pp. 1172–1176, 2007,ISSN: 00189294


[Chen2014] X.  Chen,  Y.  Wang,  M.  Nakanishi,  T.  P.  Jung,  and  X.  Gao,  “Hy-brid  frequency  and  phase  coding  for  a  high-speed  SSVEP-basedBCI  speller,”2014  36th  Annual  International  Conference  of  theIEEE  Engineering  in  Medicine  and  Biology  Society,  EMBC  2014,pp. 3993–3996, 2014.


[Chen2015a] X.   Chen,   Y.   Wang,   S.   Gao,   T.   P.   Jung,   and   X.   Gao,   “Filterbank  canonical  correlation  analysis  for  implementing  a  high-speedSSVEP-based  brain-computer  interface,”Journal  of  Neural  Engi-neering, vol. 12, no. 4, 2015,ISSN: 17412552


[Chen2015b] X.  Chen,  Y.  Wang,  M.  Nakanishi,  X.  Gao,  T.  P.  Jung,  and  S.  Gao,“High-speed  spelling  with  a  noninvasive  brain-computer  interface,”Proceedings of the National Academy of Sciences of the United Statesof America, vol. 112, no. 44, E6058–E6067, 2015,ISSN: 10916490


[Wang2017] Y.  Wang,  X.  Chen,  X.  Gao,  and  S.  Gao,  “A  Benchmark  Datasetfor  SSVEP-Based  Brain-Computer  Interfaces,”IEEE  Transactionson Neural Systems and Rehabilitation Engineering, vol. 25, no. 10,pp. 1746–1752, 2017,ISSN: 15344320
