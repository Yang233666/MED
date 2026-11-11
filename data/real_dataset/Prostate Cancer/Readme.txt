The first part of the  folder contains a traing data of 102 samples of 
Prostate Cancer on the same set of 12600 genes.  Out of which 52 are tumor, 
50 are normal. It also contains two test data set. The first one contains 
9 samples of Prostate Cancer on the same set of 12600 genes, in the class
of normal. The second one contains 25 samples of Prostate Cancer on the 
same set of 12600 genes, but in the class of tumor. 

Prostate_TN_final10701_allmeanScale_res.txt:  data matrix download from 
http://www.broad.mit.edu/cgi-bin/cancer/dataset.cgi. See more information 
there. 

GNF_prostate_data_CR61_5974.xls: data matrix download from 
http://public.gnf.org/cancer/prostate/. The data is used in Welsh et al. 
(2001). It is  also cited in Singh et al. (2002) as well as Fan and Fan, 
where it is used a test set. The original file contains measurements on 
12626 genes, out of which 12600 are the set as in Prostate_TN_final110701_allmeansScale_res.txt. In the 
Welsh_Prostate_test_normal.txt and Welsh_Prostate_test_tumor.txt, the 
following 26 lines are removed: 

8697   10755  10839  10860  10913  11213  11319  11419  11420  11512
11513  11726  11738  11761  11932  11933  12297  12303  12312  12330
12344  12601  12606  12607  12609  12614
 

Prostate_normal_matrix.txt:  50 normal samples (train). 
Prostate_tumor_matrix.txt:   52 tumor samples (train). 
Welsh_Prostate_test_normal.txt:  9 normal samples (test).
Welsh_Prostate_test_tumor.txt:  25 normal samples (test) 



The second part of this folder contains 3 data files corresponding to the 
above Prostate data. The difference is that the data set has been pre-feature 
selected and standardized including an logarithmic (base 10) transform. The 
number of features now is only 6033. See details in Dettling's paper. 

The 3 files are: 

prostate.rda, downloaded from http://stat.ethz.ch/~dettling/bagboost.html.
prostate.x.txt, data matrix (6033 by 102) ready for use in Matlab.   
prostate.y.txt, vector of class index (102 by 1). 

Relevant paper: Singh et al. (2002), Welsh et al. (2001), Fan (2007), and 
Dettling (2004). 
