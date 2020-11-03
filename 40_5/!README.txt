The test instances and some results for Table 3 in Belov & Scheithauer (2002).

The result files contain in each line:
No.    :instance number
Stages :number of cut-iterations
zi     :the best found int solution
zi-zc0 :difference zi minus the initial LP bound
zi-zc  :difference zi minus the best LP bound
time   :the time after the initial LP
m0     :number of product types
M0     :number of stock types
Lmin   :min stock length
Lmax   :max stock length

As you see in the param file ss.cfg, only LP and simple heuristical rounding (option LPonly) were made, no cuts as in the paper.