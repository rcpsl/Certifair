# #!/bin/bash

#P2 properties
#base
python certifair.py adult property5 adult_p5_base --lr 0.001 --fr 0.0 --no-bound_loss
exit 1
python certifair.py german property1 german_p1_base --lr 0.007 --fr 0.0 --no-bound_loss --layers "30,30,1"
python certifair.py compas property1 compas_p1_base --lr 0.001 --fr 0.0 --no-bound_loss
python certifair.py law property2 law_p2_base --lr 0.001 --fr 0.0 --no-bound_loss --batch_size 4096

#global
python certifair.py adult property5 adult_p5_global --lr 0.001 --fr 0.05 --property_loss
python certifair.py german property1 german_p1_global --lr 0.007 --fr 0.01 --property_loss --layers "30,30,1"
python certifair.py compas property1 compas_p1_global --lr 0.01 --fr 0.1 --property_loss
python certifair.py law property2 law_p2_global --lr 0.001 --fr 0.0130 --property_loss --batch_size 4096

#local
python certifair.py adult property5 adult_p5_local --lr 0.01 --fr 0.95 
python certifair.py german property1 german_p1_local --lr 0.007 --fr 0.2  --layers "30,30,1"
python certifair.py compas property1 compas_p1_local --lr 0.01 --fr 0.9 
python certifair.py law property2 law_p2_local --lr 0.01 --fr 0.53 --batch_size 4096


#P1 properties

#base
python certifair.py adult property6 adult_p6_base --lr 0.001 --fr 0.0 --no-bound_loss
python certifair.py german property2 german_p2_base --lr 0.007 --fr 0.0 --no-bound_loss --layers "30,30,1"
python certifair.py compas property2 compas_p2_base --lr 0.001 --fr 0.0 --no-bound_loss
python certifair.py law property3 law_p3_base --lr 0.001 --fr 0.0 --no-bound_loss

#global
python certifair.py adult property6 adult_p6_global --lr 0.001 --fr 0.01 --property_loss
python certifair.py german property2 german_p2_global --lr 0.001 --fr 0.006 --property_loss --layers "30,30,1"
python certifair.py compas property2 compas_p2_global --lr 0.001 --fr 0.02 --property_loss
python certifair.py law property3 law_p3_global --lr 0.001 --fr 0.001 --property_loss --batch_size 4096

#local
python certifair.py adult property6 adult_p6_local --lr 0.01 --fr 0.95 
python certifair.py german property2 german_p2_local --lr 0.007 --fr 0.2  --layers "30,30,1"
python certifair.py compas property2 compas_p2_local --lr 0.01 --fr 0.9 
python certifair.py law property3 law_p3_local --lr 0.01 --fr 0.53 --batch_size 4096
