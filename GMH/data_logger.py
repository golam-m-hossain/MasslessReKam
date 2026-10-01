#######################################################
#
# Log various stages during computations
#
#######################################################

#######################################################
import os, time
from os.path import dirname, abspath



def runtime_datalog(data):
    """
    Create a runtime log
    """
    d = dirname(abspath(__file__))
    filename = d + "/runtime.log"
    
    with open(filename, 'a') as fl:
        fl.write("%s: %s\n"%(time.ctime(),data))


#######################################################
