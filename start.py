import os
import subprocess as sp

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sp.call(['.\setup.bat'])