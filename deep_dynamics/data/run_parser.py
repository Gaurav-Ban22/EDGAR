import sys
import os
sys.path.append(os.path.abspath('..'))
from tools.csv_parser import write_dataset
write_dataset('Putnam_park2023_run2_1.csv', 5)
