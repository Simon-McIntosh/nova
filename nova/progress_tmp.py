import time
import sys

for i in range(100):
    time.sleep(1)
    sys.stdout.write('\r{:d}%'.format(i))
    sys.stdout.flush()