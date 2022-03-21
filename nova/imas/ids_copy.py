
import subprocess


result = subprocess.run(['scenario_summary', '-s', 'CORSICA',
                         '-c', 'shot,run,workflow'],
                         capture_output=True, text=True)
