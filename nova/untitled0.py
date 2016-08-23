from amigo.IO import PATH,qsub

jobname = 'mag_shape'
script = 'test.py'

qsub(script,jobname,t=3) 

#path = PATH('test')