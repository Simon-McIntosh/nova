import json
from amigo.IO import trim_dir
from nova.shelf import PKL



config = 'SNdtt_18TF_5PF_3CS'
config = 'SXdtt_18TF_5PF_3CS'
pkl = PKL(config,directory='../../Movies/')
sf,eq,inv = pkl.fetch(['sf','eq','inv'])

'''
datadir = trim_dir('../../../Data/') 
with open(datadir+'geom_SNdtt18.json','r') as f:
    data = json.load(f)
'''