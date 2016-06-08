from extract import ex

config = 'DD1'  # 'DD1' 
extract = ex(config)

coil = extract.coil
plasma = extract.plasma

print(plasma['LCFS'])
    
    
    
    