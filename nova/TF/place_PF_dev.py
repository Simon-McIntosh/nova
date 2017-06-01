
excl = R.define_port_exclusions(plot=True)  # very nice
Lex = R.TF.xzL(excl)  # translate to normalized coil length

print(Lex)
print(inv.Lo['value'])

pf.plot(label=True)


# inv.optimize()
inv.set_sail(Lex)

print(inv.Lo['value'])

print(inv.Lo)



