from nep.DINA.coil_force import coil_force

folder = "MD_UP_exp16"
frame_index = 150
force = coil_force(mode="control", Ip_scale=1, read_txt=False)
folder = force.dina.folders.index(folder)

force.load_file(scenario=folder, read_txt=False)

force.frame_update(frame_index=frame_index)
force.contour()
force.pf.plot(plasma=True, subcoil=True)
