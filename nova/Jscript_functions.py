class JSfunc(object):
    
    def __init__(self, f):
        from itertools import count
        self.planeID = count(1)
        self.functionID = count(1)
        self.sketchID = count(1)
        self.splineID = count(1)
        self.psID = count(1)
        self.FP = count(1)
        self.circleID = count(1)
        self.lineID = count(1)
        self.arcID = count(1)
        self.f = f
        
        self.f.write('//clear session\n')
        self.f.write('ag.m.NewSession(1) \n\n')
        self.f.write('//set units\n')
        self.f.write('agb.SetSessionUnits(ag.c.UnitMeter, ag.c.UnitDegree, ag.c.No);\n\n')
        
    def open_function(self):   
        functionID = 'plane'+str(next(self.functionID))+'SketchesOnly'
        self.f.write('function '+functionID+' (p)\n')
        self.f.write('{\n')
        self.f.write('\n')
        self.set_plane_active()
        return functionID
            
    def close_function(self):
        self.f.write('p.Plane.EvalDimCons();\n')
        self.f.write('return p;\n')
        self.f.write('}\n')
        self.f.write('\n')
    
    def end_script(self):
        self.f.write('//Finish\n')
        self.f.write('agb.Regen(); //To insure model validity\n')
        self.f.write('//End DM JScrip\n')
        self.f.write('ag.gui.ZoomFit();\n')
        self.f.write('agb.Regen(); //To insure model validity\n')
        self.f.write('\n')
        self.f.close()
            
    def set_plane_active(self):
        self.f.write('//Plane\n')
        self.f.write('p.Plane  = agb.GetActivePlane();\n')
        self.f.write('p.Origin = p.Plane.GetOrigin();\n')
        self.f.write('p.XAxis  = p.Plane.GetXAxis();\n')
        self.f.write('p.YAxis  = p.Plane.GetYAxis();\n')
        self.f.write('\n')
        
    def get_plane(self, ID):
        plane = ID+'Plane'
        self.f.write('//Call Plane JScript function\n')
        self.f.write('var '+plane+' = agb.Get'+ID+'Plane();\n')
        self.f.write('\n')
        return plane
        
    def trans_plane(self, ax, from_plane, offset, **kw):
        to_plane = 'plane'+str(next(self.planeID))
        self.f.write('var '+to_plane+' = agb.PlaneFromPlane('+from_plane+');\n')
        self.f.write(to_plane+'.AddTransform(agc.Xform'+ax+', '+str(offset)+');\n')
        if 'name' in kw.keys(): self.f.write(to_plane+'.name = "'+kw['name']+'"\n')
        self.f.write('agb.regen();\n')
        self.f.write('\n')
        return to_plane
        
    def sketch(self, plane, function):
        ps = 'ps'+str(next(self.psID))
        self.f.write('agb.SetActivePlane ('+plane+')\n')
        self.f.write('var '+ps+' = '+function+' (new Object());\n')
        self.f.write('\n')
        return ps
        
    def start_sketch(self, **kw):
        sketchID = 'Sk'+str(next(self.sketchID))
        self.f.write('//Sketch\n')
        self.f.write('p.'+sketchID+' = p.Plane.NewSketch();\n')
        if 'name' in kw.keys(): 
            self.f.write('p.'+sketchID+'.name = "'+kw['name']+'_sketch"\n')
        self.f.write('\n')    
        self.f.write('//Edges\n')
        self.f.write('with (p.'+sketchID+')\n')
        self.f.write('{\n')
        return sketchID
        
    def end_sketch(self):
        self.f.write('}\n')
        
    def sweep(self, profile_ps, profile_sk, path_ps, path_sk, ac, **kw):
        self.f.write('//Next create a Sweep\n')
        self.f.write('var Sweep = agb.Sweep(agc.'+ac+', '+profile_ps+'.'+profile_sk+',\n')
        self.f.write('                       '+path_ps+'.'+path_sk+', agc.AlignTangent,\n')
        self.f.write('                       1.0, 0.0, agc.No, 0.0, 0.0);\n')
        if 'name' in kw.keys(): self.f.write('Sweep.Name = "'+kw['name']+'_sweep"\n') 
        self.f.write('agb.Regen(); //To insure model validity \n')
        
        if 'name' in kw.keys():  # name bodies
            if 'pancake' in kw['name']: 
                name = 'pancake'
            else:
                name = kw['name']
                
            self.f.write('//name bodies\n')
            self.f.write('var N'+name+' = 0;\n')
            self.f.write('for (var i =0; i <ag.agApplet.FeatureMgr.Bodycount; i++) {\n')
            self.f.write('    var Body = ag.agApplet.FeatureMgr.Body(i)\n')
            self.f.write('    if(Body.Name=="Solid")\n')
            self.f.write('        {\n')
            self.f.write('         N'+name+'+= 1;\n')
            rename_str = '            ag.agApplet.FeatureMgr.Body(i).Name = "'+kw['name']+'"'
            if kw['name'] is 'conductor': rename_str += '+"_"+i.toString()'
            rename_str += ';\n' 
            self.f.write(rename_str)
            self.f.write('    }\n')
            self.f.write('}\n')
        
    def bodyNS(self, name):
        Nchar = len(name)
        self.f.write('//Loop over all bodies\n')
        self.f.write('ag.m.SelectionMgr.ClearSelection();\n')  # clear selection
        self.f.write('for (var i =0; i <ag.agApplet.FeatureMgr.Bodycount; i++) {\n')
        self.f.write('    var Body = ag.agApplet.FeatureMgr.Body(i)\n')
        self.f.write('    if(Body.Name.slice(0,'+str(Nchar)+')=="'+name+'") {\n')
        self.f.write('        agb.AddSelect(ag.c.TypeBody, Body);\n')  # select body
        self.f.write('    }\n')
        self.f.write('}\n')
        self.f.write('var select = ag.gui.CreateSelectionSet();\n')
        self.f.write('select.Name = "'+name+'_body";\n')
        
    def faceNS(self, name):
        Nchar = len(name)
        for i in range(8):
            self.f.write('//Loop over all bodies\n')
            self.f.write('var Face = new Array(N'+name+');\n')
            self.f.write('var j = 0;\n')
            self.f.write('ag.m.SelectionMgr.ClearSelection();\n')  # clear selection
            self.f.write('for (var i =0; i <ag.agApplet.FeatureMgr.Bodycount; i++) {\n')
            self.f.write('    var Body = ag.agApplet.FeatureMgr.Body(i)\n')
            self.f.write('    if(Body.Name.slice(0,'+str(Nchar)+')=="'+name+'") {\n')
            self.f.write('        ag.m.FindEntity(0, 0, Body.Label, false);\n')  # select body
            self.f.write('        ag.m.SelectionMgr.ShowTopologicalConnections(0);\n')  # select faces connected to body
            self.f.write('        Face[j] = ag.m.GetFaceSelListIndex('+str(i)+');\n')
            self.f.write('        j += 1;\n')
            self.f.write('    }\n')
            self.f.write('}\n')
            
            self.f.write('ag.m.SelectionMgr.ClearSelection();\n')  # clear selection
            self.f.write('for (var j =0; j < N'+name+'; j++) {\n')
            self.f.write('    agb.AddSelect(ag.c.TypeFace, Face[j]);\n')  # select face
            self.f.write('}\n')
            
            self.f.write('var select = ag.gui.CreateSelectionSet();\n')
            self.f.write('select.Name = "'+name+'_face_in_'+str(i)+'";\n')
            self.f.write('ag.m.SelectionMgr.ClearSelection();\n')  # clear selection
        
    def named_selection(self, path, file):
        with open(path+file, 'r') as f:
            for line in f:
                self.f.write(line)
            
    def circle(self, x, y, r, dr=0):
        r += dr
        circleID = 'p.Cr'+str(next(self.circleID))
        self.f.write('    '+circleID+' = Circle('+str(x)+', '+str(y)+', '+str(r)+');\n')
        
    def circle_arc(self, x, y, r, dr=0):
        import numpy as np
        r += dr
        N = 4
        origin = [x,y]
        for i in range(N):
            p1 = [x+r*np.cos(2*np.pi/N*i), y+r*np.sin(2*np.pi/N*i)]
            p2 = [x+r*np.cos(2*np.pi/N*(i+1)), y+r*np.sin(2*np.pi/N*(i+1))]
            self.arc([origin,p1,p2])

   
    def line(self, points):
        lineID = 'p.Ln'+str(next(self.lineID))
        string = '    '+lineID+' = Line('
        for i in range(2):
            string += str(points[i][0])+', '+str(points[i][1])+', '
        self.f.write(string[:-2]+');\n')
        
    def square(self, x, y, r, dr=0):
        import numpy as np
        a = np.sqrt(np.pi)*r+2*dr  # side length
        p1 = [x+a/2,y-a/2]
        p2 = [x+a/2,y+a/2]
        p3 = [x-a/2,y+a/2]
        p4 = [x-a/2,y-a/2]
        self.line([p1,p2])
        self.line([p2,p3])
        self.line([p3,p4])
        self.line([p4,p1])
    
    def rectangle(self, x, y, a):
        p1 = [x+a[1]/2,y-a[0]/2]
        p2 = [x+a[1]/2,y+a[0]/2]
        p3 = [x-a[1]/2,y+a[0]/2]
        p4 = [x-a[1]/2,y-a[0]/2]
        self.line([p1,p2])
        self.line([p2,p3])
        self.line([p3,p4])
        self.line([p4,p1])  
        
    def multi_rectangle(self, x, y, a):
        p1 = [x,y-a[0]/2]
        p2 = [x+a[1]/4,y-a[0]/2]
        p3 = [x+a[1]/2,y-a[0]/2]
        p4 = [x+a[1]/2,y+a[0]/2]
        p5 = [x+a[1]/4,y+a[0]/2]
        p6 = [x,y+a[0]/2]
        p7 = [x-a[1]/4,y+a[0]/2]
        p8 = [x-a[1]/2,y+a[0]/2]
        p9 = [x-a[1]/2,y-a[0]/2]
        p10 = [x-a[1]/4,y-a[0]/2]
        self.line([p1,p2])
        self.line([p2,p3])
        self.line([p3,p4])
        self.line([p4,p5])
        self.line([p5,p6])
        self.line([p6,p7])
        self.line([p7,p8])
        self.line([p8,p9])
        self.line([p9,p10])
        self.line([p10,p1])
        
    def arc(self, points):
        arcID = 'p.Cr'+str(next(self.arcID))
        string = '    '+arcID+' = ArcCtrEdge('
        for i in range(3):
            string += str(points[i][0])+', '+str(points[i][1])+', '
        self.f.write(string[:-2]+');\n')        
        
    def multi_arc(self, points, N):
        import numpy as np
        r = np.sqrt((points[0][0]-points[1][0])**2+
                    (points[0][1]-points[1][1])**2)   
        c = np.sqrt((points[1][0]-points[2][0])**2+
                    (points[1][1]-points[2][1])**2)
        theta_o = np.arctan2(points[0][1]-points[1][1],points[0][0]-points[1][0])
        theta = 2*np.arcsin(c/(2*r))
        dtheta = theta/N
        nPoint = points
        for n in range(N):
            for m,p in zip(range(1,-1,-1),range(2)):
                theta_m = theta_o+(n+m)*dtheta
                nPoint[1+m][0] = points[0][0]-r*np.cos(theta_m)
                nPoint[1+m][1] = points[0][1]-r*np.sin(theta_m) 
            arcID = 'p.Cr'+str(next(self.arcID))
            string = '    '+arcID+' = ArcCtrEdge('
            for i in range(3):
                string += str(nPoint[i][0])+', '+str(nPoint[i][1])+', '
            self.f.write(string[:-2]+');\n')  
        
    def spline(self, sketchID, spline_x, spline_y):
        splineID = 'p.Sp'+str(next(self.splineID))
        self.f.write('//Edges\n')
        self.f.write('with (p.'+sketchID+')\n')
        self.f.write('{\n')
        self.f.write('    '+splineID+' = SplineBegin();\n')
        self.f.write('    with('+splineID+')\n')
        self.f.write('    {\n')
        for x,y in zip(spline_x, spline_y):
            self.f.write('        SplineXY('+str(x)+', '+str(y)+');\n')    
        self.f.write('        SplineFitPtEnd();\n')
        self.f.write('    }\n')
        self.f.write('    with('+splineID+')\n')
        self.f.write('    {\n')
        ToID = ['p.XAxis']
        for x,y in zip(spline_x[1:-1], spline_y[1:-1]):
            ToID.append(''+splineID+'_Fit'+str(next(self.FP)))
            self.f.write('        '+ToID[-1]+' = CreateSplineFitPoint('
                    +str(x)+', '+str(y)+');\n')  
        self.f.write('    }\n')
        self.f.write('}\n')
        
        self.f.write('//Dimensions and/or constraints\n')
        self.f.write('with (p.Plane)\n')
        self.f.write('{\n')
        self.f.write('    //Constraints\n')
        first = 1
        '''
        for x,y,toID in zip(spline_x[:-1], spline_y[:-1], ToID):
            if first == 1:
                self.f.write('    CoincidentCon('+splineID+'.Base, '+str(x)+', '+str(y)+',\n')
                self.f.write('                  '+toID+', '+str(x)+', '+str(y)+');\n')
                first = 0
            else:
                self.f.write('    CoincidentConLock('+splineID+', '+str(x)+', '+str(y)+',\n')
                self.f.write('                  '+toID+', '+str(x)+', '+str(y)+', 1);\n')
        '''
        self.f.write('}\n')
        self.f.write('\n')