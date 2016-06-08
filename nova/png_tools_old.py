import itertools

class sample_plot:
    def __init__(self, data, x_origin, y_origin, x_ref, y_ref,
                 x_fig, y_fig, ax, fig, path, file, data_ID):

        self.cord_flag = 0
        self.count_data = itertools.count(1)
        self.data = data
        self.path = path
        self.file = file
        self.x = []
        self.y = []
        self.x_fig = x_fig
        self.y_fig = y_fig
        self.x_origin = x_origin
        self.y_origin = y_origin
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.ax = ax
        self.fig = fig
        self.cycle_click = itertools.cycle([1,2])
        self.cycle_axis = itertools.cycle([1,2,3,4])
        self.count_click = next(self.cycle_click)
        self.count_axis = next(self.cycle_axis)
        self.data_ID = data_ID
        self.cid = x_origin.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):

        if event.button == 1:  # select button

            if self.count_click == 1:  # enter axis
                
                if self.count_axis == 1:
                    self.x_origin.set_data(event.xdata, event.ydata)
                    self.ax.set_title('select y-origin (y='+str(self.y_fig[0])+')')
                    self.x_origin.figure.canvas.draw()
                    
                elif self.count_axis == 2:
                    self.y_origin.set_data(event.xdata, event.ydata)
                    self.ax.set_title('select x referance (x='+str(self.x_fig[1])+')')
                    self.y_origin.figure.canvas.draw()
                    
                elif self.count_axis == 3:
                    self.x_ref.set_data(event.xdata, event.ydata)
                    self.ax.set_title('select y referance (y='+str(self.y_fig[1])+')')
                    self.x_ref.figure.canvas.draw()
                    
                else:
                    self.y_ref.set_data(event.xdata, event.ydata)
                    self.ax.set_title('select data (select|finish|undo)')
                    self.y_ref.figure.canvas.draw()
                    self.count_click = next(self.cycle_click)
                    
                self.count_axis = next(self.cycle_axis)
                
            else:  # enter data
                self.x.append(event.xdata)
                self.y.append(event.ydata)
                self.data.set_data(self.x, self.y)
                self.y_ref.figure.canvas.draw()    
                
        if event.button == 2:  # enter button
        
            if self.count_click == 2:  # exit and save
                if len(self.x) == 0:
                    self.fig.canvas.mpl_disconnect(self.cid)
                else:
                    if self.cord_flag == 0:
                        self.set_cord()
                        self.cord_flag = 1
                    self.process_data()
                    self.x, self.y = [], []  # reset
                    self.data.set_data(self.x, self.y)
                    self.data.figure.canvas.draw()

        if event.button == 3:  # remove data points

            if self.count_click == 2: 

                if len(self.x) > 0:
                    self.x.pop(len(self.x)-1)
                if len(self.y) > 0:    
                    self.y.pop(len(self.y)-1)

                self.data.set_data(self.x, self.y)
                self.data.figure.canvas.draw() 
                
    def set_cord(self):
        
        self.x_o = self.x_origin.get_xydata()[0]
        self.x_o[1] = -self.x_o[1]
        
        self.y_o = self.y_origin.get_xydata()[0]
        self.y_o[1] = -self.y_o[1]
        
        x_ref = self.x_ref.get_xydata()[0][0]
        y_ref = -self.y_ref.get_xydata()[0][1]
        
        self.x_scale = (self.x_fig[1]-self.x_fig[0]) / (x_ref-self.x_o[0])
        self.y_scale = (self.y_fig[1]-self.y_fig[0]) / (y_ref-self.y_o[1])
                
    def process_data(self):

        data = self.data.get_xydata()
        data[:,1] = -data[:,1]
        data[:,0] = self.x_scale*(data[:,0]-self.x_o[0])+self.x_fig[0]
        data[:,1] = self.y_scale*(data[:,1]-self.y_o[1])+self.y_fig[0]

        import shelve
        var = shelve.open(self.path+self.file+'_'+self.data_ID)  
        name = 'data-'+str(next(self.count_data))
        var[name] = data
        var.close()
        
def data_mine(path, file, data_ID, x_fig, y_fig):

    from matplotlib import pyplot as plt
    from png_tools import sample_plot
    import matplotlib.image as mpimg
    
    fig = plt.figure(figsize=(22, 16))
    ax = fig.add_subplot(111)
    
    origin = 'upper'
    image = mpimg.imread(path+file+'.png')
    ax.imshow(image, origin=origin)
    print(x_fig)
    ax.set_title('select x-origin (x='+str(x_fig[0])+')')
    
    #default markers
    data, = ax.plot([0], [0], 'gs-')
    x_origin, = ax.plot([0], [0], 'ro') 
    y_origin, = ax.plot([0], [0], 'bo') 
    x_ref, = ax.plot([0], [0], 'rx') 
    y_ref, = ax.plot([0], [0], 'bx')
     
    sample_plot = sample_plot(data, x_origin, y_origin, x_ref, y_ref, x_fig, y_fig,
                              ax, fig, path, file, data_ID)
    
    plt.show()