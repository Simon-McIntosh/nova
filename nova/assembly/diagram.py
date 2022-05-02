
from IPython.display import Image, display

import dot2tex
import pydot

graph = pydot.Dot('my_graph', graph_type='digraph', bgcolor='white',
                  rankdir='LR')

graph.add_node(pydot.Node('dr_case', label='U(+-1.5)', shape='box3d'))
graph.add_node(pydot.Node('dt_case', label='U(+-1.5)', shape='box3d'))

graph.add_node(pydot.Node('dr_ccl', label='N(0, 2)', shape='box3d'))
graph.add_node(pydot.Node('dt_ccl', label='N(0, 2)', shape='box3d'))

graph.add_edge(pydot.Edge('dr_case', 'build', label=' dR case'))
graph.add_edge(pydot.Edge('dt_case', 'build', label=' RdPhi case'))

graph.add_node(pydot.Node('build', label='Assembly', shape='rectangle'))
graph.add_node(pydot.Node('ansys', label='Structural', shape='rectangle'))
graph.add_node(pydot.Node('em', label='Electromagnetic', shape='rectangle'))

graph.add_edge(pydot.Edge('build', 'ansys', label=' gap'))

graph.add_node(pydot.Node('add_r', label='+', shape='circle'))
graph.add_node(pydot.Node('add_t', label='+', shape='circle'))

graph.add_edge(pydot.Edge('ansys', 'add_r', label=' dR gap'))
graph.add_edge(pydot.Edge('dr_case', 'add_r', label=' dR case'))
graph.add_edge(pydot.Edge('dr_ccl', 'add_r', label=' dR ccl'))

graph.add_edge(pydot.Edge('ansys', 'add_t', label=' RdPhi gap'))
graph.add_edge(pydot.Edge('dt_ccl', 'add_t', label=' RdPhi ccl'))

graph.add_edge(pydot.Edge('add_r', 'em', label=' dR em'))
graph.add_edge(pydot.Edge('add_t', 'em', label=' RdPhi em'))

graph.add_node(pydot.Node('ph', label='P(H)', shape='none'))

if False:
    graph.add_node(pydot.Node('dr_blanket', label='U(+-4.5)', shape='box3d'))
    graph.add_node(pydot.Node('blanket', label='Blanket', shape='rectangle'))

    graph.add_edge(pydot.Edge('dr_blanket', 'blanket', label=' dR blanket'))

    graph.add_node(pydot.Node('sub_h', label='-', shape='circle'))
    graph.add_edge(pydot.Edge('blanket', 'sub_h', label=' dR blanket n1'))
    graph.add_edge(pydot.Edge('em', 'sub_h', label=' h(Phi)'))

    graph.add_edge(pydot.Edge('sub_h', 'ph'))
else:
    graph.add_edge(pydot.Edge('em', 'ph', label=' h(Phi)'))







display(Image(graph.create_png()))





'''
gtex = dot2tex.dot2tex(graph.to_string(), format='tikz',
                       texmode='math', crop=True)

import subprocess

with tempfile.NamedTemporaryFile('w', suffix='.tex') as tmp:
    tmp.write(gtex)
    proc = subprocess.Popen(['pdflatex', tmp.name])
    proc.communicate()

print(graph.to_string())
print(gtex)
#with open('tmp.txt', 'w') as f:
#    f.write(gtex)
'''

'''



'''

'''
, IFrame

#graph.set_type()



IFrame(pdfl, width=600, height=300)
'''
