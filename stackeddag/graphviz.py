import stackeddag.core as sd
import pydot

def fromDotFile (dot_file):
  v = pydot.graph_from_dot_file(dot_file)
  g = v[0]
  edges = g.get_edges()
  nodes = g.get_nodes()
  la = []
  for i in nodes:
    attr = i.get_attributes()
    if 'label' in attr:
      la.append((i.get_name(),attr['label'][1:-1]))
  ed = []
  for i in edges:
    ed.append((i.get_source(),[i.get_destination()]))
  return sd.edgesToText(sd.mkLabels(la),sd.mkEdges(ed))
