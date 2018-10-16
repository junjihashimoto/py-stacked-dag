import pydot
import stackeddag.core as sd

def fromDotFile(dot_file: str) -> str:
    v = pydot.graph_from_dot_file(dot_file)
    g = v[0]
    edges = g.get_edges()
    nodes = g.get_nodes()
    la = []
    for i in nodes:
        attr = i.get_attributes()
        if 'label' in attr:
            name = i.get_name()
            label = attr['label'][1:-1]
            la.append((name, label))
    ed = []
    for i in edges:
        src = i.get_source()
        dst = i.get_destination()
        ed.append((src, [dst]))
    return sd.edgesToText(sd.mkLabels(la), sd.mkEdges(ed))
