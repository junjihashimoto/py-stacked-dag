import re
import stackeddag.core as sd

def fromGraph(graph):
    graphdef = graph.as_graph_def()
    la = []
    ed = []
    for node in graphdef.node:
        output_name = node.name
        la.append((output_name, output_name))
        for input_full_name in node.input:
            parts = input_full_name.split(":")
            input_name = re.sub(r"^\^", "", parts[0])
            ed.append((input_name, [output_name]))
    return sd.edgesToText(sd.mkLabels(la), sd.mkEdges(ed))
