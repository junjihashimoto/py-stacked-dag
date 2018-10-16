import json
import stackeddag.core as sd

def fromMXNET(json_file):
    f = open(json_file, 'r')
    json_dict = json.load(f)
    la = []
    for i in json_dict['nodes']:
        la.append((i, json_dict['nodes'][i]['name']))
    ed = []
    for i in json_dict['nodes']:
        for j in json_dict['nodes'][i]['inputs']:
            ed.append((json_dict['nodes'][i]['inputs'][j][0], i))
    return sd.edgesToText(sd.mkLabels(la), sd.mkEdges(ed))
