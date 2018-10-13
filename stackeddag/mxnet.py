import stackeddag.core as sd
import json

def fromMXNET(json_file):
  f = open(json_file, 'r')
  json_dict = json.load(f)
  labels=[]
  for i in json_dict['nodes']:
    la.append((i,json_dict['nodes'][i]['name']))
  edges=[]
  for i in json_dict['nodes']:
    for j in json_dict['nodes'][i]['inputs']:
      edges.append((json_dict['nodes'][i]['inputs'][j][0],i))
  return sd.edgesToText(sd.mkLabels(la),sd.mkEdges(ed))
