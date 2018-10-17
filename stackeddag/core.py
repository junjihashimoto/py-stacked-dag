from typing import Callable, TypeVar, List, Dict, Tuple, Set, Iterator, Optional
from enum import Enum
from functools import reduce

T = TypeVar('T')
Labels = Dict[T, str]
Edges = Dict[T, Set[T]]
NodeDepth = Dict[T, int]
DepthGroup = Dict[int, List[T]]
DepthGroupWithSkip = Dict[int, Tuple[List[T], List[T]]]
DepthGroupWithDest = Dict[int, Tuple[List[Tuple[T, int, int]], List[Tuple[T, int, int]]]]

class SEnum(Enum):
    SNode = 0
    SLeft = 1
    SRight = 2
    SHold = 3
    SLMove = 4
    SRMove = 5
    SCross = 6
    SSpace = 7

class Symbol:
    def __init__(self, senum, label=""):
        self.sym = senum
        self.label = label

    def __str__(self) -> str:
        if self.sym == SEnum.SNode:
            return f"SNode \"{self.label}\""
        elif self.sym == SEnum.SHold:
            return "SHold"
        elif self.sym == SEnum.SLeft:
            return "SLeft"
        elif self.sym == SEnum.SRight:
            return "SRight"
        elif self.sym == SEnum.SLMove:
            return "SLMove"
        elif self.sym == SEnum.SRMove:
            return "SRMove"
        elif self.sym == SEnum.SCross:
            return "SCross"
        elif self.sym == SEnum.SSpace:
            return "SSpace"
        else:
            return "SOther"

    def __repr__(self) -> str:
        if self.sym == SEnum.SNode:
            return f"SNode \"{self.label}\""
        elif self.sym == SEnum.SHold:
            return "SHold"
        elif self.sym == SEnum.SLeft:
            return "SLeft"
        elif self.sym == SEnum.SRight:
            return "SRight"
        elif self.sym == SEnum.SLMove:
            return "SLMove"
        elif self.sym == SEnum.SRMove:
            return "SRMove"
        elif self.sym == SEnum.SCross:
            return "SCross"
        elif self.sym == SEnum.SSpace:
            return "SSpace"
        else:
            return "SOther"

def SNode(label: str) -> Symbol:
    return Symbol(SEnum.SNode, label)

SSpace = Symbol(SEnum.SSpace)
SHold = Symbol(SEnum.SHold)
SRight = Symbol(SEnum.SRight)
SLeft = Symbol(SEnum.SLeft)
SLMove = Symbol(SEnum.SLMove)
SCross = Symbol(SEnum.SCross)

def addsym(symbol0: Symbol, symbol1: Symbol) -> Symbol:
    if symbol0.sym == SEnum.SNode:
        return symbol0
    elif symbol1.sym == SEnum.SNode:
        return symbol1
    elif symbol1.sym == SEnum.SSpace:
        return symbol0
    elif symbol0.sym == SEnum.SSpace:
        return symbol1
    elif symbol0.sym == SEnum.SLeft and symbol1.sym == SEnum.SRight:
        return SCross
    elif symbol1.sym == SEnum.SLeft and symbol0.sym == SEnum.SRight:
        return SCross
    elif symbol0.sym == SEnum.SCross and symbol1.sym == SEnum.SRight:
        return SCross
    elif symbol0.sym == SEnum.SCross and symbol1.sym == SEnum.SLeft:
        return SCross
    elif symbol1.sym == SEnum.SCross and symbol0.sym == SEnum.SRight:
        return SCross
    elif symbol1.sym == SEnum.SCross and symbol0.sym == SEnum.SLeft:
        return SCross
    return symbol0

def groupby(fn: Callable[[T, T], bool], lists: List[T]) -> List[List[T]]:
    """
    >>> groupby(lambda x,y: x[0] == y[0] ,[(2, 0), (2, 1), (3, 2), (3, 4), (3, 6), (5, 3)])
    [[(2, 0), (2, 1)], [(3, 2), (3, 4), (3, 6)], [(5, 3)]]
    >>> groupby(lambda x,y: (x*y) % 3 == 0,[1,2,3,4,5,6,7,8,9] )
    [[1], [2, 3], [4], [5, 6], [7], [8, 9]]
    """
    if not lists:
        return []

    r = []
    j = lists[0]
    buf = [j]
    for i in lists[1:]:
        if fn(j, i):
            buf.append(i)
        else:
            r.append(buf)
            buf = [i]
            j = i
    r.append(buf)
    return r

def iconcat(list_of_list: Iterator[List[T]]) -> List[T]:
    return sum(list_of_list, [])

def concat(list_of_list: List[List[T]]) -> List[T]:
    return sum(list_of_list, [])

def mkEdges(list_of_edges: List[Tuple[T, List[T]]]) -> Edges[T]:
    """
    >>> sorted(mkEdges([(0,[2]),(1,[2]),(2,[3]),(4,[3]),(6,[3]),(3,[5])]).items())
    [(0, {2}), (1, {2}), (2, {3}), (3, {5}), (4, {3}), (6, {3})]
    """
    g = groupby(lambda x, y: x[0] == y[0], sorted(list_of_edges))
    m = map(lambda x: (x[0][0], set(iconcat(map(lambda y: y[1], x)))), g)
    return dict(m)

def mkLabels(list_of_labels: List[Tuple[T, str]]) -> Labels[T]:
    """
    >>> sorted(mkLabels([(0,"l0"),(1,"l1"),(2,"l2"),(3,"l3"),(5,"l5"),(4,"l4"),(6,"l6")]).items())
    [(0, 'l0'), (1, 'l1'), (2, 'l2'), (3, 'l3'), (4, 'l4'), (5, 'l5'), (6, 'l6')]
    """
    return dict(list_of_labels)


sampledat = mkEdges([
    (0, [2]),
    (1, [2]),
    (2, [3]),
    (4, [3]),
    (6, [3]),
    (3, [5])
])

samplelabels = mkLabels([
    (0, "l0"),
    (1, "l1"),
    (2, "l2"),
    (3, "l3"),
    (5, "l5"),
    (4, "l4"),
    (6, "l6")
])


def _getDepth2(edges: Edges[T], dmap: NodeDepth[T], v: T) -> int:
    if v in dmap:
        return dmap[v]
    else:
        if v in edges:
            m = 1 + max(map(lambda x: _getDepth2(edges, dmap, x), list(edges[v])))
            dmap[v] = m
            return m
        else:
            dmap[v] = 0
            return 0

def getNodes(edges: Edges[T]) -> Set[T]:
    """
    >>> getNodes(mkEdges([(0,[2]),(1,[2]),(2,[3]),(4,[3]),(6,[3])]))
    {0, 1, 2, 3, 4, 6}
    """
    parents = map(lambda x: x[0], edges.items())
    children = iconcat(map(lambda x: list(x[1]), edges.items()))
    r = list(parents)
    r.extend(children)
    return set(r)

def getDepth2(edges: Edges[T]) -> NodeDepth[T]:
    """
    >>> sorted(getDepth2(sampledat).items())
    [(0, 3), (1, 3), (2, 2), (3, 1), (4, 2), (5, 0), (6, 2)]
    """
    dmap: Dict[T, int] = dict([])
    for i in list(getNodes(edges)):
        _getDepth2(edges, dmap, i)
    return dmap


def pairs(edges: Edges[T]) -> List[Tuple[T, T]]:
    r = []
    for i in edges.items():
        for child in list(i[1]):
            r.append((child, i[0]))
    return r

def reverseEdges(edges: Edges[T]) -> Edges[T]:
    """
    | Reverse the directions of edges
    >>> reverseEdges(sampledat)
    {2: {0, 1}, 3: {2, 4, 6}, 5: {3}}
    """
    s = sorted(pairs(edges), key=lambda x: x[0])
    g = groupby(lambda x, y: x[0] == y[0], s)
    m = map(lambda x: (x[0][0], set(map(lambda y: y[1], x))), g)
    return dict(m)

def getDepthGroup2(labels: Labels[T], edges: Edges[T]) -> DepthGroup[T]:
    """
    >>> getDepthGroup2(samplelabels,sampledat)
    {0: [5], 1: [3], 2: [2, 4, 6], 3: [0, 1]}
    """
    d0 = getDepth2(edges)
    d1 = getDepth2(reverseEdges(edges))
    def score(nid):
        s = 0
        if nid in d0:
            s += d0[nid]
        if nid in d1:
            s += d1[nid]
        return s
    def _sort(nodes):
        return sorted(list(nodes), key=lambda x: (-score(x), lstr(labels, x)))
    d2n = map(lambda x: (x[1], x[0]), list(d0.items()))

    s = sorted(d2n, key=lambda x: x[0])
    g = groupby(lambda x, y: x[0] == y[0], s)
    m = map(lambda x: (x[0][0], _sort(set(map(lambda y: y[1], x)))), g)
    return dict(m)

def getNodeDepth(dg: DepthGroup[T]) -> NodeDepth[T]:
    d2n = dg.items()
    n2d = map(lambda x: list(map(lambda node: (node, x[0]), x[1])), d2n)
    return dict(concat(list(n2d)))

def moveOne(nodes: List[Tuple[T, int, int]]) -> List[Tuple[Tuple[T, int, int], List[Tuple[Symbol, int]]]]:
    """
    Move nodes to next step
    >>> moveOne([(0,0,4)])
    [((0, 2, 4), [(SRight, 1)])]
    >>> moveOne([(0,0,4),(0,4,0)])
    [((0, 2, 4), [(SRight, 1)]), ((0, 2, 0), [(SLeft, 3)])]
    """
    r = []
    for i in nodes:
        (n, c, g) = i
        if c < g:
            r.append(((n, c+2, g), [(SRight, c+1)]))
        else:
            if c > g:
                r.append(((n, c-2, g), [(SLeft, c-1)]))
            else:
                r.append(((n, c, g), [(SHold, c)]))
    return r

def takeNode(c: int, nodes: List[Tuple[Tuple[T, int, int], List[Tuple[Symbol, int]]]]) -> Optional[Tuple[Tuple[T, int, int], List[Tuple[Symbol, int]]]]:
    for i in nodes:
        for j in i[1]:
            if j[1] == c:
                return i
    return None


def _moveLeft(nodes):
    """
    >>> _moveLeft( [((0,0,0),[(SHold,0)]),((1,0,0),[(SLeft,1)]),((2,2,0),[(SLeft,3)])])
    [((0, 0, 0), [(SHold, 0)]), ((1, 0, 0), [(SLeft, 1)]), ((2, 0, 0), [(SLMove, 2), (SLeft, 3)])]
    >>> _moveLeft( [((0,0,0),[(SHold,0)]),\
                    ((1,0,0),[(SLeft,1)]),\
                    ((2,0,0),[(SLMove,2),(SLeft,3)]),\
                    ((3,2,0),[(SLMove,4),(SLeft,5)])])
    [((0, 0, 0), [(SHold, 0)]), ((1, 0, 0), [(SLeft, 1)]), ((2, 0, 0), [(SLMove, 2), (SLeft, 3)]), ((3, 0, 0), [(SLMove, 4), (SLeft, 5)])]
    >>> _moveLeft( [((0,2,0),[(SLeft,3)])])
    [((0, 0, 0), [(SLMove, 1), (SLMove, 2), (SLeft, 3)])]
    """
    r = []
    for nn in nodes:
        ((n, c, g), syms) = nn
        if c > g:
            n0 = takeNode(c, nodes)
            n1 = takeNode(c-1, nodes)
            if n0 is None and n1 is None:
                r.append(((n, c-2, g), concat([[(SLMove, c-1), (SLMove, c)], syms])))
            elif n0 is None and n1 is not None:
                g_ = n1[0][2]
                if g_ == g:
                    r.append(((n, c-2, g), concat([[(SLMove, c)], syms])))
                else:
                    r.append(nn)
            else:
                g_ = n0[0][2]
                if g_ == g:
                    r.append(((n, c-2, g), syms))
                else:
                    r.append(nn)
        else:
            r.append(nn)
    return r

def moveLeft(nodes):
    """
    >>> moveLeft([((0,0,0),[(SHold,0)]),((1,0,0),[(SLeft,1)]),((2,2,0),[(SLeft,3)]),((3,4,0),[(SLeft,5)])])
    [((0, 0, 0), [(SHold, 0)]), ((1, 0, 0), [(SLeft, 1)]), ((2, 0, 0), [(SLMove, 2), (SLeft, 3)]), ((3, 0, 0), [(SLMove, 4), (SLeft, 5)])]
    """
    m = _moveLeft(nodes)
    while nodes != m:
        nodes = m
        m = _moveLeft(nodes)
    return m

def moveAll(nodes, buf):
    """
    | Move nodes to the next depth
    >>> moveAll([(0,0,4)],[])
    [[(SRight, 1)], [(SRight, 3)]]
    >>> moveAll([(0,4,0)],[])
    [[(SLMove, 1), (SLMove, 2), (SLeft, 3)]]
    >>> moveAll([(0,2,0)],[])
    [[(SLeft, 1)]]
    >>> moveAll([(0,0,4),(0,4,0)],[])
    [[(SRight, 1), (SLeft, 3)], [(SRight, 3), (SLeft, 1)]]
    >>> moveAll([(0,0,4),(0,2,0)],[])
    [[(SRight, 1), (SLeft, 1)], [(SRight, 3), (SHold, 0)]]
    """
    flag = True
    for i in nodes:
        (_, c, g) = i
        if c != g:
            flag = False
            break
    if flag and buf != []:
        return buf
    else:
        nn = moveLeft(moveOne(nodes))
        nf = list(map(lambda x: x[0], nn))
        ns = list(map(lambda x: x[1], nn))
        return moveAll(nf, concat([buf, [sum(ns, [])]]))



#mergeSymbol :: [(Symbol b,Pos)] -> [(Symbol b,Pos)]
def mergeSymbol(symbols):
    s = sorted(symbols, key=lambda x: x[1])
    g = groupby(lambda x, y: x[1] == y[1], s)
    m = map(lambda x: (reduce(addsym, map(lambda y: y[0], x)), x[0][1]), g)
    return list(m)

def withSpace(syms):
    """
    | Fill spaces
    >>> withSpace([(SRight,1)])
    [SSpace, SRight]
    >>> withSpace([(SRight,1),(SLeft,3)])
    [SSpace, SRight, SSpace, SLeft]
    >>> withSpace([(SRight,3),(SLeft,1)])
    [SSpace, SLeft, SSpace, SRight]
    """
    s = dict(map(lambda x: (x[1], x[0]), syms))
    m = 0
    for i in syms:
        if m < i[1]:
            m = i[1]
    r = []
    for i in range(0, m+1):
        if i in s:
            r.append(s[i])
        else:
            r.append(SSpace)
    return r

def moveAllWithSpace(nodes):
    """
    | Move nodes and fill spaces
    >>> moveAllWithSpace([(0,0,4)])
    [[SSpace, SRight], [SSpace, SSpace, SSpace, SRight]]
    >>> moveAllWithSpace([(0,4,0)])
    [[SSpace, SLMove, SLMove, SLeft]]
    >>> moveAllWithSpace([(0,0,4),(0,4,0)])
    [[SSpace, SRight, SSpace, SLeft], [SSpace, SLeft, SSpace, SRight]]
    >>> moveAllWithSpace([(0,4,0),(1,0,4)])
    [[SSpace, SRight, SSpace, SLeft], [SSpace, SLeft, SSpace, SRight]]
    """
    m0 = moveAll(nodes, [])
    m1 = map(mergeSymbol, m0)
    m2 = map(lambda x: withSpace(list(x)), m1)
    return list(m2)

def lstr(labels, nodeid):
    if nodeid in labels:
        return labels[nodeid]
    else:
        return ""

def nodeWithSpace(labels, lst):
    (nodes, skipnodes) = lst
    a = map(lambda x: (SNode(lstr(labels, x[0])), x[1]), nodes)
    b = map(lambda x: (SHold, x[1]), skipnodes)
    return withSpace(concat([list(a), list(b)]))

def __addBypassNode(d, edges, nd, dg):
    """
    | Add bypass nodes
    >>> edges = mkEdges([(0,[1,2]),(1,[2])])
    >>> nd = getNodeDepth(getDepthGroup2(dict([]),edges))
    >>> __addBypassNode(2,edges,nd,(dict([(0,([2],[])),(1,([1],[])),(2,([0],[]))])))
    {0: ([2], []), 1: ([1], [0]), 2: ([0], [])}

    >>> edges = mkEdges([(0,[1,3]),(1,[2]),(2,[3])])
    >>> nd = getNodeDepth(getDepthGroup2(dict([]),edges))
    >>> __addBypassNode(3,edges,nd,(dict( [(0,([3],[])),(1,([2],[])),(2,([1],[])),(3,([0],[]))])))
    {0: ([3], []), 1: ([2], []), 2: ([1], [0]), 3: ([0], [])}
    >>> __addBypassNode(2,edges,nd,(dict( [(0,([3],[])),(1,([2],[])),(2,([1],[0])),(3,([0],[]))])))
    {0: ([3], []), 1: ([2], [0]), 2: ([1], [0]), 3: ([0], [])}

    >>> edges = mkEdges([(0,[1,2]),(1,[4]),(2,[3]),(3,[4])])
    >>> nd = getNodeDepth(getDepthGroup2(dict([]),edges))
    >>> __addBypassNode(2,edges,nd,(dict( [(0,([4],[])),(1,([3,1],[])),(2,([2],[0])),(3,([0],[]))])))
    {0: ([4], []), 1: ([3, 1], []), 2: ([2], [0]), 3: ([0], [])}
    """
    def nodeDepth(nid):
        if nid in nd:
            return nd[nid]
        else:
            return 0
    ee = []
    for i in edges.items():
        n = i[0]
        nids = i[1]
        ss = []
        for nid in list(nids):
            if nodeDepth(nid) < d:
                ss.append(nid)
        ee.append((n, set(ss)))
    _edges = dict(ee)
    def _elem(nid, nids):
        if nid in _edges:
            m = list(_edges[nid])
            for n in m:
                if not n in nids:
                    return False
            return True
        else:
            return True

    if d in dg:
        if d-1 in dg:
            (nids0, skipnids0) = dg[d]
            (nids1, skipnids1) = dg[d-1]
            for i in nids0:
                if not _elem(i, nids1):
                    skipnids1.append(i)
            for i in skipnids0:
                if not _elem(i, nids1):
                    skipnids1.append(i)
            dg[d-1] = (nids1, skipnids1)
    return dg


def maxDepth(dg):
    """
    | Get a maximum of depth
    >>> maxDepth (dict( [(0,([2],[])),(1,([1],[])),(2,([0],[]))]))
    2
    """
    return max(map(lambda x: x[0], dg.items()))


def _addBypassNode(edges, nd, dg):
    """
    | Add bypass nodes
    >>> edges = mkEdges([(0,[1,2]),(1,[2])])
    >>> nd = getNodeDepth (getDepthGroup2 (dict([]),edges))
    >>> _addBypassNode(edges,nd,(dict( [(0,([2],[])),(1,([1],[])),(2,([0],[]))])))
    {0: ([2], []), 1: ([1], [0]), 2: ([0], [])}
    >>> edges = mkEdges([(0,[1,3]),(1,[2]),(2,[3])])
    >>> nd = getNodeDepth( getDepthGroup2 (dict([]),edges))
    >>> _addBypassNode(edges,nd,(dict( [(0,([3],[])),(1,([2],[])),(2,([1],[])),(3,([0],[]))])))
    {0: ([3], []), 1: ([2], [0]), 2: ([1], [0]), 3: ([0], [])}
    """
    for d in range(maxDepth(dg), 1, -1):
        dg = __addBypassNode(d, edges, nd, dg)
    return dg

def addBypassNode(edges, nd, dg):
    """
    | Add bypass nodes
    >>> edges = mkEdges([(0,[1,2]),(1,[2])])
    >>> dg = getDepthGroup2(dict([]),edges)
    >>> nd = getNodeDepth(dg)
    >>> addBypassNode(edges,nd,dg)
    {0: ([2], []), 1: ([1], [0]), 2: ([0], [])}
    >>> edges = mkEdges([(0,[1,3]),(1,[2]),(2,[3])])
    >>> dg = getDepthGroup2(dict([]),edges)
    >>> nd = getNodeDepth(dg)
    >>> addBypassNode(edges,nd,dg)
    {0: ([3], []), 1: ([2], [0]), 2: ([1], [0]), 3: ([0], [])}
    >>> edges = mkEdges([(0,[1,2]),(1,[4]),(2,[3]),(3,[4])])
    >>> dg = getDepthGroup2(dict([]), edges)
    >>> nd = getNodeDepth(dg)
    >>> addBypassNode(edges,nd,dg)
    {0: ([4], []), 1: ([3, 1], []), 2: ([2], [0]), 3: ([0], [])}
    """
    return _addBypassNode(edges, nd, dict(map(lambda x: (x[0], (x[1], [])), dg.items())))


def addDest(edges, cur, nxt):
    """
    | Allocate destinations of nodes.
    >>> addDest(sampledat,([0,1],[]),([2],[]))
    ([(0, 0, 0), (1, 2, 0)], [])
    >>> addDest(mkEdges([(0,[1,2]),(1,[2])]), ([0],[]), ([1],[0]))
    ([(0, 0, 0), (0, 0, 2)], [])
    >>> addDest(mkEdges([(0,[1,2]),(1,[2])]), ([1],[0]), ([2],[]))
    ([(1, 0, 0)], [(0, 2, 0)])
    >>> addDest(mkEdges([(0,[1,3]),(1,[2]),(2,[3])]), ([1],[0]),([2],[0]))
    ([(1, 0, 0)], [(0, 2, 2)])
    """
    (curn, curs) = cur
    (nxtn, nxts) = nxt
    i = 0
    n2s = []
    for c in curn:
        ii = len(nxtn)
        for nid in nxts:
            if nid == c:
                n2s.append((c, i*2, ii*2))
                break
            ii = ii + 1
        i = i + 1

    i = len(curn)
    s2s = []
    for c in curs:
        ii = len(nxtn)
        for nid in nxts:
            if nid == c:
                s2s.append((c, i*2, ii*2))
                break
            ii = ii + 1
        i = i + 1

    i = 0
    n2n = []
    for c in curn:
        ii = 0
        if c in edges:
            for _c in list(edges[c]):
                ii = 0
                for nid in nxtn:
                    if nid == _c:
                        n2n.append((c, i*2, ii*2))
                        break
                    ii = ii + 1
        i = i + 1

    i = len(curn)
    s2n = []
    for c in curs:
        ii = 0
        if c in edges:
            for _c in list(edges[c]):
                ii = 0
                for nid in nxtn:
                    if nid == _c:
                        s2n.append((c, i*2, ii*2))
                        break
                    ii = ii + 1
        i = i + 1
    return (concat([n2n, n2s]), concat([s2n, s2s]))

def addDestWithBypass(edges, dg):
    """
    | Add destinations of nodes

    >>> edges = mkEdges([(0,[1,2]),(1,[2])])
    >>> dg = getDepthGroup2(dict([]), edges)
    >>> sorted(addDestWithBypass(edges,dict([(0,([2],[])),(1,([1],[0])),(2,([0],[]))])).items())
    [(0, ([(2, 0, 0)], [])), (1, ([(1, 0, 0)], [(0, 2, 0)])), (2, ([(0, 0, 0), (0, 0, 2)], []))]
    """
    r = []
    for d in range(maxDepth(dg), -1, -1):
        if d-1 in dg:
            r.append((d, addDest(edges, dg[d], dg[d-1])))
        else:
            (a0, a1) = dg[d]
            r0 = []
            i = 0
            for a in a0:
                r0.append((a, i*2, i*2))
                i = i + 1
            r1 = []
            for a in a1:
                r1.append((a, i*2, i*2))
                i = i + 1
            r.append((d, (r0, r1)))
    return dict(r)

def addNode(edges: Edges[T], nd: NodeDepth[T], dg: DepthGroup[T]) -> DepthGroupWithDest[T]:
    """
    | Grouping the nodes by the depth
    >>> edges = mkEdges([(0,[1,2])])
    >>> dg = getDepthGroup2(dict([]),edges)
    >>> nd = getNodeDepth(dg)
    >>> dg
    {0: [1, 2], 1: [0]}
    >>> sorted(addNode(edges,nd,dg).items())
    [(0, ([(1, 0, 0), (2, 2, 2)], [])), (1, ([(0, 0, 0), (0, 0, 2)], []))]
    """
    return addDestWithBypass(edges, addBypassNode(edges, nd, dg))

def toSymbol(labels: Labels[T], dg: DepthGroupWithDest[T]) -> List[List[Symbol]]:
    r = []
    for d in range(maxDepth(dg), -1, -1):
        (n, s) = dg[d]
        r.append(nodeWithSpace(labels, (n, s)))
        for i in moveAllWithSpace(concat([n, s])):
            r.append(i)
    return r

def edgesToText(labels: Labels[T], edges: Edges[T]) -> str:
    dg = getDepthGroup2(labels, edges)
    nd = getNodeDepth(dg)
    r = toSymbol(labels, addNode(edges, nd, dg))
    r.pop()
    return renderToText(r)

def symbolToChar(symbol: Symbol) -> str:
    if symbol.sym == SEnum.SNode:
        return "o"
    elif symbol.sym == SEnum.SHold:
        return "|"
    elif symbol.sym == SEnum.SLeft:
        return "/"
    elif symbol.sym == SEnum.SRight:
        return '\\'
    elif symbol.sym == SEnum.SLMove:
        return "_"
    elif symbol.sym == SEnum.SRMove:
        return "_"
    elif symbol.sym == SEnum.SCross:
        return "x"
    elif symbol.sym == SEnum.SSpace:
        return " "
    else:
        return " "

def renderToText(list_of_symbols: List[List[Symbol]]) -> str:
    """
    >>> print(renderToText( [[SNode("")],[SHold],[SNode("")]]),end='')
    o
    |
    o
    >>> print(renderToText([[SNode(""),SSpace,SNode("")],[SHold,SLeft],[SNode("")]]),end='')
    o o
    |/
    o
    >>> print(renderToText([[SNode("")],[SHold,SRight],[SNode(""),SSpace,SHold],[SHold,SLeft],[SNode("")]]),end='')
    o
    |\\
    o |
    |/
    o
    """
    r = ''
    for symbols in list_of_symbols:
        labelbuf = []
        for s in symbols:
            if s.sym == SEnum.SNode:
                labelbuf.append(s.label)
            r += symbolToChar(s)

        llen = len(labelbuf)
        m = 0
        if labelbuf:
            m = max(map(len, labelbuf))
        if m != 0:
            prefix = getLongestCommonPrefix(labelbuf)
            l = len(prefix)
            if l >= 4 and llen >= 2:
                r += "    "
                r += prefix
                r += "{"
                for i in range(0, llen):
                    r += labelbuf[i][l:]
                    if i != llen-1:
                        r += ","
                r += "}"
            else:
                r += "    "
                for i in range(0, llen):
                    r += labelbuf[i]
                    if i != llen-1:
                        r += ","
        r += "\n"
    return r


def getLongestCommonPrefix(strs: List[str]) -> str:
    """
    >>> getLongestCommonPrefix(["abc_aaa","abc_bb","abc_aaaa"])
    'abc_'
    >>> getLongestCommonPrefix(["b","abc_bb","abc_aaaa"])
    ''
    >>> getLongestCommonPrefix([])
    ''
    """
    l = len(strs)
    if l == 0:
        return ""
    else:
        m = min(map(len, strs))
        if m == 0:
            return ""

        for i in range(0, m):
            common = strs[0][i]
            for s in strs:
                if s[i] != common:
                    return strs[0][0:i]
        return strs[0]
