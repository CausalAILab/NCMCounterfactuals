import itertools
import re
from collections import deque


class CausalGraph:
    def __init__(self, V, directed_edges=[], bidirected_edges=[]):
        self.de = directed_edges
        self.be = bidirected_edges

        self.v = list(V)
        self.set_v = set(V)
        self.pa = {v: set() for v in V}  # parents (directed edges)
        self.ch = {v: set() for v in V}  # children (directed edges)
        self.ne = {v: set() for v in V}  # neighbors (bidirected edges)
        self.bi = set(map(tuple, map(sorted, bidirected_edges)))  # bidirected edges

        for v1, v2 in directed_edges:
            self.pa[v2].add(v1)
            self.ch[v1].add(v2)

        for v1, v2 in bidirected_edges:
            self.ne[v1].add(v2)
            self.ne[v2].add(v1)
            self.bi.add(tuple(sorted((v1, v2))))

        self.pa = {v: sorted(self.pa[v]) for v in self.v}
        self.ch = {v: sorted(self.ch[v]) for v in self.v}
        self.ne = {v: sorted(self.ne[v]) for v in self.v}

        self._sort()
        self.v2i = {v: i for i, v in enumerate(self.v)}

        self.cc = self._c_components()
        self.v2cc = {v: next(c for c in self.cc if v in c) for v in self.v}
        self.pap = {
            v: sorted(set(itertools.chain.from_iterable(
                self.pa[v2] + [v2]
                for v2 in self.v2cc[v]
                if self.v2i[v2] <= self.v2i[v])) - {v},
                      key=self.v2i.get)
            for v in self.v}
        self.c2 = self._maximal_cliques()
        self.v2c2 = {v: [c for c in self.c2 if v in c] for v in self.v}

    def __iter__(self):
        return iter(self.v)

    def subgraph(self, V_sub, V_cut_back=None, V_cut_front=None):
        assert V_sub.issubset(self.set_v)

        if V_cut_back is None:
            V_cut_back = set()
        if V_cut_front is None:
            V_cut_front = set()

        assert V_cut_back.issubset(self.set_v)
        assert V_cut_front.issubset(self.set_v)

        new_de = [(V1, V2) for V1, V2 in self.de
                  if V1 in V_sub and V2 in V_sub and V2 not in V_cut_back and V1 not in V_cut_front]
        new_be = [(V1, V2) for V1, V2 in self.be
                  if V1 in V_sub and V2 in V_sub and V1 not in V_cut_back and V2 not in V_cut_back]

        return CausalGraph(V_sub, new_de, new_be)

    def _sort(self):  # sort V topologically
        L = []
        marks = {v: 0 for v in self.v}

        def visit(v):
            if marks[v] == 2:
                return
            if marks[v] == 1:
                raise ValueError('Not a DAG.')

            marks[v] = 1
            for c in self.ch[v]:
                visit(c)
            marks[v] = 2
            L.append(v)

        for v in marks:
            if marks[v] == 0:
                visit(v)
        self.v = L[::-1]

    def _c_components(self):
        pool = set(self.v)
        cc = []
        while pool:
            cc.append({pool.pop()})
            while True:
                added = {k2 for k in cc[-1] for k2 in self.ne[k]}
                delta = added - cc[-1]
                cc[-1].update(delta)
                pool.difference_update(delta)
                if not delta:
                    break
        return [tuple(sorted(c, key=self.v2i.get)) for c in cc]

    def _maximal_cliques(self):
        # find degeneracy ordering
        o = []
        p = set(self.v)
        while len(o) < len(self.v):
            v = min((len(set(self.ne[v]).difference(o)), v) for v in p)[1]
            o.append(v)
            p.remove(v)

        # brute-force bron_kerbosch algorithm
        c2 = set()

        def bron_kerbosch(r, p, x):
            if not p and not x:
                c2.add(tuple(sorted(r)))
            p = set(p)
            x = set(x)
            for v in list(p):
                bron_kerbosch(r.union({v}),
                              p.intersection(self.ne[v]),
                              x.intersection(self.ne[v]))
                p.remove(v)
                x.add(v)

        # apply brute-force bron_kerbosch with degeneracy ordering
        p = set(self.v)
        x = set()
        for v in o:
            bron_kerbosch({v},
                          p.intersection(self.ne[v]),
                          x.intersection(self.ne[v]))
            p.remove(v)
            x.add(v)

        return c2

    def ancestors(self, C):
        """
        Returns the ancestors of set C.
        """
        assert C.issubset(self.set_v)

        frontier = [c for c in C]
        an = {c for c in C}
        while len(frontier) > 0:
            cur_v = frontier.pop(0)
            for par_v in self.pa[cur_v]:
                if par_v not in an:
                    an.add(par_v)
                    frontier.append(par_v)

        return an

    def convert_set_to_sorted(self, C):
        return [v for v in self.v if v in C]

    def serialize(self, C):
        return tuple(self.convert_set_to_sorted(C))

    @classmethod
    def read(cls, filename):
        with open(filename) as file:
            mode = None
            V = []
            directed_edges = []
            bidirected_edges = []
            try:
                for i, line in enumerate(map(str.strip, file), 1):
                    if line == '':
                        continue

                    m = re.match('<([A-Z]+)>', line)
                    if m:
                        mode = m.groups()[0]
                        continue

                    if mode == 'NODES':
                        if line.isidentifier():
                            V.append(line)
                        else:
                            raise ValueError('invalid identifier')
                    elif mode == 'EDGES':
                        if '<->' in line:
                            v1, v2 = map(str.strip, line.split('<->'))
                            bidirected_edges.append((v1, v2))
                        elif '->' in line:
                            v1, v2 = map(str.strip, line.split('->'))
                            directed_edges.append((v1, v2))
                        else:
                            raise ValueError('invalid edge type')
                    else:
                        raise ValueError('unknown mode')
            except Exception as e:
                raise ValueError(f'Error parsing line {i}: {e}: {line}')
            return cls(V, directed_edges, bidirected_edges)

    def save(self, filename):
        with open(filename, 'w') as file:
            lines = ["<NODES>\n"]
            for V in self.v:
                lines.append("{}\n".format(V))
            lines.append("\n")
            lines.append("<EDGES>\n")
            for V1, V2 in self.de:
                lines.append("{} -> {}\n".format(V1, V2))
            for V1, V2 in self.be:
                lines.append("{} <-> {}\n".format(V1, V2))
            file.writelines(lines)


def graph_search(cg, v1, v2=None, edge_type="direct"):
    """
    Uses BFS to check for a path between v1 and v2 in cg. If v2 is None, returns all reachable nodes.
    """
    assert edge_type in ["direct", "bidirect"]
    assert v1 in cg.set_v
    assert v2 in cg.set_v or v2 is None

    q = deque([v1])
    seen = {v1}
    while len(q) > 0:
        cur = q.popleft()
        if edge_type == "direct":
            cur_ne = cg.ch[cur]
        else:
            cur_ne = cg.ne[cur]

        for ne in cur_ne:
            if ne not in seen:
                if v2 is not None and ne == v2:
                    return True
                seen.add(ne)
                q.append(ne)

    if v2 is None:
        return seen

    return False
