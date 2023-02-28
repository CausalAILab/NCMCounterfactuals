from src.ds import CausalGraph, CTF, CTFTerm, graph_search

from copy import deepcopy


class Punit:
    def __init__(self, V_set, do_set=None):
        self.V_set = V_set
        if do_set is not None:
            self.do_set = do_set
        else:
            self.do_set = set()

    def set_do(self, do_set):
        self.do_set = do_set
        for V in do_set:
            if V in self.V_set:
                self.V_set.remove(V)

        if len(self.V_set) == 0:
            return False
        return True

    def _marg_remove(self, V):
        if V in self.V_set:
            self.V_set.remove(V)

    def _marg_check_remove(self, V):
        if V in self.V_set:
            return True
        return False

    def _marg_check_contains(self, V):
        if V in self.V_set:
            return 1, 0
        return 0, 0

    def get_latex(self):
        return str(self)

    def __str__(self):
        out = "P("
        for V in self.V_set:
            out = out + V + ','
        out = out[:-1]
        if len(self.do_set) > 0:
            out = out + ' | do('
            for V in self.do_set:
                out = out + V + ','
            out = out[:-1] + ')'
        out = out + ')'
        return out


class Pexpr:
    def __init__(self, upper, lower, marg_set):
        self.upper = upper
        self.lower = lower
        self.marg_set = marg_set

    def add_marg(self, marg_V):
        for V in marg_V:
            if self._marg_check_remove(V):
                self._marg_remove(V)
            else:
                self.marg_set.add(V)

    def set_do(self, do_set):
        for term in self.upper:
            success = term.set_do(do_set)
            if not success:
                self.upper.remove(term)

        for term in self.lower:
            success = term.set_do(do_set)
            if not success:
                self.lower.remove(term)

        if len(self.upper) == 0:
            return False
        return True

    def _marg_remove(self, V):
        for Pu in self.upper:
            Pu._marg_remove(V)

    def _marg_check_remove(self, V):
        upper_count, lower_count = self._marg_check_contains(V)
        if lower_count == 0 and upper_count <= 1:
            return True
        return False

    def _marg_check_contains(self, V):
        upper_count = 0
        lower_count = 0
        for Pu in self.upper:
            up, low = Pu._marg_check_contains(V)
            upper_count += up
            lower_count += low
        for Pl in self.lower:
            up, low = Pl._marg_check_contains(V)
            lower_count += up + low
        return upper_count, lower_count

    def get_latex(self):
        if len(self.marg_set) == 0 and len(self.lower) == 0 and len(self.upper) == 1:
            return self.upper[0].get_latex()

        out = "\\left["
        if len(self.marg_set) > 0:
            out = "\\sum_{"
            for M in self.marg_set:
                out = out + str(M) + ','
            out = out[:-1] + '}\\left['

        if len(self.lower) > 0:
            out = out + "\\frac{"
            for P in self.upper:
                out = out + P.get_latex()
            out = out + "}{"
            for P in self.lower:
                out = out + P.get_latex()
            out = out + "}"
        else:
            for P in self.upper:
                out = out + P.get_latex()

        out = out + '\\right]'
        return out

    def __str__(self):
        if len(self.marg_set) == 0 and len(self.lower) == 0 and len(self.upper) == 1:
            return str(self.upper[0])

        out = "["
        if len(self.marg_set) > 0:
            out = "sum{"
            for M in self.marg_set:
                out = out + str(M) + ','
            out = out[:-1] + '}['
        for P in self.upper:
            out = out + str(P)
        if len(self.lower) > 0:
            out = out + " / "
            for P in self.lower:
                out = out + str(P)
        out = out + ']'
        return out


class PlaceholderValue:
    def __init__(self, var, do_vals, id=0):
        self.var = var
        self.do_vals = do_vals
        self.id = id

    def __str__(self):
        return "D({})".format(self.var, self.do_vals)

    def __eq__(self, obj):
        if not isinstance(obj, PlaceholderValue):
            return False
        if self.id != obj.id:
            return False
        if obj.var != self.var:
            return False
        if set(self.do_vals.keys()) != set(obj.do_vals.keys()):
            return False
        for do_var in self.do_vals:
            if self.do_vals[do_var] != obj.do_vals[do_var]:
                return False
        return True

    def __hash__(self):
        return hash((self.id, self.var, frozenset(self.do_vals.items())))


class PUnitValue:
    def __init__(self, var_vals, do_vals=None):
        self.var_vals = var_vals
        if do_vals is not None:
            self.do_vals = do_vals
        else:
            self.do_vals = {}

    def set_do(self, do_set):
        self.do_vals = dict()
        for V in do_set:
            if V in self.var_vals:
                self.do_vals[V] = self.var_vals[V]
                self.var_vals.pop(V)
            else:
                self.do_vals[V] = PlaceholderValue(V, set())

        if len(self.var_vals) == 0:
            return False
        return True

    def _marg_remove(self, ph):
        if ph.var in self.var_vals and self.var_vals[ph.var] == ph:
            del self.var_vals[ph.var]

    def _marg_check_remove(self, ph):
        if ph.var in self.var_vals and self.var_vals[ph.var] == ph:
            return True
        return False

    def _marg_check_contains(self, ph):
        if ph.var in self.var_vals and self.var_vals[ph.var] == ph:
            return 1, 0
        return 0, 0

    def get_latex(self):
        return str(self)

    def __str__(self):
        out = "P("
        for var, val in self.var_vals.items():
            out = out + "{} = {},".format(var, val)
        out = out[:-1]
        if len(self.do_vals) > 0:
            out = out + ' | do('
            for V in self.do_vals:
                out = out + "{} = {},".format(V, self.do_vals[V])
            out = out[:-1] + ')'
        out = out + ')'
        return out


def identify(X, Y, G):
    """
    Takes sets of variables X and Y as input.
    If identifiable, returns P(Y | do(X)) in the form of a Pexpr object.
    Otherwise, returns FAIL.
    """
    Q_evals = dict()
    V_eval = Pexpr(upper=[Punit(G.set_v)], lower=[], marg_set=set())
    Q_evals[tuple(G.v)] = V_eval

    raw_C = G.set_v.difference(X)
    an_Y = G.subgraph(raw_C).ancestors(Y)
    marg = an_Y.difference(Y)

    Q_list = G.cc

    Qy_list = G.subgraph(an_Y).cc
    if len(Qy_list) == 1:
        Qy = set(Qy_list[0])
        for raw_Q in Q_list:
            Q = set(raw_Q)
            if Qy.issubset(Q):
                _evaluate_Q(Q, G.set_v, Q_evals, G)
                result = _identify_help(Qy, Q, Q_evals, G)
                if result == "FAIL":
                    return "FAIL"
                result.add_marg(marg)
                return result
    else:
        upper = []
        for raw_Qy in Qy_list:
            Qy = set(raw_Qy)
            for raw_Q in Q_list:
                Q = set(raw_Q)
                if Qy.issubset(Q):
                    _evaluate_Q(Q, G.set_v, Q_evals, G)
                    result = _identify_help(Qy, Q, Q_evals, G)
                    if result == "FAIL":
                        return "FAIL"
                    upper.append(result)

        result = Pexpr(upper=upper, lower=[], marg_set=set())
        result.add_marg(marg)
        return result


def _identify_help(C, T, Q_evals, G):
    T_eval = Q_evals[G.serialize(T)]
    if C == T:
        return T_eval

    an_C = G.subgraph(T).ancestors(C)
    if an_C == T:
        return "FAIL"

    marg_out = T.difference(an_C)
    an_C_eval = deepcopy(T_eval)
    an_C_eval.add_marg(marg_out)
    Q_evals[G.serialize(an_C)] = an_C_eval

    Q_list = G.subgraph(an_C).cc
    for raw_Q in Q_list:
        Q = set(raw_Q)
        if C.issubset(Q):
            _evaluate_Q(Q, an_C, Q_evals, G)
            return _identify_help(C, Q, Q_evals, G)


def _evaluate_Q(A, B, Q_evals, G):
    """
    Given variable sets B and its subset A, with Q[B] stored in Q_evals, Q[A] is computed using Q[B] and
    stored in Q_evals.
    """
    assert A.issubset(B)
    assert B.issubset(G.set_v)

    A_key = G.serialize(A)
    if A_key in Q_evals:
        return

    A_list = G.convert_set_to_sorted(A)
    B_list = G.convert_set_to_sorted(B)
    B_eval = Q_evals[G.serialize(B)]

    upper = []
    lower = []

    start = 0
    i = 0
    j = 0
    while i < len(A_list):
        while A_list[i] != B_list[j]:
            j += 1
            start += 1

        while i < len(A_list) and A_list[i] == B_list[j]:
            i += 1
            j += 1

        up_term = deepcopy(B_eval)
        if j < len(B_list):
            up_term.add_marg(set(B_list[j:]))
        upper.append(up_term)
        if start != 0:
            low_term = deepcopy(B_eval)
            low_term.add_marg(set(B_list[start:]))
            lower.append(low_term)
        start = j

    Q_evals[A_key] = Pexpr(upper=upper, lower=lower, marg_set=set())


def split_ctf(q: CTF, include_values=True):
    """
    Splits each CTFTerm in q to individual CTFTerms for each V in q.vars.
    """
    new_q = CTF()
    for term in q.term_set:
        for v in term.vars:
            if include_values:
                new_q.add_term(CTFTerm({v}, term.do_vals, {v: term.var_vals[v]}))
            else:
                new_q.add_term(CTFTerm({v}, term.do_vals, {}))
    for term in q.cond_term_set:
        for v in term.vars:
            if include_values:
                new_q.add_cond_term(CTFTerm({v}, term.do_vals, {v: term.var_vals[v]}))
            else:
                new_q.add_cond_term(CTFTerm({v}, term.do_vals, {}))
    return new_q


def merge_ctf(q: CTF):
    """
    Combines all CTFTerms in q that share the same interventions. If the same variable in the same intervention is
    observed to take different values, ZERO is returned for term set, and INCONSISTENT is returned for the conditional
    set.
    """

    def _get_merged_sets(term_set):
        do_sets = dict()
        for term in term_set:
            set_key = frozenset(term.do_vals.items())
            if set_key not in do_sets:
                do_sets[set_key] = [set(), dict(), term.do_vals]

            for var in term.vars:
                do_sets[set_key][0].add(var)
                if var in term.var_vals:
                    if var in do_sets[set_key][1]:
                        if do_sets[set_key][1][var] != term.var_vals[var]:
                            return None
                    else:
                        do_sets[set_key][1][var] = term.var_vals[var]

        return do_sets

    do_sets = _get_merged_sets(q.term_set)
    cond_do_sets = _get_merged_sets(q.cond_term_set)

    if do_sets is None:
        return "ZERO"
    if cond_do_sets is None:
        return "INCONSISTENT"

    new_q = CTF()
    for key in do_sets:
        new_q.add_term(CTFTerm(do_sets[key][0], do_sets[key][2], do_sets[key][1]))
    for key in cond_do_sets:
        new_q.add_cond_term(CTFTerm(cond_do_sets[key][0], cond_do_sets[key][2], cond_do_sets[key][1]))

    return new_q


def ctf_id(q: CTF, G: CausalGraph, Z_list):
    """
    Identifies counterfactual query q from G and P(V). Returns an expression for q in the form of P(V) or FAIL if
    not identifiable.
    """
    def _sub_id(fact_q, placeholders):
        c_comps = G.subgraph(fact_q.get_vars()).cc

        id_facts = []
        for raw_C in c_comps:
            C = set(raw_C)
            if not ctf_consistent(fact_q, C):
                return "FAIL"

            found = False
            for Z in Z_list:
                if len(C.intersection(Z)) == 0:
                    G_Z = G.subgraph(G.set_v, V_cut_back=Z)
                    B = None
                    for raw_B_i in G_Z.cc:
                        B_i = set(raw_B_i)
                        if C.issubset(B_i):
                            B = B_i
                    p_B = identify(G_Z.set_v.difference(B), B, G_Z)

                    p_C_attempt = _identify_help(C, B, {G.serialize(B): p_B}, G)
                    if p_C_attempt != "FAIL":
                        found = True
                        p_C_val = insert_pexpr_values(p_C_attempt, fact_q, C)
                        p_C_val.set_do(Z)
                        id_facts.append(p_C_val)
                        break

            if not found:
                return "FAIL"

        result = Pexpr(id_facts, [], set())
        result.add_marg(placeholders)
        return result

    assert q.has_all_values()

    new_q = ctf_filter_interventions(q, G)
    new_q = merge_ctf(new_q)
    if new_q == "ZERO":
        return 0
    elif new_q == "INCONSISTENT":
        return -1

    if merge_ctf(new_q.get_full_joint()) == "ZERO":
        return 0

    fact_q_cond = None
    cond_placeholders = None
    if len(new_q.cond_term_set) != 0:
        fact_q, placeholders, fact_q_cond, cond_placeholders = _get_simplified_joint(new_q, G)
    else:
        fact_q, placeholders, _ = factorize_ctf(new_q, G)

    main_result = _sub_id(fact_q, placeholders)
    if main_result == "FAIL":
        return main_result

    if fact_q_cond is not None:
        cond_result = _sub_id(fact_q_cond, cond_placeholders)
        return Pexpr([main_result], [cond_result], set())

    return main_result


def ctf_filter_interventions(q: CTF, G: CausalGraph):
    """
    Removes interventions with no effect for each variable in q.
    """
    def _filter_term(term: CTFTerm):
        t_var = next(iter(term.vars))
        t_do_vars = set(term.do_vals.keys())
        an_t_var = G.subgraph(G.set_v, V_cut_back=t_do_vars).ancestors({t_var})
        new_do = an_t_var.intersection(t_do_vars)
        new_do_vals = {var: term.do_vals[var] for var in new_do}
        new_term = CTFTerm({t_var}, new_do_vals, term.var_vals)
        return new_term

    new_q = CTF()
    split_q = split_ctf(q)
    for term in split_q.term_set:
        new_q.add_term(_filter_term(term))
    for term in split_q.cond_term_set:
        new_q.add_cond_term(_filter_term(term))
    return new_q


def _get_an_value(q: CTF, G: CausalGraph, var, do_vals):
    do_vars = set(do_vals.keys())
    G_xcutback = G.subgraph(G.set_v, V_cut_back=do_vars)
    var_ans = G_xcutback.ancestors({var})
    new_do_vars = set(var_ans.intersection(do_vars))
    new_do_vals = {v: do_vals[v] for v in new_do_vars}
    val = q.search_value(var, new_do_vals)
    return val, new_do_vals


def _ctf_ancestors(q: CTF, G: CausalGraph, cached_values: dict = None, include_values=True):
    """
    Returns an ancestral CTF with placeholder values for q.
    """
    assert len(q.cond_term_set) == 0

    if cached_values is None:
        cached_values = dict()

    placeholders = []
    new_q = CTF()
    for term in q.term_set:
        do_vars = set(term.do_vals.keys())
        G_xcutfront = G.subgraph(G.set_v, V_cut_front=do_vars)

        var_ans = G_xcutfront.ancestors(term.vars)
        for an in var_ans:
            an_val, new_do_vals = _get_an_value(q, G, an, term.do_vals)
            if include_values:
                if an_val is None:
                    if an not in cached_values:
                        an_val = PlaceholderValue(an, new_do_vals)
                        placeholders.append(an_val)
                        cached_values[an] = an_val
                    else:
                        an_val = cached_values[an]
                if an not in cached_values:
                    cached_values[an] = an_val
                new_q.add_term(CTFTerm({an}, new_do_vals, {an: an_val}))
            else:
                new_q.add_term(CTFTerm({an}, new_do_vals, {}))

    return merge_ctf(new_q), placeholders, cached_values


def ancestral_components(q: CTF, G: CausalGraph):
    """
    Gets the ancestral components of q in G.
    """

    def _linked(ctf1: CTF, ctf2: CTF, G: CausalGraph):
        inter = _ctf_intersection(ctf1, ctf2)
        if len(inter.term_set) != 0:
            return True

        v1 = ctf1.get_vars()
        v2 = ctf2.get_vars()
        for edge in G.bi:
            vb1, vb2 = edge
            if (vb1 in v1 and vb2 in v2) or (vb2 in v1 and vb1 in v2):
                return True

        return False

    full_q = split_ctf(q.get_full_joint())
    cond_q = split_ctf(q.get_cond_ctf())

    _, _, cached_values = _ctf_ancestors(full_q, G)

    components = []
    for term in full_q.term_set:
        cut_vars = set()
        term_ctf = CTF({term}, set())
        term_ans, _, _ = _ctf_ancestors(term_ctf, G, cached_values=cached_values)
        term_ans = split_ctf(term_ans)
        for an in term_ans.term_set:
            if an in cond_q.term_set:
                cut_vars.add(next(iter(an.vars)))

        G_cut = G.subgraph(G.set_v, V_cut_front=cut_vars)
        comp, _, _ = _ctf_ancestors(term_ctf, G_cut, cached_values=cached_values)
        components.append(comp)

        linked_comps = []
        for i, other in enumerate(components):
            if _linked(comp, other, G):
                linked_comps.append(other)

        for comp2 in linked_comps:
            comp = _ctf_union(comp, comp2)
            components.remove(comp2)
        components.append(comp)

    return components, cached_values


def _get_simplified_joint(q: CTF, G: CausalGraph):
    an_comps, cached_values = ancestral_components(q, G)
    stripped_terms = split_ctf(q.drop_cond_ctf())

    valid_comps = set()
    for term in stripped_terms.term_set:
        t_var = next(iter(term.vars))
        for comp in an_comps:
            if t_var in comp.get_vars():
                valid_comps.add(comp)

    joint_comp = CTF(set(), set())
    for comp in valid_comps:
        joint_comp = _ctf_union(joint_comp, comp)

    fact_q, _, _2 = factorize_ctf(q.get_full_joint(), G)
    placeholders = set()
    cond_placeholders = set()
    fact_q = split_ctf(fact_q)
    valid_terms = set()
    valid_cond_terms = set()
    joint_comp_vars = joint_comp.get_vars()
    non_cond_vars = q.get_vars()
    cond_term_found = False
    for term in fact_q.term_set:
        t_var = next(iter(term.vars))
        if t_var in joint_comp_vars:
            for var in term.do_vals:
                if isinstance(term.do_vals[var], PlaceholderValue):
                    placeholders.add(term.do_vals[var])
                    cond_placeholders.add(term.do_vals[var])

            new_term_var_vals = deepcopy(term.var_vals)

            for var in term.var_vals:
                if isinstance(term.var_vals[var], PlaceholderValue):
                    placeholders.add(term.var_vals[var])
                    cond_placeholders.add(term.var_vals[var])
                if var in non_cond_vars:
                    new_val = PlaceholderValue(var, term.do_vals)
                    new_term_var_vals[var] = new_val
                    cond_placeholders.add(new_val)
                else:
                    cond_term_found = True

            cond_term = CTFTerm(term.vars, term.do_vals, new_term_var_vals)
            valid_terms.add(term)
            valid_cond_terms.add(cond_term)

    valid_CTF = merge_ctf(CTF(valid_terms, set()))
    valid_cond_CTF = merge_ctf(CTF(valid_cond_terms, set()))

    if cond_term_found:
        return merge_ctf(valid_CTF), list(placeholders), merge_ctf(valid_cond_CTF), list(cond_placeholders)
    else:
        return merge_ctf(valid_CTF), list(placeholders), None, None


def factorize_ctf(q: CTF, G: CausalGraph):
    """
    Converts a CTF q into a factorizeable form.
    """
    an_q, placeholders, cached_values = _ctf_ancestors(q, G)
    new_q = CTF()
    for var in G.v:
        var_pa = G.pa[var]
        for term in an_q.term_set:
            if var in term.vars:
                new_do_vals = dict()
                for pa in var_pa:
                    if pa in term.do_vals:
                        new_do_vals[pa] = term.do_vals[pa]
                    else:
                        pa_val, _ = _get_an_value(an_q, G, pa, term.do_vals)
                        assert pa_val is not None
                        new_do_vals[pa] = pa_val
                new_q.add_term(CTFTerm({var}, new_do_vals, {var: term.var_vals[var]}))

    return merge_ctf(new_q), placeholders, cached_values


def ctf_consistent(q: CTF, C):
    """
    Checks if the variables in C are consistent in q.
    """
    var_vals = dict()
    for term in q.term_set:
        for var in term.var_vals:
            if var in C:
                if var in var_vals:
                    if var_vals[var] != term.var_vals[var]:
                        return False
                else:
                    var_vals[var] = term.var_vals[var]

    var_do_vals = dict()
    for term in q.term_set:
        for var in term.var_vals:
            if var in C:
                for do_var in term.do_vals:
                    if do_var in var_vals and term.do_vals[do_var] != var_vals[do_var]:
                        return False
                    if do_var in var_do_vals:
                        if var_do_vals[do_var] != term.do_vals[do_var]:
                            return False
                    else:
                        var_do_vals[do_var] = term.do_vals[do_var]
    return True


def insert_pexpr_values(ex: Pexpr, q: CTF, C):
    """
    Convert a valueless Pexpr to one with values.
    """
    last_value = dict()
    id_counter = dict()
    for term in q.term_set:
        exists = False
        for var in term.vars:
            if isinstance(term.var_vals[var], PlaceholderValue):
                id_counter[var] = id_counter.get(var, 0) + 1
            if var in C:
                last_value[var] = term.var_vals[var]
                exists = True
        if exists:
            for do_var in term.do_vals:
                last_value[do_var] = term.do_vals[do_var]

    return _insert_pexpr_values(ex, last_value, id_counter)


def _insert_pexpr_values(ex: Pexpr, last_value, id_counter):
    def _unit_to_value(un: Punit, last_value, id_counter):
        var_vals = dict()
        for var in un.V_set:
            if var in last_value:
                var_vals[var] = last_value[var]
            else:
                var_vals[var] = PlaceholderValue(var, dict(), id=-1)
        return PUnitValue(var_vals)

    new_upper = []
    new_lower = []
    new_last_value = {k: v for (k, v) in last_value.items()}
    new_marg = set()
    for marg_var in ex.marg_set:
        marg_id = id_counter.get(marg_var, 0)
        id_counter[marg_var] = marg_id + 1
        marg_val = PlaceholderValue(marg_var, dict(), id=marg_id)
        new_last_value[marg_var] = marg_val
        new_marg.add(marg_val)

    for Pu in ex.upper:
        if isinstance(Pu, Pexpr):
            new_upper.append(_insert_pexpr_values(Pu, new_last_value, id_counter))
        else:
            new_upper.append(_unit_to_value(Pu, new_last_value, id_counter))

    for Pl in ex.lower:
        if isinstance(Pl, Pexpr):
            new_lower.append(_insert_pexpr_values(Pl, new_last_value, id_counter))
        else:
            new_lower.append(_unit_to_value(Pl, new_last_value, id_counter))

    return Pexpr(new_upper, new_lower, new_marg)


def _ctf_intersection(ctf1: CTF, ctf2: CTF, include_values=True):
    assert len(ctf1.cond_term_set) == 0
    assert len(ctf2.cond_term_set) == 0

    t1 = split_ctf(ctf1, include_values=include_values).term_set
    t2 = split_ctf(ctf2, include_values=include_values).term_set
    new_term_set = t1.intersection(t2)
    return CTF(new_term_set, set())


def _ctf_union(ctf1: CTF, ctf2: CTF, include_values=True):
    assert len(ctf1.cond_term_set) == 0
    assert len(ctf2.cond_term_set) == 0

    t1 = split_ctf(ctf1, include_values=include_values).term_set
    t2 = split_ctf(ctf2, include_values=include_values).term_set
    new_term_set = t1.union(t2)
    return CTF(new_term_set, set())


def get_mediators(G, treatment, outcome):
    possible_mediators = set()
    for V in treatment:
        possible_mediators = possible_mediators.union(graph_search(G, V))

    possible_mediators = possible_mediators.difference(treatment)

    mediators = set()
    for V in possible_mediators:
        if graph_search(G, V, outcome):
            mediators.add(V)

    return mediators


def infer_query(G, query_type):
    query_type = query_type.upper()

    treatment = set()
    for V in G.set_v:
        if 'X' in V:
            treatment.add(V)

    mediators = get_mediators(G, treatment, 'Y')

    treat_dict = {V: 1 for V in treatment}
    notreat_dict = {V: 0 for V in treatment}
    placetreat_dict = {V: PlaceholderValue(V, {}) for V in treatment}
    med_dict = {V: PlaceholderValue(V, treat_dict) for V in mediators}

    if query_type == "ATE":
        t_y1x1 = CTFTerm({'Y'}, treat_dict, {'Y': 1})
        return CTF({t_y1x1}, set())
    elif query_type == "ETT":
        t_x1 = CTFTerm(treatment, {}, treat_dict)
        t_y1x0 = CTFTerm({'Y'}, notreat_dict, {'Y': 1})
        return CTF({t_y1x0}, {t_x1})
    elif query_type == "NDE":
        t_y1x0m = CTFTerm({'Y'}, {**notreat_dict, **med_dict}, {'Y': 1})
        t_mx1 = CTFTerm(mediators, treat_dict, med_dict)
        return CTF({t_y1x0m, t_mx1}, set())
    elif query_type == "CTFDE":
        t_y1x0m = CTFTerm({'Y'}, {**notreat_dict, **med_dict}, {'Y': 1})
        t_mx1 = CTFTerm(mediators, treat_dict, med_dict)
        t_x = CTFTerm(treatment, {}, placetreat_dict)
        return CTF({t_y1x0m, t_mx1}, {t_x})
    elif query_type == "ALT_CTFDE":
        t_y1x0m = CTFTerm({'Y'}, {**notreat_dict, **med_dict}, {'Y': 1})
        t_mx1 = CTFTerm(mediators, treat_dict, med_dict)
        t_x1 = CTFTerm(treatment, {}, treat_dict)
        return CTF({t_y1x0m, t_mx1}, {t_x1})
    elif query_type == "PNS":
        t_y1x1 = CTFTerm({'Y'}, treat_dict, {'Y': 1})
        t_y0x0 = CTFTerm({'Y'}, notreat_dict, {'Y': 0})
        return CTF({t_y1x1, t_y0x0}, set())
    else:
        return None


if __name__ == "__main__":
    graphs = {
        "id": ["conf", "simple", "backdoor", "frontdoor", "napkin", "bow", "bdm", "med", "expl"],
        "zid": ["zid_a", "zid_b", "zid_c"],
        "gid": ["gid_a", "gid_b", "gid_c", "gid_d"],
        "expl_set": ["expl", "expl_xm", "expl_xy", "expl_my"],
        "expl_do_set": ["expl_xm_dox", "expl_xy_dox", "expl_my_dox"],
    }

    Z_lists = {
        "id": [set()],
        "zid": [set(), {'Z'}],
        "gid": [{'X1'}, {'X2'}],
        "expl_set": [set()],
        "expl_do_set": [set(), {'X'}],
    }

    queries = ["ate", "ett", "nde", "ctfde", "alt_ctfde", "pns"]

    for mode in graphs:
        G_list = graphs[mode]
        Z_list = Z_lists[mode]
        for q in queries:
            print(q.upper())
            for G_name in G_list:
                G = CausalGraph.read("../../dat/cg/{}.cg".format(G_name))
                ctf_q = infer_query(G, q)
                print("{}: {}".format(G_name, ctf_id(ctf_q, G, Z_list)))
