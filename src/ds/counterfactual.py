from collections.abc import Iterable


class CTFTerm:
    def __init__(self, vars: Iterable, do_vals: dict, var_vals: dict = None):
        self.vars = set(vars)
        self.do_vals = do_vals
        if var_vals is None:
            self.var_vals = dict()
        else:
            assert set(var_vals.keys()).issubset(self.vars)
            self.var_vals = var_vals

        self.val_match = (self.vars == set(self.var_vals.keys()))

    def has_all_values(self):
        return self.val_match

    def is_degenerate(self):
        return len(self.vars) == 0

    def strip_values(self):
        return CTFTerm(self.vars, self.do_vals, {})

    def __str__(self):
        out = "["
        for i, var in enumerate(self.vars):
            if var in self.var_vals:
                out += "{} = {}".format(var, self.var_vals[var])
            else:
                out += var
            if i != len(self.vars) - 1:
                out += ", "
        if len(self.do_vals) > 0:
            out += " | do("
            for i, var in enumerate(self.do_vals):
                out += "{} = {}".format(var, self.do_vals[var])
                if i != len(self.do_vals) - 1:
                    out += ", "
            out += ")"
        out += "]"
        return out

    def __eq__(self, other):
        if not isinstance(other, CTFTerm):
            return False
        return self.vars == other.vars and self.do_vals == other.do_vals and self.var_vals == other.var_vals

    def __hash__(self):
        return hash((frozenset(self.vars), frozenset(self.do_vals.items()), frozenset(self.var_vals.items())))


class CTF:
    def __init__(self, term_set: set = None, cond_term_set: set = None, name=None):
        if term_set is None:
            self.term_set = set()
        else:
            degenerate_terms = set()
            for term in term_set:
                assert isinstance(term, CTFTerm)
                if term.is_degenerate():
                    degenerate_terms.add(term)
            self.term_set = term_set.difference(degenerate_terms)

        if cond_term_set is None:
            self.cond_term_set = set()
        else:
            degenerate_terms = set()
            for term in cond_term_set:
                assert isinstance(term, CTFTerm)
                assert term.has_all_values()
                if term.is_degenerate():
                    degenerate_terms.add(term)
            self.cond_term_set = cond_term_set.difference(degenerate_terms)

        self.name = name

    def add_term(self, term: CTFTerm):
        if not term.is_degenerate():
            self.term_set.add(term)

    def add_cond_term(self, term: CTFTerm):
        assert term.has_all_values()
        if not term.is_degenerate():
            self.cond_term_set.add(term)

    def get_vars(self):
        var_set = set()
        for term in self.term_set:
            var_set = var_set.union(term.vars)
        return var_set

    def get_cond_vars(self):
        var_set = set()
        for term in self.cond_term_set:
            var_set = var_set.union(term.vars)
        return var_set

    def has_all_values(self):
        for term in self.term_set:
            if not term.has_all_values():
                return False
        return True

    def search_value(self, q_var, q_do_vals):
        do_vars = set(q_do_vals.keys())
        for term in self.term_set:
            if q_var in term.var_vals and do_vars == set(term.do_vals.keys()):
                return term.var_vals[q_var]
        return None

    def is_degenerate(self):
        return len(self.term_set) == 0

    def get_full_joint(self):
        return CTF(self.term_set.union(self.cond_term_set))

    def drop_cond_ctf(self):
        return CTF(self.term_set, set())

    def get_cond_ctf(self):
        return CTF(self.cond_term_set, set())

    def strip_values(self):
        new_term_set = set()
        for term in self.term_set:
            new_term_set.add(term.strip_values())

        new_cond_term_set = set()
        for term in self.cond_term_set:
            new_cond_term_set.add(term.strip_values())

        return CTF(new_term_set, new_cond_term_set)

    def __str__(self):
        out = "P("
        for i, term in enumerate(self.term_set):
            out += str(term)
            if i != len(self.term_set) - 1:
                out += ", "
        if len(self.cond_term_set) > 0:
            out += " | "
            for i, term in enumerate(self.cond_term_set):
                out += str(term)
                if i != len(self.cond_term_set) - 1:
                    out += ", "
        out += ")"
        return out
