"""The QAOA as an explicit Qulacs circuit.

Every experiment in this package runs through here. The circuit is built once
per (instance, depth, variant) and its angles are rebound on each objective
call, so the optimiser's inner loop costs one state evolution rather than a
fresh circuit construction.

A k-body cost term exp(-i gamma c Z_q1..Z_qk) is applied as a single native
multi-qubit Pauli rotation, which is exact and fast. Its hardware cost is a
different question, and is reported separately by cost_report(), which compiles
the same layer the textbook way -- a CNOT ladder down the qubits of each term,
one R_z at the end, then the ladder undone -- and counts the result. That is
the decomposition under which the gate counts are quoted. The two
circuits implement the same unitary; only the second is a statement about
hardware. (Evolving through the ladder directly is also correct but about a
hundred times slower in Qulacs, which handles that gate order poorly.)

Qulacs defines its rotations with the opposite sign to the usual convention,
RZ(t) = exp(+i t Z / 2) and RX(t) = exp(+i t X / 2), so every angle below is
negated to give exp(-i gamma H_C) and exp(-i beta X).
"""
from __future__ import annotations

import numpy as np

from qaoa import angle_counts
from qaoa import _cost_angle_index as cost_angle_index


def _require():
    try:
        import qulacs                                  # noqa: F401
    except ImportError:
        raise SystemExit(
            "This package runs its circuits on Qulacs:\n"
            "    python -m pip install qulacs")


class QaoaCircuit:
    """A p-layer QAOA circuit for one problem and one angle-grouping variant.

    Usage:
        qc = QaoaCircuit(z_terms, n, p, 'ka')
        psi = qc.state(angles)                  # final statevector
        layers = qc.state(angles, per_layer=True)
    """

    def __init__(self, z_terms, n: int, p: int, variant: str):
        _require()
        from qulacs import ParametricQuantumCircuit

        self.n, self.p, self.variant = n, p, variant
        self.n_cost, self.n_mix = angle_counts(z_terms, n, variant)
        which = cost_angle_index(z_terms, variant)

        # |+>^n
        self.prep = ParametricQuantumCircuit(n)
        for q in range(n):
            self.prep.add_H_gate(q)

        # one circuit per layer, so intermediate states can be read off
        self.z_terms = z_terms
        self.layers = []
        self.bindings = []            # per layer: (param index, kind, idx, factor)
        for _ in range(p):
            circ = ParametricQuantumCircuit(n)
            bind = []
            for t, (coeff, support) in enumerate(z_terms):
                s = list(support)
                bind.append((circ.get_parameter_count(), 'cost', which[t],
                             -2.0 * coeff))
                circ.add_parametric_multi_Pauli_rotation_gate(s, [3] * len(s), 0.0)
            for q in range(n):
                idx = q if self.n_mix > 1 else 0
                bind.append((circ.get_parameter_count(), 'mix', idx, -2.0))
                circ.add_parametric_RX_gate(q, 0.0)
            self.layers.append(circ)
            self.bindings.append(bind)

    # -- angles -------------------------------------------------------------

    def _bind(self, angles: np.ndarray) -> None:
        betas = angles[:self.p * self.n_mix].reshape(self.p, self.n_mix)
        gammas = angles[self.p * self.n_mix:].reshape(self.p, self.n_cost)
        for layer, bind in enumerate(self.bindings):
            circ = self.layers[layer]
            for pi, kind, idx, factor in bind:
                value = (gammas[layer, idx] if kind == 'cost' else betas[layer, idx])
                circ.set_parameter(pi, factor * value)

    # -- running ------------------------------------------------------------

    def state(self, angles: np.ndarray, per_layer: bool = False):
        from qulacs import QuantumState

        self._bind(angles)
        st = QuantumState(self.n)
        st.set_zero_state()
        self.prep.update_quantum_state(st)

        out = []
        for circ in self.layers:
            circ.update_quantum_state(st)
            if per_layer:
                out.append(st.get_vector().copy())
        return out if per_layer else st.get_vector()

    # -- what the circuit costs ---------------------------------------------

    def compiled(self):
        """The same circuit under the textbook decomposition, for counting.

        Each k-body term becomes a CNOT ladder, one R_z, and the ladder undone,
        which is 2(k-1) entangling gates per term. This circuit is never run;
        it exists so that gate counts and depth describe what hardware would
        have to execute.
        """
        from qulacs import QuantumCircuit

        c = QuantumCircuit(self.n)
        for q in range(self.n):
            c.add_H_gate(q)
        for _ in range(self.p):
            for coeff, support in self.z_terms:
                s = list(support)
                pairs = list(zip(s[:-1], s[1:]))
                for a, b in pairs:
                    c.add_CNOT_gate(a, b)
                c.add_RZ_gate(s[-1], 0.0)
                for a, b in reversed(pairs):
                    c.add_CNOT_gate(a, b)
            for q in range(self.n):
                c.add_RX_gate(q, 0.0)
        return c

    def cost_report(self) -> dict:
        """Gate count, CNOT count and depth under the textbook decomposition.

        Qulacs packs each gate into the earliest layer whose qubits are free,
        but only in the order the gates were added; it does not reorder
        commuting gates. These are therefore upper bounds set by the emission
        order, not minima.
        """
        from qulacs import QuantumCircuit

        flat = self.compiled()
        cx_only = QuantumCircuit(self.n)
        gates = flat.get_gate_count()
        cx = 0
        for i in range(gates):
            g = flat.get_gate(i)
            if g.get_name() == 'CNOT':
                cx += 1
                cx_only.add_CNOT_gate(g.get_control_index_list()[0],
                                      g.get_target_index_list()[0])
        return {'gates': gates, 'cx': cx,
                'depth': flat.calculate_depth(),
                'cx_depth': cx_only.calculate_depth()}


def optimise(z_terms, costs, n, p, variant, x0, optimiser, maxiter=500,
             magic_layers=False):
    """Run one QAOA optimisation on the circuit. Returns (summary, magic).

    `costs` is the diagonal of the cost Hamiltonian, so the expectation is a
    weighted sum of the output probabilities. SciPy minimises, and we maximise
    the cost, hence the sign.
    """
    import time
    from scipy.optimize import minimize
    from qaoa import m2_renyi

    qc = QaoaCircuit(z_terms, n, p, variant)

    def objective(angles):
        psi = qc.state(angles)
        return -float(np.sum(np.abs(psi) ** 2 * costs))

    t0 = time.monotonic()
    res = minimize(objective, x0, method=optimiser, options={'maxiter': maxiter})
    seconds = time.monotonic() - t0

    psi = qc.state(res.x)
    magic = ([m2_renyi(s, n) for s in qc.state(res.x, per_layer=True)]
             if magic_layers else [])

    return {
        'angles': len(x0), 'terms': len(z_terms),
        'expectation': float(np.sum(np.abs(psi) ** 2 * costs)),
        'nfev': int(res.nfev), 'nit': int(res.nit),
        'success': bool(res.success), 'seconds': round(seconds, 3),
        'psi': psi, **qc.cost_report(),
    }, magic


def verify(nmax: int = 7, pmax: int = 3, seed: int = 1, tol: float = 1e-10) -> bool:
    """Check the circuit against the direct statevector route in qaoa.py."""
    from qaoa import (VARIANTS, cyclic_terms, initial_angles,
                      ising_phase_vectors, ising_terms, qaoa_state)

    worst, checks = 0.0, 0
    for n in range(4, nmax + 1):
        z = ising_terms(cyclic_terms(n), n)
        phases = ising_phase_vectors(z, n)
        for p in range(1, pmax + 1):
            for variant in VARIANTS:
                rng = np.random.default_rng(seed * 100 + n * 10 + p)
                x0 = initial_angles(z, n, variant, p, 'rand_2pi_p', rng)
                a = qaoa_state(z, n, p, x0, variant, phases)
                b = QaoaCircuit(z, n, p, variant).state(x0)
                worst = max(worst, float(np.max(np.abs(a - b))))
                checks += 1
                if worst > tol:
                    print(f"  MISMATCH at n={n} p={p} {variant}: {worst:.3e}")
                    return False
    print(f"  circuit agrees with the statevector route on {checks} cases, "
          f"largest amplitude difference {worst:.3e}")
    return True


if __name__ == '__main__':
    _require()
    raise SystemExit(0 if verify() else 1)
