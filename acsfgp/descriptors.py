"""Behler-Parrinello ACSF descriptors (G2 radial, G4 angular) with analytic gradients."""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

__all__ = ["ACSF", "DEFAULT_G2", "DEFAULT_G4"]


DEFAULT_G2 = [
    (0.5, 1.0), (0.5, 1.5), (0.5, 2.0), (0.5, 2.5),
    (0.5, 3.0), (0.5, 3.5), (0.5, 4.0),
    (2.0, 1.0), (2.0, 1.5), (2.0, 2.0), (2.0, 2.5), (2.0, 3.0),
    (4.0, 1.5), (4.0, 2.5),
]


DEFAULT_G4 = [
    (0.005, 1, 1), (0.005, 1, -1), (0.005, 4, 1), (0.005, 4, -1),
    (0.02, 1, 1), (0.02, 1, -1), (0.02, 4, 1), (0.02, 4, -1),
    (0.05, 8, 1), (0.05, 8, -1),
]

_MAX_BLOCK_ELEMENTS = 2_000_000


class ACSF:
    """ACSF descriptor calculator.

    species: chemical symbols in the system. r_cut: cutoff in Angstrom.
    g2 / g4: parameter sets, defaulting to 14 radial and 10 angular functions.
    """

    def __init__(self, species, r_cut: float = 5.0, g2=None, g4=None):
        self.species = sorted(set(species))
        self.r_cut = float(r_cut)
        self.g2 = list(DEFAULT_G2 if g2 is None else g2)
        self.g4 = list(DEFAULT_G4 if g4 is None else g4)
        self._zmap = {s: i for i, s in enumerate(self.species)}
        self.pairs = [(a, b) for a in range(self.n_species)
                      for b in range(a, self.n_species)]
        self._pidx = {p: i for i, p in enumerate(self.pairs)}
        self._eta2 = np.array([g[0] for g in self.g2])
        self._rs2 = np.array([g[1] for g in self.g2])


    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def n_features_per_atom(self) -> int:
        return self.n_species * len(self.g2) + self.n_pairs * len(self.g4)

    @property
    def n_features(self) -> int:
        """Length of the frame descriptor."""
        return self.n_species * self.n_features_per_atom

    def __repr__(self) -> str:
        return (f"ACSF(species={self.species}, r_cut={self.r_cut}, "
                f"n_g2={len(self.g2)}, n_g4={len(self.g4)}, "
                f"n_features={self.n_features})")


    def species_index(self, atoms: Atoms) -> np.ndarray:
        """Index of each atom within self.species."""
        try:
            return np.array([self._zmap[s] for s in atoms.get_chemical_symbols()])
        except KeyError as exc:
            raise ValueError(
                f"structure contains species {exc.args[0]!r} which the descriptor "
                f"was not built for (known: {self.species})") from exc

    def _cut(self, r):
        return 0.5 * (np.cos(np.pi * r / self.r_cut) + 1.0)

    def _dcut(self, r):
        return -0.5 * (np.pi / self.r_cut) * np.sin(np.pi * r / self.r_cut)

    def _neighbours(self, atoms):
        i, j, d, D = neighbor_list("ijdD", atoms, self.r_cut)
        order = np.argsort(i, kind="stable")
        return i[order], j[order], d[order], D[order]

    def _pack(self, atoms, i, j, d, D, zid):
        """Dense (n_atoms, max_neighbours) neighbour tables."""
        n = len(atoms)
        counts = np.bincount(i, minlength=n)
        K = max(int(counts.max()) if n else 1, 1)
        slot = np.concatenate([np.arange(c) for c in counts]) if len(i) else np.empty(0, int)
        dist = np.full((n, K), 1e9)
        vec = np.zeros((n, K, 3))
        nb = np.full((n, K), -1, dtype=np.int64)
        if len(i):
            dist[i, slot] = d
            vec[i, slot] = D
            nb[i, slot] = j
        valid = nb >= 0
        spn = np.where(valid, zid[np.clip(nb, 0, n - 1)], -1)
        fc = np.where(dist < self.r_cut, self._cut(np.clip(dist, 0, self.r_cut)), 0.0) * valid
        return nb, dist, vec, valid, spn, fc, K

    def _pair_ids(self, spn, ok):
        lo = np.minimum(spn[:, :, None], spn[:, None, :])
        hi = np.maximum(spn[:, :, None], spn[:, None, :])
        pid = np.full(lo.shape, -1, dtype=np.int64)
        for pr, k in self._pidx.items():
            pid[(lo == pr[0]) & (hi == pr[1])] = k
        return np.where(ok, pid, -1)


    def per_atom(self, atoms: Atoms) -> np.ndarray:
        """Per-atom descriptor, shape (n_atoms, n_features_per_atom)."""
        zid = self.species_index(atoms)
        n, NS, NP = len(atoms), self.n_species, self.n_pairs
        i, j, d, D = self._neighbours(atoms)
        nb, dist, vec, valid, spn, fc, K = self._pack(atoms, i, j, d, D, zid)

        g2 = np.zeros((n, NS, len(self.g2)))
        for gi, (eta, rs) in enumerate(self.g2):
            v = np.exp(-eta * (dist - rs) ** 2) * fc
            for s in range(NS):
                g2[:, s, gi] = (v * (spn == s)).sum(1)

        g4 = np.zeros((n, NP, len(self.g4)))
        triu = np.triu(np.ones((K, K), bool), 1)[None]
        block = max(1, int(_MAX_BLOCK_ELEMENTS // max(K * K, 1)))
        for a0 in range(0, n, block):
            a1 = min(a0 + block, n)
            B = a1 - a0
            dm, vm = dist[a0:a1], vec[a0:a1]
            vb, sb, fb = valid[a0:a1], spn[a0:a1], fc[a0:a1]
            dot = np.einsum("akx,alx->akl", vm, vm)
            den = dm[:, :, None] * dm[:, None, :]
            cos = np.where(den > 1e8, 0.0, dot / np.where(den > 1e8, 1.0, den))
            rjk2 = np.clip(dm[:, :, None] ** 2 + dm[:, None, :] ** 2 - 2 * dot, 1e-12, None)
            rjk = np.sqrt(rjk2)
            fjk = np.where(rjk < self.r_cut, self._cut(np.clip(rjk, 0, self.r_cut)), 0.0)
            Fp = fb[:, :, None] * fb[:, None, :] * fjk
            R2 = dm[:, :, None] ** 2 + dm[:, None, :] ** 2 + rjk2
            ok = (vb[:, :, None] & vb[:, None, :]) & triu & (rjk < self.r_cut)
            pid = self._pair_ids(sb, ok)
            flat_a = np.repeat(np.arange(B), K * K)
            flat_p = pid.reshape(-1)
            sel = flat_p >= 0
            lin = flat_a[sel] * NP + flat_p[sel]
            for gi, (eta, zeta, lam) in enumerate(self.g4):
                base = ((2.0 ** (1 - zeta))
                        * np.clip(1 + lam * cos, 1e-12, None) ** zeta
                        * np.exp(-eta * R2) * Fp)
                acc = np.bincount(lin, weights=base.reshape(-1)[sel], minlength=B * NP)
                g4[a0:a1, :, gi] = acc.reshape(B, NP)
            del dot, den, cos, rjk2, rjk, fjk, Fp, R2, ok, pid
        return np.concatenate([g2.reshape(n, -1), g4.reshape(n, -1)], axis=1)

    def frame(self, atoms: Atoms) -> np.ndarray:
        """Frame descriptor: per-atom vectors summed within each center species."""
        zid = self.species_index(atoms)
        pa = self.per_atom(atoms)
        return np.concatenate([pa[zid == s].sum(0) for s in range(self.n_species)])

    def frames(self, images) -> np.ndarray:
        """Frame descriptors for many structures, shape (n_images, n_features)."""
        return np.array([self.frame(a) for a in images], dtype=float)


    def forces_from_weights(self, atoms: Atoms, weights: np.ndarray) -> np.ndarray:
        """Forces implied by dE/ddescriptor = weights, shape (n_atoms, 3).

        weights are in raw (unscaled) descriptor units.
        """
        W = np.asarray(weights, dtype=float).reshape(self.n_species, -1)
        zid = self.species_index(atoms)
        n, NS, NP = len(atoms), self.n_species, self.n_pairs
        n_g2, n_g4 = len(self.g2), len(self.g4)
        i, j, d, D = self._neighbours(atoms)
        dE = np.zeros((n, 3))
        if len(i) == 0:
            return dE


        u = D / d[:, None]
        e = np.exp(-self._eta2[None, :] * (d[:, None] - self._rs2[None, :]) ** 2)
        fc, dfc = self._cut(d), self._dcut(d)
        dg2 = (e * (-2 * self._eta2[None, :] * (d[:, None] - self._rs2[None, :])) * fc[:, None]
               + e * dfc[:, None])
        wg2 = W[:, :NS * n_g2].reshape(NS, NS, n_g2)
        coef = np.einsum("pb,pb->p", wg2[zid[i], zid[j]], dg2)
        np.add.at(dE, j, coef[:, None] * u)
        np.add.at(dE, i, -coef[:, None] * u)


        wg4 = W[:, NS * n_g2:].reshape(NS, NP, n_g4)
        nb, dist, vec, valid, spn, fcm, K = self._pack(atoms, i, j, d, D, zid)
        triu = np.triu(np.ones((K, K), bool), 1)[None]
        block = max(1, int(_MAX_BLOCK_ELEMENTS // max(K * K, 1)))
        for a0 in range(0, n, block):
            a1 = min(a0 + block, n)
            B = a1 - a0
            dm, vm = dist[a0:a1], vec[a0:a1]
            vb, sb, nbb, zc = valid[a0:a1], spn[a0:a1], nb[a0:a1], zid[a0:a1]
            uij = np.zeros_like(vm)
            uij[vb] = vm[vb] / dm[vb][:, None]
            dot = np.einsum("akx,alx->akl", vm, vm)
            den = dm[:, :, None] * dm[:, None, :]
            cos = np.where(den > 1e8, 0.0, dot / np.where(den > 1e8, 1.0, den))
            rjk2 = np.clip(dm[:, :, None] ** 2 + dm[:, None, :] ** 2 - 2 * dot, 1e-12, None)
            rjk = np.sqrt(rjk2)
            ok = (vb[:, :, None] & vb[:, None, :]) & triu & (rjk < self.r_cut)
            if not ok.any():
                continue
            R2 = dm[:, :, None] ** 2 + dm[:, None, :] ** 2 + rjk2
            fij = np.where(dm < self.r_cut, self._cut(np.clip(dm, 0, self.r_cut)), 0.0)
            fjk = np.where(rjk < self.r_cut, self._cut(np.clip(rjk, 0, self.r_cut)), 0.0)
            Fp = fij[:, :, None] * fij[:, None, :] * fjk
            dfij = np.where(dm < self.r_cut, self._dcut(np.clip(dm, 0, self.r_cut)), 0.0)
            dfjk = np.where(rjk < self.r_cut, self._dcut(np.clip(rjk, 0, self.r_cut)), 0.0)
            pid = self._pair_ids(sb, ok)
            okm = pid >= 0
            wtri = wg4[zc[:, None, None], np.clip(pid, 0, NP - 1)]

            C_cos = np.zeros((B, K, K))
            C_ij = np.zeros((B, K, K))
            C_ik = np.zeros((B, K, K))
            C_jk = np.zeros((B, K, K))
            for gi, (eta, zeta, lam) in enumerate(self.g4):
                ang = np.clip(1 + lam * cos, 1e-12, None)
                Cz = ang ** zeta
                Ez = np.exp(-eta * R2)
                w = wtri[:, :, :, gi] * 2.0 ** (1 - zeta) * okm
                C_cos += w * zeta * ang ** (zeta - 1) * lam * Ez * Fp
                C_ij += w * Cz * Ez * (-2 * eta * dm[:, :, None] * Fp
                                       + dfij[:, :, None] * fij[:, None, :] * fjk)
                C_ik += w * Cz * Ez * (-2 * eta * dm[:, None, :] * Fp
                                       + fij[:, :, None] * dfij[:, None, :] * fjk)
                C_jk += w * Cz * Ez * (-2 * eta * rjk * Fp
                                       + fij[:, :, None] * fij[:, None, :] * dfjk)

            ujk = np.zeros((B, K, K, 3))
            diff = vm[:, None, :, :] - vm[:, :, None, :]
            ujk[okm] = diff[okm] / rjk[okm][:, None]
            safe_j = np.where(dm[:, :, None, None] > 1e8, 1.0, dm[:, :, None, None])
            safe_k = np.where(dm[:, None, :, None] > 1e8, 1.0, dm[:, None, :, None])
            dcos_dj = (uij[:, None, :, :] - cos[..., None] * uij[:, :, None, :]) / safe_j
            dcos_dk = (uij[:, :, None, :] - cos[..., None] * uij[:, None, :, :]) / safe_k

            gj = C_cos[..., None] * dcos_dj + C_ij[..., None] * uij[:, :, None, :] - C_jk[..., None] * ujk
            gk = C_cos[..., None] * dcos_dk + C_ik[..., None] * uij[:, None, :, :] + C_jk[..., None] * ujk
            gi_ = -(gj + gk)

            jj = np.broadcast_to(nbb[:, :, None], okm.shape)[okm]
            kk = np.broadcast_to(nbb[:, None, :], okm.shape)[okm]
            ii = np.broadcast_to(np.arange(a0, a1)[:, None, None], okm.shape)[okm]
            np.add.at(dE, jj, gj[okm])
            np.add.at(dE, kk, gk[okm])
            np.add.at(dE, ii, gi_[okm])
        return -dE


    def to_dict(self) -> dict:
        return {"species": self.species, "r_cut": self.r_cut,
                "g2": self.g2, "g4": self.g4}

    @classmethod
    def from_dict(cls, d: dict) -> "ACSF":
        return cls(species=d["species"], r_cut=d["r_cut"], g2=d["g2"], g4=d["g4"])
