import numpy as np


def Qu(beta, eps=1e-3, deltal=1):
    return np.e*beta*deltal/2.0 + np.log(1.0/eps) / np.log(np.e + 2.0*np.log(1.0/eps)/(beta*np.e*deltal))


class FragmentedQuITE:
    def __init__(self, nqubits, energy, eps=1e-3):
        """Test function for optimization."""
        self.n = nqubits
        self.E = energy
        self.Emin = np.min(self.E)
        self.query = Qu
        self.eps = eps

    def compute_query(self, params, schedule, r, b, query_depth=False):
        """Compute query optimization."""
        beta = np.array([ b * schedule(step/r, params) for step in range(1, r+1)])

        # k == 0
        PsucBr = self.Psuc(beta[r-1])
        eps_prime = self.eps / (2 * 4.0**(r-1)) * np.sqrt(PsucBr)
        Sigma = self.query(beta[0]-0, eps=eps_prime, deltal=1)

        # k > 0
        DeltaBeta = np.diff(beta)
        for k in range(r-1):
            PsucBk = 1
            if not query_depth:
                PsucBk = self.Psuc(beta[k])
            eps_prime = self.eps / 4.0**(r-(k+1)) * np.sqrt(PsucBr/PsucBk)
            Sigma += PsucBk * self.query(DeltaBeta[k], eps=eps_prime, deltal=1)

        Psbeta = 1
        if not query_depth:
            Psbeta = self.Psuc(beta[r-1])
        return 1/Psbeta * Sigma

    def Psuc(self, beta):
        Zt =  np.sum(np.exp(-beta * (self.E - self.Emin)))
        N = 2**self.n
        return Zt / N

    def F(self, r, beta):
        """Return linear query prediction."""
        return self.compute_query(params=None, schedule=lambda t, _: t, r=r, b=beta)

    def C(self, beta, Psbeta, alpha=1):
        bquery = self.query(beta=beta**alpha, eps=self.eps / 2 * np.sqrt(Psbeta), deltal=1)
        return 1/Psbeta * bquery

    def AA(self, beta, Psbeta, alpha=1):
        return 1/np.sqrt(Psbeta) * self.query(beta=beta**alpha, eps=self.eps / 2 * np.sqrt(Psbeta), deltal=1)

    def rF(self, beta):
        values = []
        r_range = []
        r = 2
        tol = 0
        while True:
            val = self.F(r, beta)
            if len(values) > 0:
                if values[-1] < val:
                    tol += 1
                    if tol > 2:
                        break
            values.append(val)
            r_range.append(r)
            r += 1
        f = np.min(values)
        f_r_best = r_range[np.argmin(values)]
        f_depth = self.compute_query(params=None, schedule=lambda t,_: t,
                                     r=f_r_best, b=beta, query_depth=True)
        return f, f_r_best, f_depth

    def rFfit(self, beta):
        from scipy.optimize import minimize
        def schedule(t, params):
            return t**params[0]
        values = []
        params = []
        r_range = []
        r = 2
        tol = 0
        while True:
            m = minimize(lambda p, _: self.compute_query(p, schedule, r, beta),
                        [1.0], 'L-BFGS-B', bounds=[(1e-3, 1e3)])
            if len(values) > 0:
                if values[-1] < m.fun:
                    tol += 1
                    if tol > 2:
                        break
            values.append(m.fun)
            params.append(m.x)
            r_range.append(r)
            r += 1
        f = np.min(values)
        f_r_best = r_range[np.argmin(values)]
        f_depth = self.compute_query(params=params[np.argmin(values)],
                                     schedule=schedule,
                                     r=f_r_best, b=beta, query_depth=True)
        return f, f_r_best, f_depth, params[np.argmin(values)]
