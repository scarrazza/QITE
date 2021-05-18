import numpy as np


def gamma_opt(beta, AA=False):
    coeff = 2
    if AA:
        coeff = 4
    return beta / 2 * (np.sqrt(1 + coeff / beta) - 1)


def Qu(beta, gamma, eps=1e-3):
    return 2.0 * (beta/gamma + 1.0) * np.log(4.0/eps)


def alpha_beta(beta):
    return np.exp(-gamma_opt(beta)) / 2.0


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
        alphab = alpha_beta(b)
        DeltaBeta = np.diff(beta)

        # k == 0
        PsucBr = self.Psuc(beta[r-1], gamma_opt(beta[r-1]))
        prod = alpha_beta(beta[0])
        prod2 = alpha_beta(beta[0])**2
        for k in range(r-1):
            prod *= alpha_beta(DeltaBeta[k])
            prod2 *= alpha_beta(DeltaBeta[k])**2
        eps_prime = self.eps / (2 * 4.0**(r-1)) * np.sqrt(PsucBr) * prod / alphab
        Sigma = self.query(beta[0]-0, gamma_opt(beta[0]-0), eps=eps_prime) 
        if not query_depth:
            Sigma = Sigma / prod2

        # k > 0
        for k in range(r-1):
            PsucBk = 1
            if not query_depth:
                PsucBk = self.Psuc(beta[k], gamma_opt(beta[k]))
            prod = 1
            prod2 = 1
            for j in range(k, r-1):
                prod *= alpha_beta(DeltaBeta[j])
                prod2 *= alpha_beta(DeltaBeta[j])**2
            eps_prime = self.eps / 4.0**(r-(k+1)) * np.sqrt(PsucBr/PsucBk) * prod * alpha_beta(beta[k]) / alphab
            if query_depth:
                Sigma += PsucBk * self.query(DeltaBeta[k], gamma_opt(DeltaBeta[k]), eps=eps_prime)
            else:
                Sigma += PsucBk * self.query(DeltaBeta[k], gamma_opt(DeltaBeta[k]), eps=eps_prime) / alpha_beta(beta[k])**2 / prod2

            
        Psbeta = self.Psuc(beta[r-1], gamma_opt(beta[r-1]))
        if query_depth:
            Psbeta = 1
            alphab = 1
        return 1/Psbeta * Sigma * alphab**2

    def Psuc(self, beta, gamma):
        Zt =  np.sum(np.exp(-beta * (self.E - self.Emin)))
        N = 2**self.n
        return Zt / N * np.exp(-2.0 * gamma) / 4.0

    def F(self, r, beta, gamma):
        """Return linear query prediction."""
        return self.compute_query(params=None, schedule=lambda t, _: t, r=r, b=beta)

    def C(self, beta, Psbeta, gamma, alpha=1):
        bquery = self.query(beta=beta**alpha, gamma=gamma, eps=self.eps / 2 * np.sqrt(Psbeta))
        return 1/Psbeta * bquery

    def AA(self, beta, Psbeta, gamma, alpha=1):
        return 1/np.sqrt(Psbeta) * self.query(beta=beta**alpha, gamma=gamma, eps=self.eps / 2 * np.sqrt(Psbeta))

    def rF(self, beta, gamma):
        values = []
        r_range = []
        r = 2
        tol = 0
        while True:
            val = self.F(r, beta, gamma)
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

    def rFfit(self, beta, gamma):
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
                        [1.0], 'L-BFGS-B', bounds=[(1e-3, 100)])
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
