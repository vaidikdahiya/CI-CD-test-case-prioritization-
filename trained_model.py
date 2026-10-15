import json
import numpy as np

class TrainedModel:
    def __init__(self, path):
        c = json.load(open(path))
        self.features = c['features']
        self.mean, self.std = np.array(c['mean']), np.array(c['std'])
        w = c['weights']
        self.mu, self.ls = np.array(w['mu']), np.array(w['log_sigma'])
        self.w, self.b = np.array(w['w']), np.array(w['b'])
        self.unknown_age = c.get('unknown_last_failure_age', 1000)

    def score(self, df):
        X = df[self.features].astype(float).replace(-1, np.nan)
        X['REC_LastFailureAge'] = X['REC_LastFailureAge'].fillna(self.unknown_age)
        X = np.log1p(X.fillna(0).values)
        X = np.clip((X - self.mean) / self.std, -5, 5)
        d = (X[:, None, :] - self.mu[None]) / np.exp(self.ls)[None]
        lf = -0.5 * (d ** 2).sum(2)
        f = np.exp(lf - lf.max(1, keepdims=True))
        f /= f.sum(1, keepdims=True)
        return (f * (X @ self.w.T + self.b)).sum(1)

    def rank(self, df):
        out = df.copy()
        out['score'] = self.score(df)
        return out.sort_values('score', ascending=False).reset_index(drop=True)
