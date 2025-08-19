import numpy as np
from scipy.spatial import Voronoi
from scipy.interpolate import RectBivariateSpline
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
       
""" Voronoi Definition """
class Voro:
    def __init__(self, pos, f):
        self.pos = pos
        self.f = f
        
        self.boundary = None
        self.d = -1.0
        self.far = -1
        
    def update(self, boundary, d, far):
        self.boundary = boundary
        self.d = d
        self.far = far
    
    def far_point(self):
        return self.boundary[self.far]
    
    def __repr__(self):
        return f"Voronoi {self.pos}, {self.f}, {self.d}"

""" eDIRECT (enhanced DIvide RECTangle) """
class eDIRECT:
    def __init__(self, bounds, maxfun, maxiter= 0, eps=1e-4, nCand=1024, len_tol=0.02, debug=False, path=None, input_v=None):
        # input params
        self.genId = 0
        self.bounds = bounds
        self.maxfun = maxfun
        self.maxiter = maxiter
        self.eps = eps
        self.nCand = nCand
        self.len_tol = len_tol
        self.debug = debug
        
        # essential params
        self.best_x = 0.0
        self.best_y = 0.0
        self.best_f = 99999.9
        self.cnt = 0

        # gaussian params
        self.gp = None
        self.rng = np.random.default_rng()
        
        # voronoi object
        self.voronoi = {}
        
        self.tester = None
        # Test Mode -> interpolator with path
        if path is not None:
            self.init_test(path)
            
        if input_v is None:
            self.initial_tmp_3() # at least 3 points
        else:
            self.initial_input(input_v)
        
    def initial_tmp_3(self):
        v1 = Voro([0.1, 0.05], self.objective(0.1, 0.05))
        self.insert_voronoi(v1)
        
        v2 = Voro([0.2, 0.8], self.objective(0.2, 0.8))
        self.insert_voronoi(v2)
        
        v3 = Voro([0.6, 0.6], self.objective(0.6, 0.6))
        self.insert_voronoi(v3)
        
        self.update_voronoi()
        
    def initial_input(self, input_v):
        for i in input_v:
            x,y,f = i[0], i[1], i[2]
            ux, uy = self.to_unit(x, isX=True), self.to_unit(y, isX=False)
            newVoro = Voro([ux, uy], f)
            self.insert_voronoi(newVoro)
        
        self.update_voronoi()
        
    def init_test(self, path):
        data = np.loadtxt(path, delimiter="\t")
        x_all, y_all = data[:, 0], data[:, 1]
        xu, yu   = np.unique(x_all), np.unique(y_all)
        z_grid   = np.zeros((len(yu), len(xu)))
        for x, y, v in data:
            i, j         = np.where(yu == y)[0][0], np.where(xu == x)[0][0]
            z_grid[i, j] = v
        self.tester = RectBivariateSpline(yu, xu, z_grid)
        
    """ helper methods """
    """ normalize methods """
    def to_unit(self, v, isX : bool):
        xlb, xub, ylb, yub = self.bounds[0][0], self.bounds[0][1], self.bounds[1][0], self.bounds[1][1]
        
        return (v - xlb) / (xub - xlb) if isX else (v - ylb) / (yub - ylb)
    
    def from_unit(self, v, isX : bool):
        xlb, xub, ylb, yub = self.bounds[0][0], self.bounds[0][1], self.bounds[1][0], self.bounds[1][1]
        
        return v * (xub - xlb) + xlb if isX else v * (yub - ylb) + ylb

    def update(self, x, y, f):
        if self.best_f > f:
            self.best_f = f
            self.best_x = x
            self.best_y = y
            
    def objective(self, x, y):
        self.cnt += 1
        ux = self.from_unit(x, isX=True)
        uy = self.from_unit(y, isX=False)
        ret = -self.tester(uy, ux)[0][0]
        #if self.debug:
        #    print(ux, uy, ret, self.cnt, x, y)
        #else:
        #    print(ux, uy, ret)
        
        #print(ux, uy, ret)
        self.update(ux, uy, ret)
        return ret    

    """ voronoi methods """
    def insert_voronoi(self, v : Voro):
        ids = self.genId
        self.genId += 1
        
        self.voronoi[ids] = v
        
    # Need to new voronoi (boundary, d, f is None)
    def update_voronoi(self):
        boundary, d, far = self.calc_voronoi()
        
        for idx, i in enumerate(boundary):
            voro = self.voronoi[idx]
            voro.update(i, d[idx], far[idx])
        
    def voronoi_construct(self, poly, n, c, tol=1e-12):
        if poly.size == 0:
            return poly
        
        res = []
        k = len(poly)
        for idx in range(k):
            A = poly[idx]
            B = poly[(idx + 1) % k]
            fa = np.dot(n, A) - c
            fb = np.dot(n, B) - c
            inA = fa <= tol
            inB = fb <= tol
            if inA and inB:
                res.append(B)
            elif inA and not inB:
                t = fa / (fa - fb + 1e-30)
                I = A + t * (B - A)
                res.append(I)
            elif (not inA) and inB:
                t = fa / (fa - fb + 1e-30)
                I = A + t * (B - A)
                res.append(I)
                res.append(B)
            else:
                pass
        return np.array(res)

    # X : np.array([ [x, y], ...   ])
    def calc_voronoi(self):
        tmpX = [self.voronoi[k].pos for k in self.voronoi]
        X = np.array(tmpX, float)

        m = len(X)
        box = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], float) # (0, 1) normalize

        # scipy.spatial.Voronoi
        vor = Voronoi(X)

        # Delaunay from ridge_points
        neigh = [set() for _ in range(m)]
        for (i, j) in vor.ridge_points:
            neigh[i].add(j); neigh[j].add(i)

        polys = []
        di = np.zeros(m, float)
        far_idx = np.full(m, -1, dtype=int)
        
        for i in range(m):
            poly = box.copy()
            xi = X[i]
            # 이웃 j와의 수직이등분선: (xj - xi)·x <= (||xj||^2 - ||xi||^2)/2
            for j in neigh[i]:
                xj = X[j]
                n = xj - xi
                c = 0.5 * (np.dot(xj, xj) - np.dot(xi, xi))
                poly = self.voronoi_construct(poly, n, c)
                if poly.size == 0:
                    break
            polys.append(poly)
            if poly.size == 0:
                di[i] = 0.0
            else:
                dists = np.linalg.norm(poly - xi, axis=1)  # 꼭짓점까지 거리들
                k = int(np.argmax(dists))                  # <<< 추가: 최장 꼭짓점 '인덱스'
                di[i] = dists[k]                           # 최대 거리
                far_idx[i] = k                             # <<< 추가: 인덱스 저장

        return polys, di, far_idx
    
    # check Counter Clock Wise
    def CCW(self, v1, v2, vt):
        v1_d, v2_d, vt_d = v1[1], v2[1], vt[1]
        v1_f, v2_f, vt_f = v1[2], v2[2], vt[2]
        
        isCCW = (v2_d - v1_d) * (vt_f - v1_f) - (vt_d - v1_d) * (v2_f - v1_f)
        return isCCW > 0
            
    """ main method """
    def select_PO(self):
        checker = []
        tmp_checker = []
        for k in self.voronoi:
            voro = self.voronoi[k]
            tmp_checker.append([k, voro.d, voro.f])
        tmp_checker.sort(key=lambda x: (x[1], x[2]))
        
        now_d = -1.0
        for i in tmp_checker:
            if abs(now_d - i[1]) < 1e-10 or i[1] < self.len_tol:
                continue
            
            checker.append(i)
            now_d = i[1]
            
        if len(checker) == 0:
            self.cnt+=1
            return []
        
        convex_hull = self.lower_convex_hull(checker)
        
        return self.eps_filter(convex_hull)
    
    # Andrew's Monotone Chain Algorithm - (Graham Scan)
    def lower_convex_hull(self, checker):
        convex_hull = []
        for target in checker:
            while True:
                if len(convex_hull) < 2:
                    convex_hull.append(target)
                    break
                else:
                    isCCW = self.CCW(convex_hull[-2], convex_hull[-1], target)
                    if isCCW:
                        convex_hull.append(target)
                        break
                    else:
                        convex_hull.pop()
              
        return convex_hull

        
    def eps_filter(self, convex_hull):
        ret = []
        if len(convex_hull) == 0:
            return ret
        
        compare = self.best_f - self.eps * abs(self.best_f)
        for i in range(len(convex_hull)-1):
            target_d, target_f = convex_hull[i][1], convex_hull[i][2]
            next_d, next_f = convex_hull[i+1][1], convex_hull[i+1][2]

            k = (next_f - target_f) / (next_d - target_d)
            value = target_f - k*target_d
            if value <= compare:
                ret.append(convex_hull[i])

        ret.append(convex_hull[-1])

        return ret
    
    def divide(self, PO):
        if len(PO) == 0:
            return
        
        biggest = self.voronoi[PO[-1][0]]
        self.divide_far(biggest)
        
        for i in range(len(PO)-1):
            voro = self.voronoi[PO[i][0]]
            if voro.f > 60:
                self.divide_far(voro)
            else:
                #self.divide_exploit(voro, n_cand=32, p_idw=2.0)
                self.divide_exploit_kriging(voro, n_cand=self.nCand)
        
    def divide_far(self, v : Voro):
        x, y = v.far_point()
        f = self.objective(x, y)
        
        new_voronoi = Voro([x, y], f)
        self.insert_voronoi(new_voronoi)
    
    def sample_in_convex_poly(self, poly, n=32, rng=None):
        #if rng is None:
            #rng = np.random.default_rng(self.genId)  # iteration마다 시드 달라지게
        rng = self.rng
        poly = np.asarray(poly, float)
        if poly.ndim != 2 or len(poly) < 3:
            return np.empty((0, 2), float)
        v0 = poly[0]
        tris = [(v0, poly[i], poly[i+1]) for i in range(1, len(poly)-1)]
        areas = np.array([0.5*abs(np.cross(b-a, c-a)) for (a,b,c) in tris])
        probs = areas / (areas.sum() + 1e-30)
        out = []
        for _ in range(n):
            t = rng.choice(len(tris), p=probs)
            a,b,c = tris[t]
            u, v = rng.random(), rng.random()
            if u + v > 1.0:
                u, v = 1.0 - u, 1.0 - v
            p = a + u*(b-a) + v*(c-a)
            out.append(p)
        return np.array(out)


    def divide_exploit(self, v: Voro, n_cand=32, p_idw=2.0):
        poly = v.boundary
        # 폴백: 폴리곤이 비거나 꼭짓점<3 → far 사용
        if poly is None or poly.size < 6:
            if v.far >= 0:
                self.divide_far(v)
            return
    
        # (a) 셀 내부 후보 생성
        cand = self.sample_in_convex_poly(poly, n=n_cand)
        if cand.size == 0:
            if v.far >= 0:
                self.divide_far(v)
            return
    
        # (b) exploration 항: 거리 정규화 (0~1 범위)
        xi = np.array(v.pos, float)
        dist = np.linalg.norm(cand - xi, axis=1)
        dist_n = dist / (v.d + 1e-12)
    
        # (c) exploitation 항: IDW 기반 예측값 μ(z)
        #     모든 현 시점 site를 사용(가볍고 안정적)
        X = np.array([self.voronoi[k].pos for k in self.voronoi], float)
        y = np.array([self.voronoi[k].f   for k in self.voronoi], float)
        diff = cand[:, None, :] - X[None, :, :]
        D = np.linalg.norm(diff, axis=2) + 1e-12      # (R, m)
        W = 1.0 / (D**p_idw)                          # IDW weight
        mu = (W * y[None, :]).sum(axis=1) / W.sum(axis=1)
    
        # 스케일러블하게 cand-내에서 0~1 정규화
        mu_min, mu_max = mu.min(), mu.max()
        mu_n = (mu - mu_min) / (mu_max - mu_min + 1e-12)
    
        # (d) 가중치: 셀 작을수록 exploitation↑
        dmax = max((self.voronoi[k].d for k in self.voronoi), default=v.d)
        wi = float(1.0 - min(v.d / (dmax + 1e-12), 1.0))   # wi∈[0,1]
    
        # (e) 점수: 낮을수록 좋음
        score = wi * mu_n - (1.0 - wi) * dist_n
        idx = int(np.argmin(score))
        x_new, y_new = cand[idx]
        f_new = self.objective(x_new, y_new)
    
        new_v = Voro([x_new, y_new], f_new)
        self.insert_voronoi(new_v)

    def fit_gp(self, n_restarts=2, seed=0):
        # 현재 sites/values
        X = np.array([self.voronoi[k].pos for k in self.voronoi], float)
        y = np.array([self.voronoi[k].f   for k in self.voronoi], float)
    
        # 점이 너무 적으면 피팅 생략
        if len(X) < 2:
            self.gp = None
            return
    
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.2, 0.2],
                                              length_scale_bounds=(1e-2, 10.0),
                                              nu=2.5) \
                 + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      normalize_y=True,
                                      n_restarts_optimizer=n_restarts,
                                      random_state=seed)
        gp.fit(X, y)
        self.gp = gp

    def divide_exploit_kriging(self, v: Voro, n_cand=32):
        # 폴리곤이 비거나 꼭짓점 부족하면 far로 폴백
        poly = v.boundary
        if poly is None or poly.size < 6 or v.far < 0:
            self.divide_far(v)
            return
    
        # 1) 후보 생성 (셀 내부 균일)
        cand = self.sample_in_convex_poly(poly, n=n_cand)
        if cand.size == 0:
            self.divide_far(v)
            return
    
        # 2) exploration 항: 거리 정규화 (0~1)
        xi = np.array(v.pos, float)
        dist = np.linalg.norm(cand - xi, axis=1)
        dist_n = dist / (v.d + 1e-12)
    
        # 3) exploitation 항: Kriging 예측 μ(z)
        if self.gp is None:
            self.fit_gp()
        if self.gp is None:
            # 아직 피팅 불가(점 적음) → far 폴백
            self.divide_far(v)
            return
    
        mu, sigma = self.gp.predict(cand, return_std=True)
        # cand 내에서 0~1 정규화
        mu_min, mu_max = float(np.min(mu)), float(np.max(mu))
        mu_n = (mu - mu_min) / (mu_max - mu_min + 1e-12)
    
        # 4) 가중치 w_i: 셀 작을수록 exploitation↑ (간단 튜닝이 쉬움)
        #    (대안) sigma 기반: sigma_sites = self.gp.predict(X_sites, return_std=True)[1]
        #           w_i = 1 - sigma_sites_i / (sigma_sites.max()+1e-12)
        dmax = max((self.voronoi[k].d for k in self.voronoi), default=v.d)
        wi = float(1.0 - min(v.d / (dmax + 1e-12), 1.0))  # wi ∈ [0,1]
    
        # 5) 점수 = wi*mu_n - (1-wi)*dist_n  (작을수록 좋음)
        score = wi * mu_n - (1.0 - wi) * dist_n
        idx = int(np.argmin(score))
        x_new, y_new = cand[idx]
        f_new = self.objective(x_new, y_new)
    
        new_v = Voro([x_new, y_new], f_new)
        self.insert_voronoi(new_v)


    """ direct run method """
    def direct(self):
        if not self.maxiter == 0:
            iters = 0
            while iters < self.maxiter:
                iters+=1
                print("\n===== ", iters, " =====")
                PO = self.select_PO()
                self.divide(PO)
                self.update_voronoi()
        else:
            while self.cnt < self.maxfun:
                if self.debug:
                    print("\n======================")
                PO = self.select_PO()
                self.divide(PO)
                self.update_voronoi()
                # if self.voronoi.get(2):
                #     print(self.voronoi[2])
                
        #print("\n",self.cnt, self.best_x, self.best_y, self.best_f)

""" main only use for Testing """
if __name__ == "__main__":       
    
    """ Test Random Bounds """
    def rand(lb, ub, offset):
        lo = random.uniform(lb, lb+offset)
        hi = random.uniform(ub-offset, ub)

        return lo, hi
    
    """ Test Data Prepare """
    DATA_FILE, X_TARGET, Y_TARGET, X_LIM, Y_LIM, gbounds = r"D:/SVN/Practice/2dop/result2.txt", 153.0, 158.5, 15.0, 9.5, 300-60
    #DATA_FILE, X_TARGET, Y_TARGET, X_LIM, Y_LIM, gbounds = r"D:/SVN/Practice/2dop/result.txt", 75.0, 78.0, 12.0, 9.0, 150-50

    xlo, xhi = rand(0, gbounds, 50)
    ylo, yhi = rand(0, gbounds, 50)

    bounds = [(xlo, xhi), (ylo, yhi)]
    
    d = eDIRECT(bounds, 150, path=DATA_FILE, eps=1e-4, debug=False)
    d.direct()

























