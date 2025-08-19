import numpy as np
import warnings

from scipy.spatial import Voronoi

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

from DIRECT_abstract import DIRECT

warnings.filterwarnings("ignore", category=ConvergenceWarning)
       
""" Region Definition """
""" Rect (original DIRECT)"""
class Rect:
    def __init__(self, lx, ly, ux, uy, f):
        # rect boundary
        self.llx = lx   # lower left
        self.lly = ly
        self.urx = ux   # upper right
        self.ury = uy
        
        # center pos
        self.cx = (lx+ux)/2.0
        self.cy = (ly+uy)/2.0
        
        self.f = f
        self.d = self.distance()
    
    def distance(self):
        return round(0.5 * max(self.urx - self.llx, self.ury - self.lly), 10) # DIRECT_L (Locally Biased)
        #return round(np.sqrt((self.cx-self.llx)**2 + (self.cy-self.lly)**2), 6) # DIRECT original
        
    def divide_w(self, ret):
        w = self.urx - self.llx
        
        llx = self.llx
        urx = self.urx
        self.llx += w/3.0
        self.urx -= w/3.0
        ret.append([self.urx, self.lly, urx, self.ury])
        ret.append([llx, self.lly, self.llx, self.ury])
        
        self.d = self.distance()

    def divide_h(self, ret):
        h = self.ury - self.lly
        
        lly = self.lly
        ury = self.ury
        self.lly += h/3.0
        self.ury -= h/3.0
        ret.append([self.llx, self.ury, self.urx, ury])
        ret.append([self.llx, lly, self.urx, self.lly])
        
        self.d = self.distance()
        
    def __repr__(self):
        return f"Rect [{self.cx}\t{self.cy}] - f:\t{self.f}\td:\t{self.d}"

""" BiRect (BIRECT) """
class BiRect:
    def __init__(self, llx, lly, urx, ury, clx, cly, cux, cuy, lf, uf):
        self.llx = llx  # lower left
        self.lly = lly
        self.urx = urx  # upper right
        self.ury = ury
        self.clx = clx  # center lower
        self.cly = cly
        self.cux = cux  # center upper
        self.cuy = cuy
        self.lf = lf    # lower f
        self.uf = uf    # upper f
        
        self.d = self.distance()
        
    def distance(self):
        dx = min(self.clx - self.llx, self.urx - self.clx)
        dy = self.ury - self.cly
        return np.sqrt(dx**2 + dy**2)
    
    def get_best(self):
        return min(self.cuf, self.clf)
    
    def divide(self):
        w = self.urx - self.llx
        h = self.ury - self.lly
        
        # ret: pos
        # -> [llx, lly, urx, ury]
        # fret: f (objective)
        # -> [x, y, f] (0.0 f means not yet measure)
        ret, fret = [], []
        if w >= h:
            self.divide_w(ret, fret)
        else:
            self.divide_h(ret, fret)
            
        return ret, fret

    # divide width has two case
    # 1) center lower is left relatively
    # 2) center lower is right relatively
    def divide_w(self, ret, fret):
        nx = (self.llx + self.urx) / 2.0
        
        ret.append([nx, self.lly, self.urx, self.ury])
        ret.append([self.llx, self.lly, nx, self.ury])
        
        if self.clx > self.cux:
            fret.append([[self.clx, self.cly, self.lf], [nx+self.urx-self.clx, self.cuy, 0.0]])
            fret.append([[nx+self.llx-self.cux, self.cly, 0.0], [self.cux, self.cuy, self.uf]])
        else:
            fret.append([[nx+self.urx-self.cux, self.cly, 0.0], [self.cux, self.cuy, self.uf]])
            fret.append([[self.clx, self.cly, self.lf], [nx+self.llx-self.clx, self.cuy, 0.0]])
            
    # divide heigth has only one case
    # center lower -> center upper of new rect
    # center upper -> center lower of new rect
    def divide_h(self, ret, fret):
        ny = (self.lly + self.ury) / 2.0
        
        ret.append([self.llx, ny, self.urx, self.ury])
        ret.append([self.llx, self.lly, self.urx, ny])
        
        fret.append([[self.cux, self.cuy, self.uf], [self.clx, ny+self.ury-self.cuy, 0.0]])
        fret.append([[self.cux, ny+self.lly-self.cly, 0.0], [self.clx, self.cly, self.lf]])
        
    def __repr__(self):
        return f"BiRect [{self.clx}\t{self.cly}\t,\t{self.cux}\t{self.cuy}] - f:\t{self.get_best()}\td:\t{self.d}"
        
""" Voronoi (eDIRECT) """
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
        return f"Voronoi [{self.pos}] - f:\t{self.f}\td:\t{self.d}"

""" original DIRECT (DIvide RECTangle) """
class oDIRECT(DIRECT):
    def __init__(self, bounds, maxfun, eps=1e-4, len_tol=0.02, path=None, debug=False, isPrint=True):
        super().__init__(bounds, maxfun, eps, len_tol, path, debug, isPrint)
        
        self.initial()
        self.mem = {}
        
    # DIRECT starts center
    def initial(self):
        cx, cy = 0.5, 0.5
        f = self.objective(cx, cy)
        initRect = Rect(0.0, 0.0, 1.0, 1.0, f)
        
        self.insert_region(initRect)
        
    def objective_mem(self, x, y, round_key=5):
        scale = 10 ** round_key
        rx, ry = int(round(x*scale)), int(round(y*scale))
        
        getValue = self.mem.get((rx, ry))
        if getValue is not None:
            return getValue
        
        ret = self.objective(x, y)
        self.mem[(rx, ry)] = ret
        
        return ret
        
    """ Inheritance Implementation """
    def divide(self, PO):
        for i in PO:
            rect = self.regions[i.idx]
            
            ret, fret = self.check_side(rect)
            
            for jdx, j in enumerate(ret):
                f = fret[jdx]
                new_rect = Rect(j[0], j[1], j[2], j[3], f)
                self.insert_region(new_rect)
    
    # Rect updates itself when divide -> just return
    def update_regions(self):      
        return
        
    """ ========================== """
    # if w == h, divide having better objective value region
    # pre-measure, caching them for later dividing
    # upper to lower, right to left
    def check_side(self, r : Rect):
        w = r.urx - r.llx
        h = r.ury - r.lly
        
        ret = []
        fret = []
        if abs(w - h) < 1e-12:
            frx = self.objective_mem((5*r.urx+r.llx)/6.0, r.cy)
            flx = self.objective_mem((r.urx+5*r.llx)/6.0, r.cy)
            wx = min(flx, frx)
            
            fuy = self.objective_mem(r.cx, (5*r.ury+r.lly)/6.0)
            fly = self.objective_mem(r.cx, (r.ury+5*r.lly)/6.0)
            wy = min(fly, fuy)
            
            if wx > wy:
                r.divide_h(ret)
                fret = [fuy, fly]
            else:
                r.divide_w(ret)
                fret = [frx, flx]
        elif w > h:
            frx = self.objective_mem((5*r.urx+r.llx)/6.0, r.cy)
            flx = self.objective_mem((r.urx+5*r.llx)/6.0, r.cy)
            r.divide_w(ret)
            fret = [frx, flx]
        else:
            fuy = self.objective_mem(r.cx, (5*r.ury+r.lly)/6.0)
            fly = self.objective_mem(r.cx, (r.ury+5*r.lly)/6.0)
            r.divide_h(ret)
            fret = [fuy, fly]
            
        return ret, fret
    
    def get_datas(self):
        ret = []
        for key in self.mem:
            f = self.mem[key]
            ret.append([key[0], key[1], f])
            
        return ret

""" bDIRECT (biDIRECT) """
class bDIRECT(DIRECT):
    def __init__(self, bounds, maxfun, eps=1e-4, len_tol=0.02, path=None, debug=False, isPrint=True):
        super().__init__(bounds, maxfun, eps, len_tol, path, debug, isPrint)
        
        self.initial()
        self.delete_list = []
        
        # BIRECT starts center
        def initial(self):
            nv = 1.0/3.0
            nv2 = 2.0*nv
            lf = self.objective(nv, nv)
            uf = self.objecitve(nv2, nv2)
            initRect = Rect(0.0, 0.0, 1.0, 1.0, nv, nv, nv2, nv2, lf, uf)
            
            self.insert_region(initRect)
            
        """ Inheritance Implementation """
        def divide(self, PO):
            for i in PO:
                rect = self.regions[i.idx]
                ret, fret = rect.divide()

                for jdx, j in enumerate(ret):
                    lower = fret[jdx][0]
                    upper = fret[jdx][1]
                    
                    if lower[2] == 0.0:
                        lower[2] = self.objective(lower[0], lower[1])
                    else:
                        upper[2] = self.objective(upper[0], upper[1])
                    
                    new_rect = Rect(j[0], j[1], j[2], j[3], lower[0], lower[1], upper[0], upper[1], lower[2], upper[2])
                    self.insert_region(new_rect)
                
                self.delete_list.append(i.idx)
        
        # Exception : Rect should be deleted after divide in BIRECT
        def update_regions(self):
            for idx in self.delete_list:
                self.regions.pop(idx, None)
            self.delete_list.clear()
            
        """ ========================== """
    
""" eDIRECT (enhanced DIvide RECTangle) """
class eDIRECT(DIRECT):
    def __init__(self, bounds, maxfun, eps=1e-4, len_tol=0.02, path=None, debug=False, isPrint=True, nCand=1024, input_v=None):
        # input params
        super().__init__(bounds, maxfun, eps, len_tol, path, debug, isPrint)
        self.nCand = nCand

        # gaussian params
        self.gp = None
        self.gp_fit_n = 0
        self.gp_opt_cnt = 0
        self.gp_refit_cnt = 0
        
        self.gp_refit_each = 6
        self.gp_opt_each = 10
        self.seed = 0
        
        self.rng = np.random.default_rng()
            
        if input_v is None:
            self.initial_tmp_3() # at least 3 points
        else:
            self.initial_input(input_v)
        
    # eDIRECT starts spatial points (original : Hyper Latin Cube)
    # very simple implementation of least 3 points
    def initial_tmp_3(self):
        v1 = Voro([0.1, 0.05], self.objective(0.1, 0.05))
        self.insert_region(v1)
        
        v2 = Voro([0.2, 0.8], self.objective(0.2, 0.8))
        self.insert_region(v2)
        
        v3 = Voro([0.6, 0.6], self.objective(0.6, 0.6))
        self.insert_region(v3)
        
        self.update_regions()
        
    def initial_input(self, input_v):
        for i in input_v:
            x, y, f = i[0], i[1], i[2]
            ux, uy = self.to_unit(x, isX=True), self.to_unit(y, isX=False)
            newVoro = Voro([ux, uy], f)
            self.insert_region(newVoro)
        
        self.update_regions()
    
    """ helper methods """
    # use when input_v given
    def update_best(self, x, y, f):
        if f < self.best_f:
            self.best_x = x
            self.best_y = y
            self.best_f = f
        
    """ voronoi methods """       
    # get Voronoi Boundary points : Sutherland-Hodgman polygon clipping algorithm
    def voronoi_construct(self, poly, n, c, tol=1e-12):
        if poly.size == 0:
            return poly
        
        res = []
        k = len(poly)
        for idx in range(k):
            # idx i, i+1 iteration
            before = poly[idx]
            target = poly[(idx + 1) % k]
            
            # determine each point inside
            fBefore = np.dot(n, before) - c
            fTarget = np.dot(n, target) - c
            inBefore = fBefore <= tol
            inTarget = fTarget <= tol
            
            # target compare neighbor
            # case 1. inside completely
            # case 2. target is outside -> intersection add
            # case 3. target inside, neighbor outside -> target, intersection add
            # case 4. outside
            
            if inBefore and inTarget:           # case 1
                res.append(target)
            elif inBefore and not inTarget:     # case 2
                t = fBefore / (fBefore - fTarget + 1e-30)
                intersect = before + t * (target - before)
                res.append(intersect)
            elif (not inBefore) and inTarget:   # case 3
                t = fBefore / (fBefore - fTarget + 1e-30)
                intersect = before + t * (target - before)
                res.append(intersect)
                res.append(target)
            else:                               # case 4
                pass
        return np.array(res)

    # X : np.array([ [x, y], ...   ])
    def calc_voronoi(self):
        X = np.array([self.regions[key].pos for key in self.regions], float)

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
            # for a, is close with xi than xj?
            # ||a - xi||^2 >= ||a - xj||^2
            # => (xj - xi)*a <= (||xj||^2 - ||xi||^2) / 2
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
                dists = np.linalg.norm(poly - xi, axis=1)  # distance each points
                k = int(np.argmax(dists))                  # distance max index
                di[i] = dists[k]                           # i-th voronoi maximum distance -> di[i]
                far_idx[i] = k                             # i-th voronoi maximum distance index -> far_idx[i]

        return polys, di, far_idx
    
    """ Kriging, GP methods """
    def sample_in_convex_poly(self, poly, n = 1024):
        poly = np.asarray(poly, float)
        if poly.ndim != 2 or len(poly) < 3:
            return np.empty((0, 2), float)
        
        # split to triangles
        v0 = poly[0]
        tris = [(v0, poly[i], poly[i+1]) for i in range(1, len(poly)-1)]
        areas = np.array([0.5*abs(np.cross(b-a, c-a)) for (a,b,c) in tris])
        probs = areas / (areas.sum() + 1e-30)

        # sampling
        out = []
        for _ in range(n):
            t = self.rng.choice(len(tris), p=probs)
            a,b,c = tris[t]
            u, v = self.rng.random(), self.rng.random()
            if u + v > 1.0:
                u, v = 1.0 - u, 1.0 - v
            p = a + u*(b-a) + v*(c-a)
            out.append(p)
        return np.array(out)

    def refit_gp(self, force=False):
        n = len(self.regions)
        if n < 2:
            self.gp = None
            return False
        
        need_refit = force or (self.gp is None) or (n - self.gp_fit_n >= self.gp_refit_cnt)
        if not need_refit:
            return False
        
        initial_opt = (self.gp_opt_cnt == 0 and n >= 10)
        optimize_now = initial_opt or (self.gp_refit_cnt % self.gp_opt_each == 0)
        
        self.fit_gp(n_restarts=2, seed=self.seed, optimize=optimize_now, warm_start=True)
        
        self.gp_fit_n = n
        if optimize_now:
            self.gp_opt_cnt = n
        self.gp_refit_cnt += 1
        
        return True
        
    def fit_gp(self, n_restarts=2, seed=0, optimize=True, warm_start=True):
        X = np.array([self.regions[key].pos for key in self.regions], float)
        y = np.array([self.regions[key].f   for key in self.regions], float)
    
        if warm_start and (self.gp is not None) and hasattr(self.gp, "kernel_"):
            gp_kernel = self.gp.kernel_
        else:
            cKernel = C(1.0, (1e-3, 1e3))
            mKernel = Matern(length_scale = np.full(X.shape[1], 0.2), length_scale_bounds=(1e-3, 1e3), nu=2.5)
            wKernel = WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-12, 1e-2))
            gp_kernel = cKernel * mKernel + wKernel
        
        optimizer = "fmin_l_bfgs_b" if optimize else None
        gp = GaussianProcessRegressor(kernel=gp_kernel,
                                      normalize_y=True,
                                      n_restarts_optimizer=n_restarts,
                                      optimizer=optimizer,
                                      random_state=seed)
        gp.fit(X, y)
        self.gp = gp
        
    """ Inheritance Implementation """
    def divide(self, PO):
        biggest = self.regions[PO[-1].idx]
        self.divide_far(biggest)
        
        for i in range(len(PO)-1):
            voro = self.regions[PO[i].idx]
            if voro.f > 60:
                self.divide_far(voro)
            else:
                self.divide_kriging(voro)
    
    # Need to new voronoi (boundary, d, far is None)        
    def update_regions(self):      
        boundary, d, far = self.calc_voronoi()
        
        for idx, i in enumerate(boundary):
            voro = self.regions[idx]
            voro.update(i, d[idx], far[idx])
                
    """ ========================== """
    
    def divide_far(self, v : Voro):
        x, y = v.far_point()
        f = self.objective(x, y)
        
        new_voronoi = Voro([x, y], f)
        self.insert_region(new_voronoi)


    def divide_kriging(self, v: Voro):
        poly = v.boundary
        if poly is None or poly.size < 6 or v.far < 0:
            self.divide_far(v)
            return
    
        # create candidates
        cand = self.sample_in_convex_poly(poly, n=self.nCand)
        if cand.size == 0:
            self.divide_far(v)
            return
    
        # normalize
        xi = np.array(v.pos, float)
        dist = np.linalg.norm(cand - xi, axis=1)
        dist_n = dist / (v.d + 1e-12)
    
        # Kriging
        self.refit_gp()
        
        # gp fit not ready
        if self.gp is None:
            self.divide_far(v)
            return
    
        mu, sigma = self.gp.predict(cand, return_std=True)
    
        # calculate weight
        dmax = max((self.regions[k].d for k in self.regions), default=v.d)
        wi = float(1.0 - min(v.d / (dmax + 1e-12), 1.0))  # wi ∈ [0,1]
    
        # 5) score - smaller is better
        # 5-1) distance
        # normalize
        mu_min, mu_max = float(np.min(mu)), float(np.max(mu))
        mu_n = (mu - mu_min) / (mu_max - mu_min + 1e-12)
        score = wi * mu_n - (1.0 - wi) * dist_n
        
        # 5-1) LCB
        # beta = 1.0 + 0.2*np.log(len(self.regions) + 1)
        # acq = mu - beta*sigma
        # score = wi*acq - (1.0 - wi)*dist_n

        idx = int(np.argmin(score))
        x_new, y_new = cand[idx]
        f_new = self.objective(x_new, y_new)
    
        new_v = Voro([x_new, y_new], f_new)
        self.insert_region(new_v)























