import numpy as np
from dataclasses import dataclass
from abc import abstractmethod, ABCMeta

from scipy.interpolate import RectBivariateSpline

@dataclass
class Datas:
    idx : int # index
    d   : float # distance
    f   : float # power
        
""" DIRECT abstact """
class DIRECT(metaclass = ABCMeta):
    def __init__(self, bounds, maxfun, eps=1e-4, len_tol=0.02, path=None, debug=False, isPrint=True):
        # input params
        self.genId = 0
        self.bounds = bounds
        self.maxfun = maxfun
        self.eps = eps
        self.len_tol = len_tol
        self.path = path
        self.debug = debug
        self.isPrint = isPrint
        
        # record params
        self.best_x = 0.0
        self.best_y = 0.0
        self.best_f = 99999.9 # minimize
        self.cnt = 0
        
        self.regions = {}
        
        self.tester = None
        if self.path is not None:
            self.init_test(self.path)

    """ Test Function """
    """ x, y, f -> split and interpolation """
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
        
        if self.tester is not None:
            ux = self.from_unit(x, isX=True)
            uy = self.from_unit(y, isX=False)
            ret = -self.tester(uy, ux)[0][0]
        else:
            print("TODO : real objective")
        
        if self.debug:
            print(ux, uy, ret)
        
        self.update(ux, uy, ret)
        return ret  
    
    # is Counter Clock Wise
    def CCW(self, o1 : Datas, o2 : Datas, ot : Datas):
        isCCW = (o2.d - o1.d) * (ot.f - o1.f) - (ot.d - o1.d) * (o2.f - o1.f)
        return isCCW > 0
    
    def insert_region(self, region):
        ids = self.genId
        self.genId += 1
        
        self.regions[ids] = region
        
    """ main methods """
    # select Potentail Optimal
    def select_PO(self):
        PO = []
        regions = []
        for key in self.regions:
            region = self.regions[key]
            regions.append(Datas(key, region.d, region.f))
        regions.sort(key=lambda x: (x.d, x.f))
        
        cur_d = -1.0
        for region in regions:
            if abs(cur_d - region.d) < 1e-10 or region.d < self.len_tol:
                continue
            
            PO.append(region)
            cur_d = region.d
        
        convex_hull = self.lower_convex_hull(PO)
        return self.eps_filter(convex_hull)
    
    # Andrew's Monotone Chain Algorithm - (Graham Scan)
    def lower_convex_hull(self, PO):
        convex_hull = []
        for target in PO:
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

    # eps filter with Lipschitz Constant
    def eps_filter(self, convex_hull):
        ret = []
        if len(convex_hull) == 0:
            return ret
        
        compare = self.best_f - self.eps * abs(self.best_f)
        for i in range(len(convex_hull)-1):
            target = convex_hull[i]
            target_next = convex_hull[i+1]

            k = (target_next.f - target.f) / (target_next.d - target.d)
            value = target.f - k*target.d
            
            if value <= compare:
                ret.append(target)

        # Biggest Region should be divided
        ret.append(convex_hull[-1])

        return ret

    """ Inheritance Implementation """
    @abstractmethod
    def divide(self, PO):
        pass

    @abstractmethod
    def update_regions(self):
        pass
                        
    """ run method """
    def direct(self):
        while self.cnt < self.maxfun:
            # if self.debug:
            #     print("\n======================")
            PO = self.select_PO()
            if len(PO) == 0:
                break
            
            self.divide(PO)
            self.update_regions()
        
        if self.isPrint:
            print(f"\nResults : {self.cnt} evals - [{self.best_x:.6f}\t{self.best_y:.6f}]\t{self.best_f:.6f}")
            