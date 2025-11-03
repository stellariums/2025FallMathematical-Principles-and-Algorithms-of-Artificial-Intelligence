##### start of Listing 5.9 ##### 
import math

class Integrator:
    def __init__(self, a, b, n):
        self.a, self.b, self.n = a, b, n
        self.points, self.weights = self.compute_points()

    def compute_points(self):
        raise NotImplementedError(self.__class__.__name__)

    def integrate(self, f):  
        value=0
        for i,w in zip(self.points, self.weights):
            value+=f(i)*w
        return value

class Trapezoidal(Integrator):
    def compute_points(self):  
        weights=[]
        h=(self.b-self.a)/self.n
        for i in range(self.n+1):
            if i==0 or i==self.n:
                weights.append(h/2)
            else:
                weights.append(h)
        return [self.a+((self.b-self.a)/self.n)*i for i in range(self.n+1)],weights

class Simpson(Integrator):
    def compute_points(self):  
        if self.n%2!=0:
            self.n=self.n+1
        weights=[]
        h=(self.b-self.a)/self.n
        for i in range(self.n+1):
            if i==0 or i==self.n:
                weights.append(h/3)
            elif i%2==0:
                weights.append(h*2/3)
            else:
                weights.append(h*4/3)
        return [self.a+((self.b-self.a)/self.n)*i for i in range(self.n+1)],weights

class GaussLegendre(Integrator):
    def compute_points(self):  
        if self.n%2==0:
            self.n=self.n+1
        weights=[]
        points=[]
        h=2*(self.b-self.a)/(self.n+1)
        for i in range(self.n+1):
            if i%2==0:
                points.append(self.a+((i+1)/2)*h-(math.sqrt(3)/6)*h)
                weights.append(h/2)
            else:
                points.append(self.a+((i)/2)*h+(math.sqrt(3)/6)*h)
                weights.append(h/2)
        return points,weights

def test():
    def f(x): return (x * math.cos(x) + math.sin(x)) * \
                      math.exp(x * math.sin(x))
    def F(x): return math.exp(x * math.sin(x))

    a = 2; b = 3; n = 200
    I_exact = F(b) - F(a)
    tol = 1E-3

    methods = [Trapezoidal, Simpson, GaussLegendre]
    for method in methods:
        integrator = method(a, b, n)
        I = integrator.integrate(f)
        rel_err = abs((I_exact - I) / I_exact)
        print('%s: %g' % (method.__name__, rel_err))
        if rel_err > tol:
            print('Error in %s' % method.__name__)

if __name__ == '__main__':
    test()
##### end of Listing 5.9 ##### 

