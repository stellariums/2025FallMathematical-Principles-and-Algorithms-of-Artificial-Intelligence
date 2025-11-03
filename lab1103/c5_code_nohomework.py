##### start of Listing 5.1 ##### 
import math
class Point2D:
    def __init__(self, x, y, name=''):
        self.x = x  
        self.y = y  
        self.name = name  

    def move_x(self, delta_x):
        self.x += delta_x

    def move_y(self, delta_y): 
        self.y += delta_y

    def rotate(self, p, t): 
        xr = self.x - p.x; yr = self.y - p.y
        x1 = p.x + xr * math.cos(t) - yr * math.sin(t)
        y1 = p.y + xr * math.sin(t) + yr * math.cos(t)
        self.x = x1; self.y = y1;

    def distance(self, p): 
        xr = self.x - p.x; yr = self.y - p.y
        return math.sqrt(xr * xr + yr * yr)

    def __str__(self): 
        if len(self.name) < 1:
            return '(%g, %g)' % (self.x, self.y)
        else:
            return '%s: (%g, %g)' % (self.name, self.x, self.y)

a = Point2D(-5, 2, 'a')
print(a)  
a.move_x(-1); print(a)             
a.move_y(2); print(a)              
b = Point2D(3, 4, 'b')
print(b)  
print('The distance between a and b is %f' % a.distance(b))
b.rotate(a, math.pi/2)
print(a); print(b)     
a.rotate(b, math.pi)
print(a); print(b)     
##### end of Listing 5.1 ##### 

##### start of Listing 5.4 ##### 
tol = 1E-15
class Polynomial:
    def __init__(self, poly):
        self.poly = {}
        for power in poly:
            if abs(poly[power]) > tol:
                self.poly[power] = poly[power]

    def __call__(self, x):
        value = 0.0
        for power in self.poly:
            value += self.poly[power]*x**power
        return value

    def __add__(self, other): 
        sum = self.poly.copy()   
        for power in other.poly:
            if power in sum:
                sum[power] += other.poly[power]
            else:
                sum[power] = other.poly[power]
        return Polynomial(sum) 

    def __mul__(self, other):
        sum = {}
        for self_power in self.poly:
            for other_power in other.poly:
                power = self_power + other_power
                m = self.poly[self_power] * \
                    other.poly[other_power]
                if power in sum:
                    sum[power] += m
                else:
                    sum[power] = m
        return Polynomial(sum)  

    def __str__(self):
        s = ''
        for power in sorted(self.poly):
            s += ' + %g*x^%d' % (self.poly[power], power)
        s = s.replace('+ -', '- ')
        s = s.replace('x^0', '1')
        s = s.replace(' 1*', ' ')
        s = s.replace('x^1 ', 'x ')
        if s[0:3] == ' + ':  
            s = s[3:]
        if s[0:3] == ' - ':  
            s = '-' + s[3:]
        return s

p1 = Polynomial({0: -1, 2: 1, 7: 3}); print(p1)
p2 = Polynomial({0: 1, 2: -1, 5: -2, 3: 4}); print(p2)
p3 = p1 + p2; print(p3)
p4 = p1 * p2; print(p4)
print(p4(5))   
##### end of Listing 5.4 ##### 

##### start of Listing 5.5 ##### 
class Differentiation:
    def __init__(self, f, h=1E-5, dfdx_exact=None):
        self.f = f
        self.h = float(h)
        self.exact = dfdx_exact

    def get_error(self, x):  
        if self.exact is not None:
            df_numerical = self(x) 
            df_exact = self.exact(x) 
            return abs( (df_exact - df_numerical) / df_exact )

class Forward1(Differentiation):  
    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x+h) - f(x))/h

class Backward1(Differentiation):  
    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x) - f(x-h))/h

class Central2(Differentiation):  
    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x+h) - f(x-h))/(2*h)

class Central4(Differentiation):  
    def __call__(self, x):
        f, h = self.f, self.h
        return (4./3)*(f(x+h) - f(x-h))  /(2*h) - \
               (1./3)*(f(x+2*h) - f(x-2*h))/(4*h)

def table(f, x, h_values, methods, dfdx=None):
    print('%-10s' % 'h', end=' ')
    for h in h_values: print('%-8.2e' % h, end=' ')
    print()
    for method in methods:
        print('%-10s' % method.__name__, end=' ')
        for h in h_values:
            if dfdx is not None:
                d = method(f, h, dfdx)
                output = d.get_error(x)
            else:
                d = method(f, h)
                output = d(x)
            print('%-8.6f' % output, end=' ')
        print()

import math
def g(x): return math.exp(x*math.sin(x))

import sympy as sym
sym_x = sym.Symbol('x')  
sym_gx = sym.exp(sym_x*sym.sin(sym_x)) 
sym_dgdx = sym.diff(sym_gx, sym_x) 
dgdx = sym.lambdify([sym_x], sym_dgdx)

table(f=g, x=-0.65, h_values=[10**(-k) for k in range(1, 7)],
      methods=[Forward1, Central2, Central4], dfdx=dgdx)
##### end of Listing 5.5 ##### 

##### start of Listing 5.6 ##### 
# h          1.00e-01 1.00e-02 1.00e-03 1.00e-04 1.00e-05 1.00e-06
# Forward1   0.104974 0.010906 0.001095 0.000110 0.000011 0.000001
# Central2   0.004611 0.000046 0.000000 0.000000 0.000000 0.000000
# Central4   0.000080 0.000000 0.000000 0.000000 0.000000 0.000000
##### end of Listing 5.6 ##### 

##### start of Listing 5.7 ##### 
class Parent:
    def __init__(self):
        self.c = 1
    def m(self):
        print('Calling m in class Parent')
	
class Child1(Parent):
    def __init__(self):  
        super().__init__()  
        self.d = 2
    def m(self):  
        print('Calling m in class Child1')
	
class Child2(Parent):
    def m(self):  
        super().m()  
        print('Calling m in class Child2')

class Child3(Parent):
    def __init__(self):  
        self.f = 3
    def m2(self):
        print('Calling m2 in class Child3')	
	
c1 = Child1()           
c2 = Child2()           
c3 = Child3()           
p = Parent()            
c1.m()                  
c2.m()                  
c3.m()                  
p.m()                   
print(c1.__dict__)      
print(c2.__dict__)      
print(c3.__dict__)      
##### end of Listing 5.7 ##### 
