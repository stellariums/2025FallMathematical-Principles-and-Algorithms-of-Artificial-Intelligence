##### start of Listing 5.8 ##### 
def gcd(a, b):
    a, b = abs(a), abs(b)
    while b != 0:
        a, b = b, a % b
    return a


class Rational:
    def __init__(self, n=0, d=1):  
        _nu = n; _de = d
        self.__dict__['nu'] = _nu; self.__dict__['de'] = _de

    def __setattr__(self, name, value):
        raise TypeError('Error: Rational objects are immutable')

    def __str__(self): return '%d/%d' % (self.nu, self.de)

    def __add__(self, other):  
        new_nu = self.nu * other.de + self.de * other.nu
        new_de = self.de * other.de
        new_gcd = gcd(new_nu, new_de)
        return Rational(new_nu // new_gcd, new_de // new_gcd)

    def __sub__(self, other):  
        new_nu = self.nu * other.de - self.de * other.nu
        new_de = self.de * other.de
        new_gcd = gcd(new_nu, new_de)
        return Rational(new_nu // new_gcd, new_de // new_gcd)

    def __mul__(self, other):  
        new_nu = self.nu * other.nu
        new_de = self.de * other.de
        return Rational(new_nu, new_de)
    def __truediv__(self, other):  
        if other.nu == 0:
            raise ZeroDivisionError("Division by zero")
        new_nu = self.nu * other.de
        new_de = self.de * other.nu
        return Rational(new_nu, new_de)
    def __eq__(self, other):  
        return self.nu * other.de == self.de * other.nu
    def __ne__(self, other):  
        return not self.__eq__(other)
    def __gt__(self, other):  
        return self.nu * other.de > self.de * other.nu
    def __lt__(self, other):  
        return self.nu * other.de < self.de * other.nu
    def __ge__(self, other):  
        return self.nu * other.de >= self.de * other.nu
    def __le__(self, other):  
        return self.nu * other.de <= self.de * other.nu
def test():
    testsuite = [
        ('Rational(2, 3) + Rational(-70, 40)',
          Rational(-13, 12)),
        ('Rational(-20, 3) - Rational(120, 470)',
          Rational(-976,141)),
        ('Rational(-6, 19) * Rational(-114, 18)',
          Rational(2, 1)),
        ('Rational(-6, 19) / Rational(-114, -28)',
          Rational(-28,361)),

        ('Rational(-6, 19) == Rational(-14, 41)', False),
        ('Rational(-6, 19) != Rational(-14, 41)', True),
        ('Rational(6, -19) > Rational(14, -41)', True),
        ('Rational(-6, 19) < Rational(-14, 41)', False),
        ('Rational(-6, 19) >= Rational(-14, 41)', True),
        ('Rational(6, -19) <= Rational(14, -41)', False),
        ('Rational(-15, 8) == Rational(120, -64)', True),
    ]
    for t in testsuite:
        try:
            result = eval(t[0])
        except:
            print('Error in evaluating ' + t[0]); continue

        if result != t[1]:
            print('Error:  %s != %s' % (t[0], t[1]))

if __name__ == '__main__':
    test()
    
##### end of Listing 5.8 ##### 