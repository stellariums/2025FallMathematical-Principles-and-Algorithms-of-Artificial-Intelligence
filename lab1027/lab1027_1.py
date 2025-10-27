##### start of Listing 4.22 ##### 
"""
Module for performing arithmetic operations for rational numbers.

To run the module, user needs to supply three named parameters:
1. op stands for the operation:
    add for addition
    sub for subtraction
    mul for multiplication
    div for division
2. x stands for the first operand
3. y stands for the second operand

x and y must be enclosed in paired parentheses.

For example:

>>> run rational.py --op add --x (2/3) --y (-70/40)
-13/12
>>> run rational.py --op sub --x (-20/3) --y (120/470)
-976/141
>>> run rational.py --op mul --x (-6/19) --y (114/-18)
2/1
>>> run rational.py --op div --x (-6/19) --y (-114/-28)
-28/361
"""

import sys, math

def test_all_functions():  
    pass
    

def gcd(a, b):  
    a,b=abs(a),abs(b)
    while a != b:
        if a > b:
            a -= b
        else:
            b -= a
    return a

def reduce(n, d):  
    if d < 0:
        n,d=-n,-d
    maxfac=gcd(n,d)
    return n//maxfac,d//maxfac

def add(x, y):  
    new=[]
    new.append(x[0]*y[1]+y[0]*x[1])
    new.append(x[1]*y[1])
    new[0],new[1]=reduce(new[0],new[1])
    return new

def sub(x, y):  
    new=[]
    new.append(x[0]*y[1]-y[0]*x[1])
    new.append(x[1]*y[1])
    new[0],new[1]=reduce(new[0],new[1])
    return new

def mul(x, y):  
    new=[]
    new.append(x[0]*y[0])
    new.append(x[1]*y[1])
    new[0],new[1]=reduce(new[0],new[1])
    return new

def div(x, y):  
    new=[]
    new.append(x[0]*y[1])
    new.append(x[1]*y[0])
    new[0],new[1]=reduce(new[0],new[1])
    return new

def output(x):  
    print(f'{x[0]}/{x[1]}')

def get_rational(s):  
    s = s.strip().strip('()')
    parts = s.split('/')
    n = int(parts[0])
    d = int(parts[1])
    return [n,d]

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(__doc__)
    elif len(sys.argv) == 2 and sys.argv[1] == '-h':
        print(__doc__)
    elif len(sys.argv) == 2 and sys.argv[1] == 'test':
        test_all_functions()
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--op', type=str)
        parser.add_argument('--x', type=str)
        parser.add_argument('--y', type=str)
        args = parser.parse_args()
        op = args.op
        x = get_rational(args.x); y = get_rational(args.y)
        f = {'add':add, 'sub':sub, 'mul':mul, 'div':div}
        output(f[op](x, y))
##### end of Listing 4.22 ##### 