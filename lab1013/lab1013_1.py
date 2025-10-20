price=1000000
year=30

def p(ratio):
    pay=(price*ratio/12)/(1-(1+ratio/12)**(-12*year))
    sum=pay*12*year
    return (round(pay, 2), round(sum, 2))

print(p(0.04))
print(p(0.05))
print(p(0.06))