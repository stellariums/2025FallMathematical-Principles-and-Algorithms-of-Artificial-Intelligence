import math

a=3
b=6
c=7

x=math.degrees(math.acos((a**2-b**2-c**2)/(-2*c*b)))
y=math.degrees(math.acos((b**2-a**2-c**2)/(-2*c*a)))
z=math.degrees(math.acos((c**2-b**2-a**2)/(-2*a*b)))

print(x,y,z)
print(x+y+z)