nums={25,18,91,365,12,78,59}
multiplier_of_3=[]
square_of_odds=[]
for n in nums:
    if n%3==0:
        multiplier_of_3.append(n)
    if n%2==1:
        square_of_odds.append(n*n)
print(multiplier_of_3)
print(set(square_of_odds))


s = [25, 18, 91, 365, 12, 78, 59, 18, 91]

sr = {}
for n in set(s):
    sr[n] = n % 3
print(sr)

tr = {}
for n, r in sr.items():
    if r == 0:
        tr[n] = r
print(tr)
