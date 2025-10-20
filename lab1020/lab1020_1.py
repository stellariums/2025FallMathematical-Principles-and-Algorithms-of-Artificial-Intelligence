num=[]
def klz(n):
    num.append(n)
    while n!=1:
        if n%2==0:
            n=n//2
            num.append(n)
        else:
            n=n*3+1
            num.append(n)
    return num

inputnum=int(input("enter your num:"))
numlist=klz(inputnum)

for i in range(len(num)):
    if i!=len(num):
        print(num[i],end=" ")
    else:
        print(num[i],end="")