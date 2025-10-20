def secret(n,inputstr):
    num=len(inputstr)
    change=[chr(ord(i)+n) for i in inputstr]
    return change

inputstr=str(input("enter your string:"))
n=int(input("enter your n:"))
newstr=secret(n,inputstr)

print('\"',end='')
for i in newstr:
    print(i,end='')
print("\"",end='')