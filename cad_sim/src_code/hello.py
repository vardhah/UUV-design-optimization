import os
print('in hello.py')
print('pwd is:',os.getcwd())
if __name__== '__main__':
    with open('./data/string.txt','r') as f:
         name= f.read()
    print("Hello",name)

