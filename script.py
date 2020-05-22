file = open('train.py')
for line in file:
    if 'Sequential' in line:
        print ('Sequential')
    elif 'Regression' in line:
        print ('Regression')


