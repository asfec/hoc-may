from cvxopt import matrix, solvers
c = matrix([5.,10.,15.,4.])
G = matrix([[1.,0.,-1.,0.,0.,0.],
           [1.,0.,0.,-1.,0.,0.],
           [0.,1.,0.,0.,-1.,0.],
           [0.,1.,0.,0.,0.,-1.]])
h = matrix([800.,700,0.,0.,0.,0.])
A = matrix([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
b= matrix([600.,400.])
solvers.options['show_progress'] = False
sol = solvers.lp(c, G, h,A ,b)

print('Solution"')
print(sol['x'])