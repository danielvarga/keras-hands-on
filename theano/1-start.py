import numpy as np
import theano
import theano.tensor as T

x = T.fscalar('x')
y = T.fscalar('y')
f = x**10 + y
f_fn = theano.function([x, y], f)

print "f(x, y) =", theano.printing.pprint(f)

print "f(2, 3) = ", f_fn(2, 3)

print "=" * 30

X = T.fmatrix('X')
y = T.fvector('y')
f = X.dot(y)
f_fn = theano.function([X, y], f)
X0 = np.array(range(90), dtype=np.float32).reshape(9, 10)
y0 = np.array(range(10), dtype=np.float32)

print "f(X, y) =", theano.printing.pprint(f)
print "X ="
print X0
print "y ="
print y0
print "Xy ="
print f_fn(X0, y0)
