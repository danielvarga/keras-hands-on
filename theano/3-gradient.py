import numpy as np
import theano
import theano.tensor as T

x = T.fscalar('x')
y = T.fscalar('y')
f = x**5 + y**2 + x*y
f_fn = theano.function([x, y], f)

print "f(x, y) =", theano.printing.pprint(f)

print "f(2, 3) =", f_fn(2, 3)

dfdx = T.grad(f, [x])

# print "dfdx(x, y) =", theano.printing.pprint(dfdx)

print "df/dx (2, 3) =", theano.function([x, y], dfdx)(2, 3)
