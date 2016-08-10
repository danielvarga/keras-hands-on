import numpy as np
import theano
import theano.tensor as T

x = T.fscalar('x')
y = T.fscalar('y')
f = x
for i in range(10):
    f = f * y
f_fn = theano.function([x, y], f)

print theano.printing.debugprint(f)

print "f(2, 3) = ", f_fn(2, 3)
