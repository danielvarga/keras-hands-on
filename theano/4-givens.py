import numpy as np

import theano
import theano.tensor as T

inDim = 3

initialsVar = theano.shared(np.ones(inDim, dtype=np.float32), "initials")
parametersVar = theano.shared(np.ones(inDim, dtype=np.float32), "parameters")
freeAgainVar = T.vector('free')

loss = T.dot(initialsVar, parametersVar)

loss_fn = theano.function([], loss)
print loss_fn()

loss_1arg_fn = theano.function([freeAgainVar], loss, givens=[(initialsVar, freeAgainVar)])
print loss_1arg_fn( np.ones(inDim, dtype=np.float32)*3 )
