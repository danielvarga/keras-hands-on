virtualenv virtualenv --python=$(which python)
. virtualenv/bin/activate

# Get the latest pip for sure
pip install --upgrade pip

# Install backend
BACKEND="$1"; shift
if [ "--tensorflow" == "$BACKEND" ]; then
    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0rc0-py2-none-any.whl
    pip install --upgrade $TF_BINARY_URL
else
    pip install Theano
fi

# Install Keras
pip install -r requirements.txt
