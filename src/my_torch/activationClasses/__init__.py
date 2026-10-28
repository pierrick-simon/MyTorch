from .leaky_relu import LeakyReLU
from .linear import Linear
from .relu import ReLU
from .sigmoid import Sigmoid
from .tanh import Tanh

ActivationClass = type[Sigmoid | Tanh | ReLU | LeakyReLU | Linear]

ActivationClassNames = {
    "Sigmoid": Sigmoid,
    "Tanh": Tanh,
    "ReLU": ReLU,
    "LeakyReLU": LeakyReLU,
    "Linear": Linear,
}