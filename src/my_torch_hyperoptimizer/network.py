import yaml
from typeguard import typechecked

from src.my_torch.activationClasses import ActivationClassNames
from src.my_torch.MLP import MLP
from src.my_torch_hyperoptimizer.constants import NB_INPUTS, NB_OUTPUTS, PATIENCE, VALIDATION_INTERVAL

@typechecked
def buildNetwork(config: dict, seed: int) -> MLP:
    activations = (ActivationClassNames[config["hiddenActivationFunction"]], ActivationClassNames[config["outputActivationFunction"]])
    return MLP(NB_INPUTS, config["hiderLayersLayout"], NB_OUTPUTS, config["learningRate"], activations, seed,
        config["momentum"], config["initialBias"], batchSize=config["batchSize"])

@typechecked
def writeYml(config: dict, seed: int, path: str) -> None:
    content = {"nbInputs": NB_INPUTS, "nbOutputs": NB_OUTPUTS, **config, "seed": seed,
               "patience": PATIENCE, "validationInterval": VALIDATION_INTERVAL}
    with open(path, "w") as f:
        yaml.safe_dump(content, f, sort_keys=False, default_flow_style=None)
