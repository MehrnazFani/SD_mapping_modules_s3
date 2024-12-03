from .linknet import LinkNet34, LinkNet34MTL
from .stack_module import StackHourglassNetMTL
from .stack_module_multi import StackHourglassNetMTL_Multi



MODELS = {"StackHourglassNetMTL": StackHourglassNetMTL}

MODELS_REFINE = {"LinkNet34": LinkNet34}
