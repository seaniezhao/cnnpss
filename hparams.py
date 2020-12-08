import json

def create_harmonic_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = dict(
        type=0,
        layers=3,
        blocks=2,
        dilation_channels=130,
        residual_channels=130,
        skip_channels=240,
        input_channel=60,
        condition_channel=621,
        output_channel=240,
        sample_channel=60,
        initial_kernel=10,
        kernel_size=2,
        bias=True
    )

    return dict2obj(hparams)


def create_aperiodic_hparams( hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = dict(
        type=1,
        layers=3,
        blocks=2,
        dilation_channels=20,
        residual_channels=20,
        skip_channels=16,
        input_channel=64,
        condition_channel=621,
        output_channel=16,
        sample_channel=4,
        initial_kernel=10,
        kernel_size=2,
        bias=True
    )

    return dict2obj(hparams)


def create_vuv_hparams( hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = dict(
        type=2,
        layers=3,
        blocks=2,
        dilation_channels=20,
        residual_channels=20,
        skip_channels=4,
        input_channel=65,
        condition_channel=621,
        output_channel=1,
        sample_channel=1,
        initial_kernel=10,
        kernel_size=2,
        bias=True
    )

    return dict2obj(hparams)


def create_f0_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = dict(
        type=3,
        layers=7,
        blocks=2,
        dilation_channels=100,
        residual_channels=100,
        skip_channels=100,
        input_channel=1,
        condition_channel=708,
        output_channel=4,
        sample_channel=1,
        initial_kernel=20,
        kernel_size=2,
        bias=True
    )

    return dict2obj(hparams)


# declaringa a class
class obj:

    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=obj)
