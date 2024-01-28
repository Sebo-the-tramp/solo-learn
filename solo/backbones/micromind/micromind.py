# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Authors:
#     - Sebastian Cavada, 2024

# Extending the solo-learn repo with micromind backbone

from micromind.networks import PhiNet, XiNet

from timm.models.registry import register_model

@register_model
def phinet(**kwargs):

    # find a way to import the args for phinet hier

    model_args = {
        'input_shape': (3, 224, 224),
        'alpha': 0.7,
        'num_layers': 7,
        'beta': 2,
        't_zero': 1,
        'divisor': 8,
        'downsampling_layers': [],
        'compatibility': False,  # Added to match the PhiNet constructor        
        # 'return_layers': None,   # Uncomment if needed
        # 'num_classes': None,     # Uncomment and provide a value if needed
    }

    model = PhiNet(
                input_shape=model_args["input_shape"],
                alpha=model_args["alpha"],
                num_layers=model_args["num_layers"],
                beta=model_args["beta"],
                t_zero=model_args["t_zero"],
                compatibility=False,
                divisor=model_args["divisor"],
                downsampling_layers=model_args["downsampling_layers"],
                #return_layers=hparams.return_layers,
                # classification-specific
                include_top=True,
                #num_classes=hparams.num_classes,
            )
    return model

@register_model
def xinet(**kwargs):    
    # find a way to import the args for phinet hier

    model_args = dict(input_shape=(224,224), alpha=0.7, gamma=0.3, num_layers=7, beta=2, t_zero=1, divisor=8, downsampling_layers=[], num_classes=10, return_layers=None)
    model = XiNet(
                input_shape=model_args.input_shape,
                alpha=model_args.alpha,
                gamma=model_args.gamma,
                num_layers=model_args.num_layers,
                return_layers=model_args.return_layers,
                # classification-specific
                include_top=True,
                num_classes=model_args.num_classes,
            )
    return model