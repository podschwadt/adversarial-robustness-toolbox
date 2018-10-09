# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys
from keras.utils import Progbar

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Attack(ABC):
    """
    Abstract base class for all attack classes.
    """
    attack_params = ['classifier']

    def __init__(self, classifier):
        """
        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        """
        self.classifier = classifier

    def generate(self, x, **kwargs):
        """
        Generate adversarial examples and return them as an array. This method should be overridden by all concrete
        attack implementations.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param kwargs: Attack-specific parameters used by child classes.
        :type kwargs: `dict`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def start_progress_bar(self, target, width=30, verbose=1, interval=0.05, stateful_metrics=None):
        """
        Creates the progress bar that can be used during generating adversarial examples.


        :param target: Total number of steps expected, None if unknown.
        :type x: `int`
        :param width: Progress bar width on screen.
        :type width: `int`
        :param verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        :type width: `int`
        :param interval: Minimum visual progress update interval (in seconds).
        :type interval: `float`
        :type width: `int`
        :param stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        :param stateful_metrics: `list[str]`
        """
        self.prog_bar = Progbar(target, width, verbose, interval, stateful_metrics)

    def update_progress_bar(self, current, values=None):
        """
        Updates the progress bar.

        :param current: Index of current step.
        :type current: `int`
        :param values: List of tuples:
                    `(name, value_for_last_step)`.
                    If `name` is in `stateful_metrics`,
                    `value_for_last_step` will be displayed as-is.
                    Else, an average of the metric over time will be displayed.
        :type values: `list`
        """
        self.prog_bar.update(current, values)

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: a dictionary of attack-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        return True
