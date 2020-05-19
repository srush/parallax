import torch
from dataclasses import dataclass, fields, field, replace
from frozendict import frozendict
from typing import Tuple, FrozenSet
module = dataclass(frozen=True, repr=False)

def random_split(seed, n_splits):
    "Split the random seed in n_parts (fake jax.random.split)"
    return [seed] * n_splits

@module
class Parameter:
    init : any
    data : torch.tensor
    _initialized: bool

    @staticmethod
    def setup(shape, init):
        return Parameter(init=init, data=shape,
                         _initialized=False)

    def create(self, rng):
        torch.random.set_rng_state(rng)

        # This would be pure.
        data = torch.zeros(self.data, requires_grad=True)
        self.init(data)
        #
        return data

    @property
    def shape(self):
        if self._initialized:
            return self.data.shape
        else:
            return self.data

module = dataclass(frozen=True, repr=False)

@module
class Module:
    _initialized: bool
    mode : str
    rng : float
    _parameters : FrozenSet = field(init=False)
    _modules : FrozenSet = field(init=False)
    _constants : FrozenSet = field(init=False)

    @classmethod
    def _user_fields(cls):
        return [f for f in fields(cls)
                if f.name not in ["_initialized", "mode", "rng",
                                  "_parameters", "_modules", "_constants"] ]

    @classmethod
    def init(cls, **kwargs):
        return cls(**kwargs, _initialized=False, mode="train", rng=None,)

    @classmethod
    def setup(cls, **kwargs):
        return cls.init(**kwargs)


    def split(self, num_splits):
        rngs = random_split(self.rng, num_splits)
        return [replace(self, rng=p_rng) for p_rng in rngs]

    def __post_init__(self):
        super().__setattr__('_modules',
                            frozenset([f.name
                                       for f in self._user_fields()
                                       if issubclass(f.type, Module)]))
        super().__setattr__('_parameters',
                            frozenset([f.name
                                       for f in self._user_fields()
                                       if issubclass(f.type, Parameter)]))
        super().__setattr__('_constants',
                            frozenset([f.name
                                       for f in self._user_fields()
                                       if not issubclass(f.type, Parameter) and
                                       not issubclass(f.type, Module)
                            ]))


    def __getattr__(self, name):
        if name in self._parameters:
            return self.__dict__[name].data
        return self.__dict__[name]

    def _base(self):
        d = frozendict({"_initialized" : self._initialized,
                        "mode" : self.mode,
                        "rng" : self.rng})
        for c in self._constants:
            d = d.copy(**{c : self.__dict__[c]})
        return d

    def initialize(self, rng):
        d  = self._base()
        d = d.copy(_initialized = True)

        splits = random_split(rng, len(self._parameters))
        for p, p_rng in zip(self._parameters, splits):
            d = d.copy(**{p : self.__dict__[p].create(p_rng)})

        splits = random_split(rng, len(self._modules))
        for m, p_rng in zip(self._modules, splits):
            d = d.copy(**{m:self.__dict__[m].initialize(p_rng)})

        return self.__class__(**d)

    def init_state(self, rng, mode=None):
        d  = self._base()
        d = d.copy(rng = rng,
                   mode = mode if mode is not None else self.mode)

        for p in self._parameters:
            q = self.__dict__[p]

            # Update
            if q.grad is not None:
                q.grad.zero_()
            #
            d = d.copy(**{p: q})
        splits = random_split(rng, len(self._modules))
        for m, p_rng in zip(self._modules, splits):
            d = d.copy(**{m: self.__dict__[m].init_state(p_rng, mode)})
        return self.__class__(**d)

    def grad(self):
        d  = self._base()
        d.copy(rng = None)
        for p in self._parameters:
            d = d.copy(**{p: self.__dict__[p].grad})

        for m in self._modules:
            d = d.copy(**{m: self.__dict__[m].grad()})
        return self.__class__(**d)

    def update(self, fn, other):
        d  = self._base()
        d.copy(rng = None)
        for p in self._parameters:
            d = d.copy(**{p: fn(self.__dict__[p], other.__dict__[p])})

        for m in self._modules:
            d = d.copy(**{m: self.__dict__[m].update(fn, other.__dict__[m])})

        return self.__class__(**d)

    def __call__(self, *args):
        return self.forward(*args)

    def modules(self):
        for f in self._modules:
            yield f, self.__dict__[f]

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self):
        return ""

    def __repr__(self):
        "Mimic the pytorch pretty printer"
        def _addindent(s_, numSpaces):
            s = s_.split('\n')
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []

        for key, module in self.modules():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + (" uninit " if not self._initialized else "") + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str
