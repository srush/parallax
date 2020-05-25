import itertools
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Union, Tuple

from frozendict import frozendict
import jax


class ParamInit:
    initializer : Any  # function (rng, shape) |-> tensor
    shape : Tuple[int]

    def __init__(self, shape, initializer):
        self.shape = shape
        self.initializer = initializer

    def instantiate(self, rng):
        """Returns a tensor created according to this init."""
        return self.initializer(key=rng, shape=self.shape)

    def __repr__(self):
        return "ParamInit(" + ", ".join([str(d) for d in self.shape]) + ")"


Parameter = Union[ParamInit, jax.interpreters.xla.DeviceArray]


@jax.tree_util.register_pytree_node_class
@dataclass
class ParameterTuple:
    parameters : Tuple[Parameter]

    def __init__(self, parameters):
        self.parameters = tuple(parameters)

    def instantiate(self, rng):
        rngs = jax.random.split(rng, len(self.parameters))
        return ParameterTuple(p.instantiate(rng) for p, rng in zip(self.parameters, rngs))

    def __repr__(self):
        return "ParameterTuple(" + ", ".join([repr(p) for p in self.parameters]) + ")"

    def __iter__(self):
        return self.parameters.__iter__()

    def tree_flatten(self):
        aux = [self.__class__]
        leaves = []
        for p in self.parameters:
            if isinstance(p, ParameterTuple):
                l, a = p.tree_flatten()
                leaves += l
                aux.append((len(l), a))
            else:
                leaves.append(p)
                aux.append(None)           
        return leaves, aux 

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        parameters = []
        i = 0
        for p in aux[1:]:
            if p is None:
                parameters.append(leaves[i])
                i += 1
            else:
                nleaves, a = p
                parameters.append(
                    cls.tree_unflatten(a, leaves[i:i+nleaves])
                )
                i += nleaves
        assert i == len(leaves)
        return cls(parameters)


def _recursive_all_annotations(cls):
    d = frozendict()
    for c in cls.__mro__[::-1]:
        if "__annotations__" in c.__dict__:
            d = d.copy(**c.__annotations__)
    return d


@jax.tree_util.register_pytree_node_class
class Module:
    _is_initialized: bool = False
    mode : str
    rng : jax.interpreters.xla.DeviceArray
    _parameters : Tuple[Union[Parameter, ParameterTuple]]
    _modules : Tuple[Union["Module", "ModuleTuple"]]  # apparently that's the best we can do for recursive types :(
    _constants : Tuple[Any]

    ModField = namedtuple("ModField", ["name", "type"])

    def __init__(self):
        self._is_initialized = False
        self.mode = "train"
        self.rng = None
        self._register_fields()

    def __setattr__(self, name, value):
        if self._is_initialized:
            raise Exception(f"Can't set {name}, class is already initialized!")
        elif name not in _recursive_all_annotations(self.__class__).keys():
            raise Exception(f"Field {name} was not declared in {self.__class__} or ancestors!")
        else:
            self.__dict__[name] = value

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node_class(cls)
            
    @classmethod
    def _user_fields(cls):
        return sorted([
            cls.ModField(k, v)
            for k, v in _recursive_all_annotations(cls).items()
            if k not in ["_is_initialized", "mode", "rng", "_parameters",
                         "_modules", "_constants"]
        ], key=lambda f: f.name)

    @classmethod
    def _new_from(cls, **kwargs):
        obj = cls.__new__(cls)
        for k, v in kwargs.items():
            obj.__dict__[k] = v
        obj._register_fields()
        return obj

    def _updated_with(self, **kwargs):
        obj = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.items():
            if k in kwargs:
                obj.__dict__[k] = kwargs[k]
            else:
                obj.__dict__[k] = v
        return obj

    def _register_fields(self):
        super().__setattr__('_modules',
                            tuple([f.name
                                   for f in self._user_fields()
                                   if not f.type == Parameter and
                                   not issubclass(f.type, ParameterTuple) and
                                   (issubclass(f.type, ModuleTuple) or
                                    issubclass(f.type, Module))]))
        super().__setattr__('_parameters',
                            tuple([f.name
                                   for f in self._user_fields()
                                   if f.type == Parameter or
                                   issubclass(f.type, ParameterTuple)]))
        super().__setattr__('_constants',
                            tuple([f.name
                                   for f in self._user_fields()
                                   if not f.type == Parameter and
                                   not issubclass(f.type, ParameterTuple) and
                                   not issubclass(f.type, ModuleTuple) and
                                   not issubclass(f.type, Module)
                            ]))

    def _all_constants(self):
        d = frozendict({"_is_initialized" : self._is_initialized,
                        "mode" : self.mode,
                        "rng" : self.rng})
        for c in self._constants:
            d = d.copy(**{c : self.__dict__[c]})
        return d

    def split(self, num_splits):
        rngs = jax.random.split(self.rng, num_splits)
        return [self._updated_with(rng=rng) for rng in rngs]

    def initialized(self, rng):
        d = self._all_constants().copy(_is_initialized = True)
        rng_p, rng_m = jax.random.split(rng)
        rngs = jax.random.split(rng_p, len(self._parameters))
        for p, rng in zip(self._parameters, rngs):
            assert isinstance(self.__dict__[p], ParamInit) or \
                   isinstance(self.__dict__[p], ParameterTuple)
            d = d.copy(**{p : self.__dict__[p].instantiate(rng)})

        rngs = jax.random.split(rng_m, len(self._modules))
        for m, rng in zip(self._modules, rngs):
            if isinstance(self.__dict__[m], Module):
                assert not self.__dict__[m]._is_initialized
            d = d.copy(**{m: self.__dict__[m].initialized(rng)})

        return self.__class__._new_from(**d)

    def new_state(self, rng, mode=None):
        d = frozendict({"rng": rng, "mode": mode or self.mode})
        rngs = jax.random.split(rng, len(self._modules))
        for m, rng in zip(self._modules, rngs):
            d = d.copy(**{m: self.__dict__[m].new_state(rng, mode)})
        return self._updated_with(**d)

    def __call__(self, *args):
        return self.forward(*args)

    def grad(self, input):
        return jax.grad(self.__class__.forward)(self, input)

    def tree_flatten(self):
        flat_module_names = tuple(self._modules)
        flat_modules = [self.__dict__[m].tree_flatten()
                        for m in flat_module_names]
        flat_parameter_names = tuple(self._parameters)
        flat_parameters = [self.__dict__[p].tree_flatten()
                           if isinstance(self.__dict__[p], ParameterTuple)
                           else ([self.__dict__[p]], None)
                           for p in flat_parameter_names]
        leaves = tuple(itertools.chain(
            *[leaves for (leaves, _) in flat_modules + flat_parameters],
        ))
        aux = (
            self.__class__,
            self._all_constants(),
            [
                (name, len(leaves), aux)
                for name, (leaves, aux) in zip(
                    flat_module_names + flat_parameter_names,
                    flat_modules + flat_parameters
                )
            ],
        )
        return (leaves, aux)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        _cls, d, aux_fields = aux
        assert cls == _cls
        i = 0
        add_d = {}
        for name, n_leaves, aux in aux_fields:
            if aux is None:
                assert n_leaves == 1
                add_d[name] = leaves[i]
            else:
                add_d[name] = aux[0].tree_unflatten(aux, leaves[i:i+n_leaves])
            i += n_leaves
        assert i == len(leaves)
        d = d.copy(**add_d)
        return cls._new_from(**d)

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

        main_str = self._get_name() + (" uninit " if not self._is_initialized else "") + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str


@jax.tree_util.register_pytree_node_class
@dataclass
class ModuleTuple:
    modules : Tuple[Module]

    def __init__(self, modules):
        self.modules = tuple(modules)

    def initialized(self, rng):
        rngs = jax.random.split(rng, len(self.modules))
        return ModuleTuple(m.initialized(rng) for m, rng in zip(self.modules, rngs))

    def new_state(self, rng, mode=None):
        rngs = jax.random.split(rng, len(self.modules))
        return ModuleTuple(
            m.new_state(rng, mode)
            for m, rng in zip(self.modules, rngs)
        )

    def __repr__(self):
        return "ModuleTuple(" + ", ".join([repr(m) for m in self.modules]) + ")"

    def __iter__(self):
        return self.modules.__iter__()

    def tree_flatten(self):
        aux = [self.__class__]
        leaves = []
        for m in self.modules:
            l, a = m.tree_flatten()
            leaves += l
            aux.append((m.__class__, len(l), a))         
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        modules = []
        i = 0
        for m in aux[1:]:
            child_cls, nleaves, a = m
            modules.append(
                child_cls.tree_unflatten(a, leaves[i:i+nleaves])
            )
            i += nleaves
        assert i == len(leaves)
        return cls(modules)
