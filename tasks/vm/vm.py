"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import types
import typing as tp

CO_VARARGS = 4
CO_VARKEYWORDS = 8

ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
ERR_MISSING_POS_ARGS = 'Missing positional arguments'
ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
ERR_POSONLY_PASSED_AS_KW = ('Positional-only argument '
                            'passed as keyword argument')


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.11/Include/frameobject.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """
    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.cur_offset = 0
        self.return_value = None

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        dis_dict = {instruction.offset: instruction for instruction
                    in dis.get_instructions(self.code)}
        max_offset = max(dis_dict.keys())

        while self.cur_offset <= max_offset:
            pred_offset = self.cur_offset
            cur_instruction = dis_dict[self.cur_offset]
            if cur_instruction:
                opname = cur_instruction.opname.lower()

                if opname.startswith('binary_subscr'):
                    getattr(self, 'binary_op_op')('subscr')
                elif opname.startswith('binary_'):
                    getattr(self, 'binary_op_op')(cur_instruction.argrepr)
                elif opname.startswith('format_value'):
                    getattr(self, 'format_value_op')(cur_instruction.arg)
                elif opname.startswith('load_global'):
                    getattr(self, 'load_global_op')(cur_instruction.argval,
                                                    cur_instruction.arg)
                else:
                    getattr(self, cur_instruction.opname.lower()
                            + "_op")(cur_instruction.argval)

            if pred_offset == self.cur_offset:
                self.cur_offset += 2

        return self.return_value

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def push_null_op(self, arg: int) -> tp.Any:
        self.push(None)

    def precall_op(self, arg: int) -> tp.Any:
        self.cur_offset += 4

    def call_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-CALL
        """
        arguments = self.popn(arg)
        func_or_self = self.pop()
        f = self.pop()
        if f is None:
            self.push(func_or_self(*arguments))
        else:
            self.push(f(*arguments))
        self.cur_offset += 10

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_NAME
        """
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError

    def load_global_op(self, arg: str, namei: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_GLOBAL
        """
        if namei & 1:
            self.push(None)
        if arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError
        self.cur_offset += 12

    def load_fast_op(self, arg: str) -> None:
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise UnboundLocalError

    def binary_op_op(self, op: str) -> None:
        x, y = self.popn(2)
        if op == 'subscr':
            self.push(x[y])
            self.cur_offset += 10
        else:
            if op == '+=':
                x += y
            elif op == '**=':
                x **= y
            elif op == '*=':
                x *= y
            elif op in ['//=']:
                x //= y
            elif op == '/=':
                x /= y
            elif op == '%=':
                x %= y
            elif op == '-=':
                x -= y
            elif op == '<<=':
                x <<= y
            elif op == '>>=':
                x >>= y
            elif op == '&=':
                x &= y
            elif op == '^=':
                x ^= y
            elif op == '|=':
                x |= y
            elif op == '+':
                x = x + y
            elif op == '**':
                x = x ** y
            elif op == '*':
                x = x * y
            elif op == '//':
                x = x // y
            elif op == '/':
                x = x / y
            elif op == '%':
                x = x % y
            elif op == '-':
                x = x - y
            elif op == '<<':
                x = x << y
            elif op == '>>':
                x = x >> y
            elif op == '&':
                x = x & y
            elif op == '^':
                x = x ^ y
            elif op == '|':
                x = x | y
            else:
                raise NameError
            self.push(x)
            self.cur_offset += 4

    def unary_positive_op(self, arg: tp.Any) -> None:
        self.push(+self.pop())

    def unary_negative_op(self, arg: tp.Any) -> None:
        self.push(-self.pop())

    def unary_not_op(self, arg: tp.Any) -> None:
        self.push(not self.pop())

    def unary_invert_op(self, arg: tp.Any) -> None:
        self.push(~self.pop())

    def store_global_op(self, arg: str) -> None:
        self.globals[arg] = self.pop()

    def store_name_op(self, arg: str) -> None:
        self.locals[arg] = self.pop()

    def store_fast_op(self, arg: str) -> None:
        self.locals[arg] = self.pop()

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(arg)

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-RETURN_VALUE
        """
        self.return_value = self.pop()

    def pop_top_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_TOP
        """
        self.pop()

    def get_iter_op(self, arg: tp.Any) -> None:
        self.push(iter(self.pop()))

    def for_iter_op(self, delta: int) -> None:
        try:
            self.push(next(self.top()))
        except StopIteration:
            self.pop()
            self.cur_offset = delta

    def unpack_sequence_op(self, count: int) -> None:
        tos = self.pop()
        for el in reversed(tos):
            self.push(el)
        self.cur_offset += 4

    def compare_op_op(self, op: str) -> None:
        x, y = self.popn(2)
        if op == '==':
            self.push(x == y)
        elif op == '!=':
            self.push((x != y))
        elif op == '<=':
            self.push((x <= y))
        elif op == '>=':
            self.push((x >= y))
        elif op == '<':
            self.push((x < y))
        elif op == '>':
            self.push((x > y))
        else:
            raise NameError
        self.cur_offset += 6

    def pop_jump_forward_if_false_op(self, delta: int) -> None:
        if not self.pop():
            self.cur_offset = delta

    def pop_jump_backward_if_false_op(self, delta: int) -> None:
        if not self.pop():
            self.cur_offset = delta

    def pop_jump_forward_if_true_op(self, delta: int) -> None:
        if self.pop():
            self.cur_offset = delta

    def pop_jump_backward_if_true_op(self, delta: int) -> None:
        if self.pop():
            self.cur_offset = delta

    def build_slice_op(self, argc: int) -> None:
        if argc in {2, 3}:
            self.push(slice(*self.popn(argc)))
        else:
            raise ValueError

    def build_list_op(self, count: int) -> None:
        self.push(self.popn(count))

    def build_tuple_op(self, count: int) -> None:
        self.push(tuple(self.popn(count)))

    def build_set_op(self, count: int) -> None:
        self.push(set(self.popn(count)))

    def build_map_op(self, count: int) -> None:
        toses = self.popn(count * 2)
        keys = toses[::2]
        values = toses[1::2]
        self.push(dict(zip(keys, values)))

    def load_build_class_op(self, arg: tp.Any) -> None:
        self.push(builtins.__build_class__)

    def load_method_op(self, arg: str) -> None:
        tos = self.pop()
        if hasattr(tos, arg):
            self.push(getattr(tos, arg))
            self.push(tos)
        else:
            self.push(None)
            self.push(tos)
        self.cur_offset += 22

    def store_subscr_op(self, count: int) -> None:
        tos2, tos1, tos = self.popn(3)
        tos1[tos] = tos2
        self.push(tos2)
        self.push(tos1)
        self.cur_offset += 4

    def delete_subscr_op(self, count: int) -> None:
        tos1, tos = self.popn(2)
        del tos1[tos]
        self.push(tos1)
        self.push(tos)

    def list_extend_op(self, i: int) -> None:
        tos = self.pop()
        list.extend(self.data_stack[-i], tos)

    def set_update_op(self, i: int) -> None:
        tos = self.pop()
        set.update(self.data_stack[-i], tos)

    def build_string_op(self, count: int) -> None:
        self.push(''.join(self.popn(count)))

    def format_value_op(self, flags: int) -> None:
        if (flags & 0x03) == 0x00:
            return
        elif (flags & 0x03) == 0x01:
            self.push(str(self.pop()))
        elif (flags & 0x03) == 0x02:
            self.push(repr(self.pop()))
        elif (flags & 0x03) == 0x03:
            self.push(ascii(self.pop()))

    def build_const_key_map_op(self, count: int) -> None:
        tos = self.pop()
        values = self.popn(len(tos))
        self.push(dict(zip(tos, values)))

    def store_attr_op(self, arg: str) -> None:
        setattr(self.pop(), arg, self.pop())
        self.cur_offset += 10

    def load_attr_op(self, arg: str) -> None:
        self.push(getattr(self.pop(), arg))
        self.cur_offset += 10

    def nop_op(self, arg: tp.Any) -> None:
        pass

    def delete_attr_op(self, arg: str) -> None:
        delattr(self.pop(), arg)

    def copy_op(self, i: int) -> None:
        self.push(self.data_stack[-i])

    def swap_op(self, i: int) -> None:
        self.data_stack[-1], self.data_stack[-i] \
            = self.data_stack[-i], self.data_stack[-1]

    def is_op_op(self, inverse: int) -> None:
        x, y = self.popn(2)
        if inverse == 1:
            self.push(x is not y)
        else:
            self.push(x is y)

    def delete_name_op(self, arg: str) -> None:
        if arg in self.locals:
            del self.locals[arg]
        elif arg in self.globals:
            del self.globals[arg]
        elif arg in self.builtins:
            del self.builtins[arg]
        else:
            raise NameError

    def delete_global_op(self, arg: str) -> None:
        if arg in self.globals:
            del self.globals[arg]
        elif arg in self.builtins:
            del self.builtins[arg]
        else:
            raise NameError

    def delete_fast_op(self, arg: str) -> None:
        if arg in self.locals:
            del self.locals[arg]
        elif arg in self.globals:
            del self.globals[arg]
        elif arg in self.builtins:
            del self.builtins[arg]
        else:
            raise NameError

    def import_name_op(self, name: str) -> None:
        self.push(__import__(name, self.pop(), self.pop()))

    def import_star_op(self, arg: tp.Any) -> None:
        name = self.pop()
        for attr in dir(name):
            if attr[0] != '_':
                self.locals[attr] = getattr(name, attr)

    def import_from_op(self, arg: str) -> None:
        name = self.pop()
        self.push(0)
        if arg in dir(name):
            self.push(getattr(name, arg))
        else:
            raise NameError

    def jump_if_true_or_pop_op(self, delta: int) -> None:
        if self.data_stack[-1]:
            self.cur_offset = delta
        else:
            self.pop()

    def jump_if_false_or_pop_op(self, delta: int) -> None:
        if not self.data_stack[-1]:
            self.cur_offset = delta
        else:
            self.pop()

    def jump_forward_op(self, delta: int) -> None:
        self.cur_offset = delta

    def jump_backward_op(self, delta: int) -> None:
        self.cur_offset = delta

    def pop_jump_forward_if_not_none_op(self, delta: int) -> None:
        if self.pop() is not None:
            self.cur_offset = delta

    def pop_jump_forward_if_none_op(self, delta: int) -> None:
        if self.pop() is None:
            self.cur_offset = delta

    def pop_jump_backward_if_not_none_op(self, delta: int) -> None:
        if self.pop() is not None:
            self.cur_offset = delta

    def pop_jump_backward_if_none_op(self, delta: int) -> None:
        if self.pop() is None:
            self.cur_offset = delta

    def contains_op_op(self, invert: int) -> None:
        tos1, tos = self.popn(2)
        if invert == 1:
            self.push(tos1 not in tos)
        else:
            self.push(tos1 in tos)

    def list_to_tuple_op(self, arg: tp.Any) -> None:
        self.push(tuple(self.pop()))

    def get_len_op(self, arg: tp.Any) -> None:
        self.push(len(self.data_stack[-1]))

    def setup_annotations_op(self, arg: tp.Any) -> None:
        if '__annotations__' not in self.locals:
            self.locals['__annotations__'] = dict()

    def make_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-MAKE_FUNCTION
        """
        code = self.pop()  # the code associated with the function (at TOS1)
        kw_defaults = {}
        defaults = ()
        annotations = ()

        if arg & 0x04:
            annotations = self.pop()
        if arg & 0x02:
            kw_defaults = self.pop()
        if arg & 0x01:
            defaults = self.pop()

        def bind_args(code: tp.Any, *args: tp.Any,
                      **kwargs: tp.Any) -> dict[str, tp.Any]:
            """Bind values from `args` and `kwargs` to
            corresponding arguments of `func`

            :param code: function to be inspected
            :param args: positional arguments to be bound
            :param kwargs: keyword arguments to be bound
            :return: `dict[argument_name] = argument_value`
                      if binding was successful,
                     raise TypeError with one of `ERR_*`
                     error descriptions otherwise
            """
            ans: dict[str, tp.Any] = dict()
            names = code.co_varnames
            pos_only = code.co_posonlyargcount
            named_only = code.co_kwonlyargcount
            num_of_args = code.co_argcount
            have_starargs = bool(code.co_flags & CO_VARARGS)
            have_starkwargs = bool(code.co_flags & CO_VARKEYWORDS)
            def_values_of_pos = defaults
            default_varnames_of_pos = names[num_of_args -
                                            len(def_values_of_pos):num_of_args]
            def_values_of_named = kw_defaults

            for i in range(pos_only):
                if names[i] in kwargs and not have_starkwargs:
                    raise TypeError(ERR_POSONLY_PASSED_AS_KW)
                elif len(args) < i + 1:
                    if not default_varnames_of_pos.count(names[i]):
                        raise TypeError(ERR_MISSING_POS_ARGS)
                    else:
                        ans[names[i]] = def_values_of_pos[
                            default_varnames_of_pos.index(names[i])]
                else:
                    ans[names[i]] = args[i]

            for i in range(pos_only, num_of_args):
                if len(args) < i + 1:
                    if (not default_varnames_of_pos.count(names[i])
                            and names[i] not in kwargs):
                        raise TypeError(ERR_MISSING_POS_ARGS)
                    elif names[i] in kwargs:
                        ans[names[i]] = kwargs[names[i]]
                    else:
                        ans[names[i]] = def_values_of_pos[
                            default_varnames_of_pos.index(names[i])]
                else:
                    if names[i] in kwargs:
                        raise TypeError(ERR_MULT_VALUES_FOR_ARG)
                    ans[names[i]] = args[i]

            if have_starargs:
                if num_of_args < len(args):
                    ans[names[num_of_args + named_only]] = \
                        tuple(args[num_of_args:])
                else:
                    ans[names[num_of_args + named_only]] = tuple()
            else:
                if num_of_args < len(args):
                    raise TypeError(ERR_TOO_MANY_POS_ARGS)

            for i in range(num_of_args, num_of_args + named_only):
                if names[i] not in kwargs:
                    if names[i] not in def_values_of_named:
                        raise TypeError(ERR_MISSING_KWONLY_ARGS)
                    else:
                        ans[names[i]] = def_values_of_named[names[i]]
                else:
                    ans[names[i]] = kwargs[names[i]]

            if have_starkwargs:
                name = names[num_of_args + named_only + have_starargs]
                ans[name] = {}
                for key, value in kwargs.items():
                    if key not in ans or ans[key] != value:
                        ans[name][key] = value
            else:
                for key, value in kwargs.items():
                    if key not in ans:
                        raise TypeError(ERR_TOO_MANY_KW_ARGS)
                    elif ans[key] != value:
                        raise TypeError(ERR_MULT_VALUES_FOR_ARG)

            return ans

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            parsed_args: dict[str, tp.Any] = bind_args(code, *args, **kwargs)
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            # Run code in prepared environment
            frame = Frame(code, self.builtins, self.globals, f_locals)
            return frame.run()

        f.__annotations__ = dict(zip(annotations[::2], annotations[1::2]))

        self.push(f)


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'],
                      globals_context, globals_context)
        return frame.run()
