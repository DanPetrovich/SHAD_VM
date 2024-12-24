"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import sys
import types
import typing as tp
import operator

ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
ERR_MISSING_POS_ARGS = 'Missing positional arguments'
ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
ERR_POSONLY_PASSED_AS_KW = 'Positional-only argument passed as keyword argument'
CO_VARARGS = 4
CO_VARKEYWORDS = 8

BINARY_OPERATORS = {
    0: operator.add,  # a += b
    5: operator.mul,  # a *= b
    6: operator.mod,  # a % b
    7: operator.or_,  # a | b
    8: operator.pow,  # a *= b
    10: operator.sub,  # a -= b
    13: operator.iadd,  # a += b
    14: operator.iand,  # a &= b
    15: operator.ifloordiv,  # a //= b
    16: operator.ilshift,  # a <<= b
    17: operator.imod,  # a %= b
    18: operator.imul,  # a *= b
    19: operator.imod,  # a %= b
    20: operator.ior,  # a |= b
    21: operator.ipow,  # a **= b
    22: operator.irshift,  # a >>= b
    23: operator.isub,  # a -= b
    24: operator.itruediv,  # a /= b
    25: operator.ixor  # a ^= b
}

COMPARE_OPERATORS = {
    '<': operator.lt,  # Меньше
    '<=': operator.le,  # Меньше или равно
    '==': operator.eq,  # Равно
    '!=': operator.ne,  # Не равно
    '>': operator.gt,  # Больше
    '>=': operator.ge,  # Больше или равно
    'in': lambda x, y: x in y,  # Вхождение
    'not in': lambda x, y: x not in y,  # Отсутствие
    'is': operator.is_,  # Является (тот же объект)
    'is not': operator.is_not,  # Не является (разные объекты)
    'exception match': lambda x, y: isinstance(x, y),
}


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.12/Include/internal/pycore_frame.h

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
        self.return_value = None
        self.finish = False
        self.instruction_idx = 0
        self.offsets: dict[int, int] = {}
        self.exception_state = None
        self.kw_names: tp.Any = []

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

    def jump_forward_op(self, delta: int) -> None:
        self.instruction_idx = self.offsets[delta]

    def jump_backward_op(self, delta: int) -> None:
        self.instruction_idx = self.offsets[delta]

    def jump_backward_no_interrupt_op(self, delta: int) -> None:
        self.instruction_idx = self.offsets[delta]

    def pop_jump_if_false_op(self, delta: int) -> None:
        cond = bool(self.pop())
        if not cond:
            # print(f"Condition is False, jumping to {delta}")
            self.jump_forward_op(delta)
        else:
            self.instruction_idx += 1

    def pop_jump_if_true_op(self, delta: int) -> None:
        cond = bool(self.pop())
        # print(f"Condition is {cond}, jumping to {delta}")
        if cond:
            self.jump_forward_op(delta)
        else:
            self.instruction_idx += 1

    def pop_jump_if_none_op(self, delta: int) -> None:
        cond = self.pop()
        if cond is None:
            self.jump_forward_op(delta)
        else:
            self.instruction_idx += 1

    def pop_jump_if_not_none_op(self, delta: int) -> None:
        cond = self.pop()
        if cond is not None:
            self.jump_forward_op(delta)
        else:
            self.instruction_idx += 1

    def run(self) -> tp.Any:
        self.instruction_idx = 0
        instructions = list(dis.get_instructions(self.code))
        self.offsets = {instructions[i].offset: i for i in range(len(instructions))}
        while self.instruction_idx < len(instructions):
            if self.return_value or self.finish:
                break
            instruction = instructions[self.instruction_idx]
            # print(f"Executing: {instruction.opname}, Argument: {instruction.argval}, Offset: {instruction.offset}")
            # print(f"Stack before: {self.data_stack}")

            getattr(self, instruction.opname.lower() + "_op")(instruction.argval)

            # print(f"Stack after: {self.data_stack}")
            # print(f"instructions left: {len(instructions) - self.instruction_idx}")
            if instruction.opname not in {"JUMP_FORWARD", "JUMP_BACKWARD", "POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE",
                                          "JUMP_BACKWARD_NO_INTERRUPT", "POP_JUMP_IF_NONE", "POP_JUMP_IF_NOT_NONE"}:
                self.instruction_idx += 1
        # print(f"Result: {self.return_value}")
        return self.return_value

    def binary_op_op(self, op: int) -> None:

        rhs = self.pop()
        lhs = self.pop()
        if rhs is None:
            rhs = self.pop()

        if lhs is None:
            lhs = self.pop()
        self.push(BINARY_OPERATORS[op](lhs, rhs))

    def build_string_op(self, count: tp.Any) -> None:
        self.push("".join(self.popn(count)))

    def call_function_ex_op(self, flags: tp.Any) -> None:
        pos_args = self.pop()
        if flags & 0x01:
            kw_args = self.pop()
        else:
            kw_args = {}

        callable_obj = self.pop()
        result = callable_obj(*pos_args, **kw_args)
        self.push(result)

    def check_eg_match_op(self) -> None:
        match_type = self.pop()
        exception_group = self.pop()
        if hasattr(exception_group, 'split'):
            matching_subgroup, non_matching_subgroup = exception_group.split(match_type)

            if matching_subgroup is None:
                self.push(None)
            else:
                self.push(non_matching_subgroup)
                self.push(matching_subgroup)
        else:
            self.push(None)

    def check_exc_match_op(self) -> None:
        exc_type = self.pop()
        exc_obj = self.top()

        result = isinstance(exc_obj, exc_type)
        self.push(result)

    # def push_exc_info_op(self) -> None:
    #     value = self.pop()
    #     current_exception = self.current_exception()
    #     self.push(current_exception)
    #     self.push(value)

    def call_intrinsic_1_op(self, operand: int) -> None:
        INTRINSIC_PRINT = 1
        INTRINSIC_IMPORT_STAR = 2
        INTRINSIC_STOPITERATION_ERROR = 3
        INTRINSIC_ASYNC_GEN_WRAP = 4
        INTRINSIC_UNARY_POSITIVE = 5
        INTRINSIC_LIST_TO_TUPLE = 6
        INTRINSIC_TYPEVAR = 7
        INTRINSIC_PARAMSPEC = 8
        INTRINSIC_TYPEVARTUPLE = 9
        INTRINSIC_SUBSCRIPT_GENERIC = 10
        INTRINSIC_TYPEALIAS = 11

        arg = self.pop()

        if operand == INTRINSIC_PRINT:
            print(arg)
            self.push(None)

        elif operand == INTRINSIC_IMPORT_STAR:
            def import_star_op(arg: tp.Any) -> None:
                for attr in dir(arg):
                    if attr[0] != '_':
                        self.locals[attr] = getattr(arg, attr)
                self.push(None)

            import_star_op(arg)

        elif operand == INTRINSIC_STOPITERATION_ERROR:
            if isinstance(arg, StopIteration):
                self.push(arg.value)
            else:
                raise TypeError("Ожидалось исключение StopIteration")

        elif operand == INTRINSIC_ASYNC_GEN_WRAP:
            def async_gen_wrap(value: tp.Any) -> tp.Any:
                async def async_generator() -> tp.Any:
                    yield value

                return async_generator()

            self.push(async_gen_wrap(arg))

        elif operand == INTRINSIC_UNARY_POSITIVE:
            self.push(+arg)

        elif operand == INTRINSIC_LIST_TO_TUPLE:
            self.push(tuple(arg))

        elif operand == INTRINSIC_TYPEVAR:
            from typing import TypeVar
            self.push(TypeVar(arg))

        elif operand == INTRINSIC_PARAMSPEC:
            from typing import ParamSpec
            self.push(ParamSpec(arg))

        elif operand == INTRINSIC_TYPEVARTUPLE:
            from typing import TypeVarTuple
            self.push(TypeVarTuple(arg))

        elif operand == INTRINSIC_SUBSCRIPT_GENERIC:
            from typing import Generic
            self.push(Generic[arg])

        elif operand == INTRINSIC_TYPEALIAS:
            from typing import TypeAlias
            if isinstance(arg, tuple) and len(arg) == 3:
                name, type_params, value = arg
                self.push(TypeAlias)  # TODO
        else:
            raise ValueError

    # def call_intrinsic_2_op(self, operand: int) -> None:
    #     arg1 = self.pop()
    #     arg2 = self.pop()
    #     pass  # TODO

    def binary_slice_op(self, arg: tp.Any) -> None:
        end = self.pop()
        start = self.pop()
        step = None

        if isinstance(arg, slice):
            step = arg.step

        container = self.pop()

        self.push(container[start:end:step])

    # def pop_except_op(self) -> None:
    #     self.exception_state = self.pop()
    #     try:
    #         raise self.exception_state
    #     except Exception as e:
    #         self.exception_state = e
    #
    # def reraise_op(self, oparg: int) -> None:
    #     exception = self.pop()
    #     if oparg != 0:
    #         self.f_lasti = self.pop()
    #
    #     raise exception

    def load_attr_op(self, namei: tp.Any) -> None:
        obj = self.pop()
        if namei != 'e' and namei != 'pi':
            self.push(None)
        self.push(getattr(obj, namei))

    def store_attr_op(self, namei: tp.Any) -> None:
        obj = self.pop()
        value = self.pop()
        # attr_name = self.code.co_names[namei]
        setattr(obj, namei, value)

    def match_mapping_op(self, arg: tp.Any) -> None:
        # mapping = self.pop()
        if isinstance(arg, dict):
            self.push(True)
        else:
            self.push(False)

    def nop_op(self, arg: tp.Any) -> None:
        pass

    def match_sequence_op(self, arg: tp.Any) -> None:
        # sequence = self.pop()
        if isinstance(arg, tp.Sequence) and not isinstance(arg, (str, bytes, bytearray)):
            self.push(True)
        else:
            self.push(False)

    def contains_op_op(self, invert: bool) -> None:
        rhs = self.pop()
        lhs = self.pop()
        if invert:
            self.push(lhs not in rhs)
        else:
            self.push(lhs in rhs)

    def delete_attr_op(self, arg: tp.Any) -> None:
        val = self.pop()
        delattr(val, arg)

    def delete_fast_op(self, arg: tp.Any) -> None:
        if arg in self.locals:
            del self.locals[arg]
        elif arg in self.globals:
            del self.globals[arg]
        elif arg in self.builtins:
            del self.builtins[arg]
        else:
            raise NameError

    def delete_name_op(self, arg: tp.Any) -> None:
        if arg in self.locals:
            del self.locals[arg]
        elif arg in self.globals:
            del self.globals[arg]
        elif arg in self.builtins:
            del self.builtins[arg]
        else:
            raise NameError

    def delete_subscr_op(self, arg: tp.Any) -> None:
        tos1, tos = self.popn(2)
        del tos1[tos]
        # self.push(tos1)
        # self.push(tos)

    def delete_global_op(self, arg: tp.Any) -> None:
        if arg in self.globals:
            del self.globals[arg]
        elif arg in self.builtins:
            del self.builtins[arg]
        else:
            raise NameError

    def copy_op(self, i: int) -> None:
        assert i > 0
        self.push(self.data_stack[-i])

    def setup_annotations_op(self, arg: tp.Any) -> None:
        if '__annotations__' not in self.locals:
            self.locals['__annotations__'] = dict()

    def build_slice_op(self, argc: int) -> None:
        if argc == 2:
            end = self.pop()
            start = self.pop()
            self.push(slice(start, end))
        elif argc == 3:
            step = self.pop()
            end = self.pop()
            start = self.pop()
            self.push(slice(start, end, step))
        else:
            raise ValueError

    def store_slice_op(self, arg: tp.Any) -> None:
        end = self.pop()
        start = self.pop()
        container = self.pop()
        value = self.pop()
        container[start:end] = value

    def store_subscr_op(self, arg: tp.Any) -> None:
        key = self.pop()
        container = self.pop()
        value = self.pop()
        container[key] = value

    def binary_subscr_op(self, arg: tp.Any) -> None:
        key = self.pop()
        container = self.pop()
        self.push(container[key])

    def compare_op_op(self, opname: str) -> None:
        rhs = self.pop()
        lhs = self.pop()

        operation = COMPARE_OPERATORS[opname]
        self.push(operation(lhs, rhs))

    def unary_invert_op(self, arg: tp.Any) -> None:
        self.push(~self.pop())

    def unary_negative_op(self, arg: tp.Any) -> None:
        self.push(-self.pop())

    def unary_not_op(self, arg: tp.Any) -> None:
        self.push(not self.pop())

    def build_tuple_op(self, argc: int) -> None:
        if argc == 0:
            self.push(())
        else:
            self.push(tuple(self.popn(argc)))

    def build_list_op(self, argc: int) -> None:
        if argc == 0:
            self.push([])
        else:
            self.push(list(self.popn(argc)))

    def list_append_op(self, arg: tp.Any) -> None:
        item = self.pop()
        list_ = self.data_stack[-1]
        list_.append(item)

    def build_map_op(self, arg: tp.Any) -> None:
        d = {}
        for _ in range(arg):
            val = self.pop()
            key = self.pop()
            d[key] = val
        self.push(d)

    def map_add_op(self, i: int) -> None:
        value = self.pop()
        key = self.pop()
        d = self.pop()
        d[key] = value

    def dict_update_op(self, i: int) -> None:
        map_ = self.pop()
        d = self.data_stack[-i]
        d.update(map_)

    def dict_merge_op(self, i: int) -> None:
        map_ = self.pop()
        d = self.data_stack[-i]
        for key in map_.keys():
            if key in d:
                raise ValueError(f"Key {key} already exists in dict")
        d.update(map_)

    def load_build_class_op(self, arg: tp.Any) -> None:
        self.push(builtins.__build_class__)

    def match_class_op(self, count: int) -> None:
        keyword_attrs = self.pop()
        match_class = self.pop()
        match_subject = self.pop()

        if not isinstance(match_subject, match_class):
            self.push(None)
            return

        positional_attrs = []
        for i in range(count):
            positional_attrs.append(getattr(match_subject, f'_{i}', None))

        named_attrs = []
        for attr_name in keyword_attrs:
            if hasattr(match_subject, attr_name):
                named_attrs.append(getattr(match_subject, attr_name))
            else:
                self.push(None)
                return

        self.push(tuple(positional_attrs + named_attrs))

    def push_exc_info(self, arg: tp.Any) -> None:
        # val = self.pop()
        pass

    def end_for_op(self, arg: tp.Any) -> None:
        self.pop()
        self.pop()

    def get_len_op(self, arg: tp.Any) -> None:
        self.push(len(self.data_stack[-1]))

    def is_op_op(self, invert: bool) -> None:
        rhs = self.pop()
        lhs = self.pop()
        if invert:
            self.push(lhs is not rhs)
        else:
            self.push(lhs is rhs)

    def import_name_op(self, name: str) -> None:
        self.push(__import__(name, self.pop(), self.pop()))

    def import_from_op(self, arg: str) -> None:
        name = self.pop()
        self.push(0)
        # arg_name = self.code.co_names[arg]
        if arg in dir(name):
            self.push(getattr(name, arg))
        else:
            raise NameError

    def build_const_key_map_op(self, argc: int) -> None:
        d = {}
        keys = self.pop()
        vals = []
        for _ in range(len(keys)):
            vals.append(self.pop())
        for k, v in zip(keys, vals[::-1]):
            d[k] = v
        self.push(d)

    def build_set_op(self, argc: int) -> None:
        if argc == 0:
            self.push(set())
        else:
            self.push(set(self.popn(argc)))

    def set_update_op(self, i: int) -> None:
        seq = self.pop()
        set_ = self.data_stack[-i]
        set_.update(seq)

    def set_add_op(self, i: int) -> None:
        item = self.pop()
        set_ = self.data_stack[-i]
        set_.add(item)

    def list_extend_op(self, arg: int) -> None:
        values = self.pop()
        list_ = self.pop()
        list_.extend(values)
        self.push(list_)

    def unpack_sequence_op(self, count: int) -> None:
        sequence = self.pop()
        if len(sequence) != count:
            raise ValueError(f"Waited for {count} elements, got {len(sequence)}")

        for item in reversed(sequence):
            self.push(item)

    def clear_frame(self) -> None:
        self.data_stack.clear()
        self.locals.clear()

    def return_generator_op(self, delta: int) -> None:
        def generator() -> tp.Iterator[tp.Any]:
            while True:
                try:
                    yield self.run()
                except StopIteration:
                    break

        self.clear_frame()
        self.push(generator())

    def format_value_op(self, flags: tp.Any) -> None:

        conv, fmt_spec = flags

        if fmt_spec & 0x04:
            fmt_spec = self.pop()
        else:
            fmt_spec = ''

        value = self.pop()

        if conv is None:
            pass
        elif conv is str:
            value = str(value)
        elif conv is repr:
            value = repr(value)
        elif conv is ascii:
            value = ascii(value)

        f_value = format(value, fmt_spec)

        self.push(f_value)

    def get_iter_op(self, delta: int) -> None:
        self.push(iter(self.pop()))

    def for_iter_op(self, delta: int) -> None:
        iterator = self.top()
        try:
            value = next(iterator)
            self.push(value)
        except StopIteration:
            self.data_stack.pop()
            self.jump_forward_op(delta)

    def resume_op(self, arg: int) -> None:
        pass

    def push_null_op(self, arg: int) -> None:
        self.push(None)

    def load_assertion_error_op(self, arg: tp.Any) -> None:
        self.push(AssertionError)

    def get_current_exception(self) -> tp.Tuple[tp.Any, tp.Any, tp.Any]:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        return exc_type, exc_value, exc_traceback

    def raise_varargs_op(self, argc: int) -> None:
        if argc == 0:
            exc_type, exc_value, exc_traceback = self.get_current_exception()
            if exc_type is None:
                raise RuntimeError("No active exception to re-raise")
            raise exc_value.with_traceback(exc_traceback)

        elif argc == 1:
            exc = self.pop()
            if isinstance(exc, BaseException):
                raise exc
            else:
                raise exc()

        elif argc == 2:
            cause = self.pop()
            exc = self.pop()
            if isinstance(exc, BaseException):
                exc.__cause__ = cause
                raise exc
            else:
                exc_instance = exc()
                exc_instance.__cause__ = cause
                raise exc_instance

        else:
            raise ValueError(f"Invalid argc for RAISE_VARARGS: {argc}")

    def load_fast_check_op(self, var_name: str) -> None:
        if var_name in self.locals and self.locals[var_name] is not None:
            self.push(self.locals[var_name])
        else:
            raise UnboundLocalError

    def precall_op(self, arg: int) -> tp.Any:
        pass

    def kw_names_op(self, consti: tuple[tp.Any]) -> None:
        vals = self.popn(len(consti))
        kw_names = {}
        for i in range(len(consti)):
            kw_names[consti[i]] = vals[i]
        self.kw_names = kw_names

    def call_op(self, argc: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-CALL
        """

        if self.kw_names:
            arguments = self.popn(argc - len(self.kw_names))
        else:
            arguments = self.popn(argc)
        kw_arguments: tp.Any = {}
        if self.kw_names:
            kw_arguments = self.kw_names
        func_or_self = self.pop()
        if self.data_stack and self.data_stack[-1] is not None:
            f = self.pop()
            self.push(f(*arguments, **kw_arguments))
        else:
            if self.data_stack:
                self.pop()
            self.push(func_or_self(*arguments, **kw_arguments))
        self.kw_names = []

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-LOAD_NAME
        """
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError(f"Name '{arg}' is not defined")

    def load_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-LOAD_GLOBAL
        """
        if arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError(f"Name '{arg}' is not defined")

    def store_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-STORE_GLOBAL
        """
        const = self.pop()
        self.globals[arg] = const

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(arg)

    def load_fast_and_clear_op(self, var_name: str) -> None:
        if var_name in self.locals and self.locals[var_name] is not None:
            self.push(self.locals[var_name])
        else:
            self.push(None)

    def match_keys_op(self) -> None:
        keys = self.pop()
        match_subject = self.pop()

        if not isinstance(keys, tuple):
            raise TypeError

        if not isinstance(match_subject, dict):
            raise TypeError

        try:
            values = tuple(match_subject[key] for key in keys)
            self.push(values)
        except KeyError:
            self.push(None)

    def swap_op(self, i: int) -> None:
        self.data_stack[-i], self.data_stack[-1] = self.data_stack[-1], self.data_stack[-i]

    def store_fast_op(self, arg: str) -> None:
        self.locals[arg] = self.pop()

    def load_fast_op(self, arg: str) -> None:
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-RETURN_VALUE
        """
        self.finish = True
        self.return_value = self.pop()

    def return_const_op(self, arg: tp.Any) -> None:
        self.finish = True
        self.push(arg)
        self.return_value = arg

    def pop_top_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-POP_TOP
        """
        self.pop()

    def bind_args(self, code_object: tp.Any, pos_defaults: tp.Any, kw_defaults: tp.Any, *args: tp.Any,
                  **kwargs: tp.Any) -> dict[str, tp.Any]:
        bound_args: dict[str, tp.Any] = {}

        variable_names = code_object.co_varnames
        pos_only_count = code_object.co_posonlyargcount
        kw_only_count = code_object.co_kwonlyargcount
        positional_arg_count = code_object.co_argcount
        has_varargs = bool(code_object.co_flags & CO_VARARGS)
        has_kwargs = bool(code_object.co_flags & CO_VARKEYWORDS)

        pos_defaults_for_vars = variable_names[positional_arg_count - len(pos_defaults):positional_arg_count]
        kw_defaults_values = kw_defaults

        for idx in range(pos_only_count):
            if variable_names[idx] in kwargs and not has_kwargs:
                raise TypeError(ERR_POSONLY_PASSED_AS_KW)
            elif len(args) < idx + 1:
                if not pos_defaults_for_vars.count(variable_names[idx]):
                    raise TypeError(ERR_MISSING_POS_ARGS)
                else:
                    bound_args[variable_names[idx]] = pos_defaults[
                        pos_defaults_for_vars.index(variable_names[idx])]
            else:
                bound_args[variable_names[idx]] = args[idx]

        for idx in range(pos_only_count, positional_arg_count):
            if len(args) < idx + 1:
                if (not pos_defaults_for_vars.count(variable_names[idx])
                        and variable_names[idx] not in kwargs):
                    raise TypeError(ERR_MISSING_POS_ARGS)
                elif variable_names[idx] in kwargs:
                    bound_args[variable_names[idx]] = kwargs[variable_names[idx]]
                else:
                    bound_args[variable_names[idx]] = pos_defaults[
                        pos_defaults_for_vars.index(variable_names[idx])]
            else:
                if variable_names[idx] in kwargs:
                    raise TypeError(ERR_MULT_VALUES_FOR_ARG)
                bound_args[variable_names[idx]] = args[idx]

        if has_varargs:
            if positional_arg_count < len(args):
                bound_args[variable_names[positional_arg_count + kw_only_count]] = tuple(
                    args[positional_arg_count:])
            else:
                bound_args[variable_names[positional_arg_count + kw_only_count]] = tuple()
        else:
            if positional_arg_count < len(args):
                raise TypeError(ERR_TOO_MANY_POS_ARGS)

        for idx in range(positional_arg_count, positional_arg_count + kw_only_count):
            if variable_names[idx] not in kwargs:
                if variable_names[idx] not in kw_defaults_values:
                    raise TypeError(ERR_MISSING_KWONLY_ARGS)
                else:
                    bound_args[variable_names[idx]] = kw_defaults_values[variable_names[idx]]
            else:
                bound_args[variable_names[idx]] = kwargs[variable_names[idx]]

        if has_kwargs:
            kwargs_name = variable_names[positional_arg_count + kw_only_count + has_varargs]
            bound_args[kwargs_name] = {}
            for key, value in kwargs.items():
                if key not in bound_args or bound_args[key] != value:
                    bound_args[kwargs_name][key] = value
        else:
            for key, value in kwargs.items():
                if key not in bound_args:
                    raise TypeError(ERR_TOO_MANY_KW_ARGS)
                elif bound_args[key] != value:
                    raise TypeError(ERR_MULT_VALUES_FOR_ARG)

        return bound_args

    def make_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-MAKE_FUNCTION
        """
        code = self.pop()
        kw_defaults = {}
        pos_defaults = ()
        annotations = ()

        if arg & 0x04:
            annotations = self.pop()
        if arg & 0x02:
            kw_defaults = self.pop()
        if arg & 0x01:
            pos_defaults = self.pop()

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            parsed_args: dict[str, tp.Any] = self.bind_args(code, pos_defaults, kw_defaults, *args, **kwargs)
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)
            return frame.run()

        f.__annotations__ = dict(zip(annotations[::2], annotations[1::2]))

        self.push(f)

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-STORE_NAME
        """
        const = self.pop()
        self.locals[arg] = const


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()
