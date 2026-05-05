from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, Optional, Tuple, TypeVar


A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


# ============================================================
# Q1
# ============================================================
ChurchBool = Callable[[Any], Callable[[Any], Any]]


def church_true(x: Any) -> Callable[[Any], Any]:
    return lambda y: x


def church_false(x: Any) -> Callable[[Any], Any]:
    return lambda y: y


def church_not(b: ChurchBool) -> ChurchBool:
    return lambda x: lambda y: b(y)(x)


def from_church_bool(b: ChurchBool) -> bool:
    return b(True)(False)


# ============================================================
# Q2
# ============================================================
def compose(f: Callable[[B], C], g: Callable[..., B]) -> Callable[..., C]:
    return lambda *args, **kwargs: f(g(*args, **kwargs))


# ============================================================
# Q3
# ============================================================
def map_recursive(fn: Callable[[A], B], xs: list[A]) -> list[B]:
    if not xs:
        return []
    return [fn(xs[0])] + map_recursive(fn, xs[1:])


# ============================================================
# Q4
# ============================================================
def fix(f: Callable[[Callable[..., Any]], Callable[..., Any]]) -> Callable[..., Any]:
    return (lambda x: f(lambda *args: x(x)(*args)))(
        lambda x: f(lambda *args: x(x)(*args))
    )


# ============================================================
# Q5
# ============================================================
class Type:
    pass


@dataclass(frozen=True)
class TInt(Type):
    pass


@dataclass(frozen=True)
class TBool(Type):
    pass


@dataclass(frozen=True)
class TFun(Type):
    arg: Type
    ret: Type


class Expr:
    pass


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class Lam(Expr):
    param: str
    param_type: Type
    body: Expr


@dataclass(frozen=True)
class App(Expr):
    fn: Expr
    arg: Expr


@dataclass(frozen=True)
class IntLit(Expr):
    value: int


@dataclass(frozen=True)
class BoolLit(Expr):
    value: bool


@dataclass(frozen=True)
class If(Expr):
    cond: Expr
    then_branch: Expr
    else_branch: Expr


@dataclass(frozen=True)
class Add(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Sub(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Mod(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class IsZero(Expr):
    value: Expr


TypeEnv = Dict[str, Type]


def type_of(expr: Expr, env: TypeEnv | None = None) -> Type:
    if env is None:
        env = {}

    if isinstance(expr, IntLit):
        return TInt()

    if isinstance(expr, BoolLit):
        return TBool()

    if isinstance(expr, Var):
        if expr.name in env:
            return env[expr.name]
        raise TypeError(f"Unbound variable: {expr.name}")

    if isinstance(expr, Lam):
        new_env = env.copy()
        new_env[expr.param] = expr.param_type
        body_type = type_of(expr.body, new_env)
        return TFun(expr.param_type, body_type)

    if isinstance(expr, App):
        fn_type = type_of(expr.fn, env)
        arg_type = type_of(expr.arg, env)

        if not isinstance(fn_type, TFun):
            raise TypeError("Trying to apply a non-function")

        if fn_type.arg != arg_type:
            raise TypeError("Argument type mismatch")

        return fn_type.ret

    if isinstance(expr, If):
        cond_type = type_of(expr.cond, env)
        if cond_type != TBool():
            raise TypeError("Condition must be boolean")

        t1 = type_of(expr.then_branch, env)
        t2 = type_of(expr.else_branch, env)

        if t1 != t2:
            raise TypeError("Branches must have same type")

        return t1

    if isinstance(expr, (Add, Sub, Mod)):
        left_type = type_of(expr.left, env)
        right_type = type_of(expr.right, env)

        if left_type != TInt() or right_type != TInt():
            raise TypeError("Arithmetic requires integers")

        return TInt()

    if isinstance(expr, IsZero):
        val_type = type_of(expr.value, env)
        if val_type != TInt():
            raise TypeError("IsZero requires integer")

        return TBool()

    raise TypeError("Unknown expression")


# ============================================================
# Q6 + Q7
# ============================================================
@dataclass
class Node:
    value: int
    next: Node | None = None


def ll_from_iterable(values: Iterable[int]) -> Node | None:
    head = None
    tail = None
    for v in values:
        n = Node(v)
        if head is None:
            head = n
            tail = n
        else:
            tail.next = n
            tail = n
    return head


def ll_to_list(head: Node | None) -> list[int]:
    out = []
    cur = head
    while cur:
        out.append(cur.value)
        cur = cur.next
    return out


def merge_sorted_lists(a: Node | None, b: Node | None) -> Node | None:
    if not a:
        return b
    if not b:
        return a

    if a.value <= b.value:
        a.next = merge_sorted_lists(a.next, b)
        return a
    else:
        b.next = merge_sorted_lists(a, b.next)
        return b


def merge_sort_linked(head: Node | None) -> Node | None:
    if not head or not head.next:
        return head

    # split using slow-fast pointer
    slow = head
    fast = head.next

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    mid = slow.next
    slow.next = None

    left = merge_sort_linked(head)
    right = merge_sort_linked(mid)

    return merge_sorted_lists(left, right)


# ============================================================
# Q8 + Q9
# ============================================================
@dataclass(frozen=True)
class Tree(Generic[A]):
    value: A
    left: Optional[Tree[A]] = None
    right: Optional[Tree[A]] = None


def foldl(fn: Callable[[B, A], B], init: B, xs: Iterable[A]) -> B:
    acc = init
    for x in xs:
        acc = fn(acc, x)
    return acc


def fold_tree(leaf: B, node_fn: Callable[[A, B, B], B], t: Optional[Tree[A]]) -> B:
    if t is None:
        return leaf
    l = fold_tree(leaf, node_fn, t.left)
    r = fold_tree(leaf, node_fn, t.right)
    return node_fn(t.value, l, r)


# ============================================================
# Q10
# ============================================================
def q5_curry(h: Callable[[Tuple[A, B]], C]) -> Callable[[A], Callable[[B], C]]:
    return lambda a: lambda b: h((a, b))


if __name__ == "__main__":
    # Test Q1
    print(from_church_bool(church_not(church_true)))
    # False

    # Test Q2
    print(compose(lambda x: x + 1, lambda x: x * 2)(10))
    # 21

    # Test Q3
    print(map_recursive(lambda x: x * x, [1, 2, 3]))
    # [1, 4, 9]

    # Test Q4
    fact = fix(lambda rec: lambda n: 1 if n == 0 else n * rec(n - 1))
    print(fact(5))
    # 120

    # Test Q5
    print(type_of(Add(IntLit(3), IntLit(2))))
    # TInt()

    # Test Q6 + Q7
    a = ll_from_iterable([1, 3, 5])
    b = ll_from_iterable([2, 4, 6])
    print(ll_to_list(merge_sorted_lists(a, b)))
    # [1, 2, 3, 4, 5, 6]

    # Test Q8
    print(foldl(lambda acc, x: acc + x, 0, [1, 2, 3, 4]))
    # 10

    # Test Q9
    t = Tree(2, Tree(1), Tree(3))
    print(fold_tree([], lambda v, l, r: l + [v] + r, t))
    # [1, 2, 3]

    # Test Q10
    print(q5_curry(lambda p: f"{p[0]}:{p[1]}")("A")("B"))
    # A:B