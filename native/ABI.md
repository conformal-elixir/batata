# Zig term runtime ABI (declaration-first manifest)

This file is the source of truth for the `ex` term universe ABI. Beaver's
conversion plan (`Beaver.MLIR.Conversion.Ex`) emits calls to the symbols
listed here, and [`term_runtime.zig`](./term_runtime.zig) implements them.
The two sides must stay in lockstep; change the manifest first, then both
sides.

## Term representation

A term is a 64-bit **tagged word** (`i64`). The low 3 bits hold the tag; the
remaining 61 bits hold the payload.

| tag | value | meaning |
| --- | --- | --- |
| `int` | `0b000` | immediate integer: `value << 3` |
| `atom` | `0b001` | immediate atom: `id << 3 \| 1` |
| `tuple` | `0b010` | heap pointer to a tuple header |
| `list` | `0b011` | heap pointer to a cons cell |
| `map` | `0b100` | heap pointer to a map header |
| `binary` | `0b101` | heap pointer to a binary header |
| `fun` | `0b110` | heap pointer to a closure header |

Heap objects are 8-byte aligned, so the low 3 bits of a container pointer are
always zero and the tag can be OR-ed in. `nil` is the atom with id 0
(`word == 1`); it doubles as the empty list, matching BEAM.

Heap layouts (all fields are `i64` words):

```
tuple:  [len] [elem × len]
map:    [len] [entry × 2*len]   (flat key/value pairs; len = pair count)
binary: [len] [byte × len]
list:   cons cells [head] [tail]
fun:    [fn_idx] [env_len] [env × env_len]
```

## Intrinsics

All functions use the C ABI and return/accept `i64` tagged words unless noted.

| symbol | signature | semantics |
| --- | --- | --- |
| `ex.term.list_cons` | `(head: i64, tail: i64) -> i64` | cons a word onto a list |
| `ex.term.self` | `() -> i64` | pid of the current actor (atom word with id 1) |
| `ex.term.send` | `(pid: i64, msg: i64) -> i64` | enqueue a message; returns the message, nil when the mailbox is full |
| `ex.term.receive` | `() -> i64` | dequeue the oldest message; nil when empty |
| `ex.term.mailbox_clear` | `() -> i64` | reset the mailbox; the compiled entry calls this at startup |
| `ex.term.to_int` | `(word: i64) -> i64` | untag an integer term to its scalar value; 0 for non-integers |
| `ex.term.make_fun` | `(fn_idx: i64, env_len: i64, e0..e3: i64) -> i64` | closure word referencing `__fn_*` by index with up to four captured env words |
| `ex.term.fun_idx` | `(fun: i64) -> i64` | function index of a closure; 0 for non-functions |
| `ex.term.fun_env` | `(fun: i64, index: i64) -> i64` | captured env word at index; nil for non-functions / out-of-range |
| `ex.term.tuple_from_list` | `(list: i64) -> i64` | proper list -> tuple |
| `ex.term.tuple_get` | `(tuple: i64, index: i64) -> i64` | element at index; nil when out of range or not a tuple |
| `ex.term.tuple_length` | `(tuple: i64) -> i64` | tuple arity; 0 for non-tuples |
| `ex.term.map_length` | `(map: i64) -> i64` | map pair count; 0 for non-maps |
| `ex.term.jmp_buf_size` | `() -> i64` | byte size of libc `jmp_buf`, for stack allocation in compiled code |
| `ex.term.setjmp_addr` | `() -> i64` | address of libc `setjmp`, for indirect calls that avoid ORC symbol resolution |
| `ex.term.try_push` | `(buf: ptr) -> i64` | push a setjmp buffer for a try region; -1 when the 16-slot stack is full |
| `ex.term.try_pop` | `() -> i64` | pop the innermost try region's buffer |
| `ex.term.throw` | `(value: i64) -> noreturn` | longjmp to the innermost try with a thrown term; aborts when uncaught |
| `ex.term.catch_value` | `() -> i64` | the term delivered by the most recent throw (read from the catch region) |
| `ex.term.list_head` | `(list: i64) -> i64` | head; nil for empty/non-lists |
| `ex.term.list_tail` | `(list: i64) -> i64` | tail; nil for empty/non-lists |
| `ex.term.list_length` | `(list: i64) -> i64` | list length; 0 for nil |
| `ex.term.eq` | `(left: i64, right: i64) -> i64` | deep equality: exact for immediates, structural for containers |
| `ex.term.binary_length` | `(binary: i64) -> i64` | byte length; 0 for non-binaries |
| `ex.term.binary_get` | `(binary: i64, index: i64) -> i64` | byte at index as a tagged int term; nil out of range / non-binary |
| `ex.term.binary_slice` | `(binary: i64, start: i64) -> i64` | materialized binary of bytes [start..len); nil for non-binaries / bad start |
| `ex.term.binary_utf8_get` | `(binary: i64, index: i64) -> i64` | UTF-8 codepoint at index as a tagged int term; nil for invalid/out-of-range |
| `ex.term.binary_utf8_width` | `(binary: i64, index: i64) -> i64` | UTF-8 codepoint byte width; 0 for invalid/out-of-range |
| `ex.term.map_from_list` | `(list: i64) -> i64` | flat key/value list -> map |
| `ex.term.binary_from_list` | `(list: i64) -> i64` | integer byte list -> binary |
| `ex.term.is_integer` | `(word: i64) -> i64` | 1 if int |
| `ex.term.is_atom` | `(word: i64) -> i64` | 1 if atom (incl. nil) |
| `ex.term.is_binary` | `(word: i64) -> i64` | 1 if binary |
| `ex.term.is_list` | `(word: i64) -> i64` | 1 if list (incl. `[]`) |
| `ex.term.is_tuple` | `(word: i64) -> i64` | 1 if tuple |
| `ex.term.is_map` | `(word: i64) -> i64` | 1 if map |

Predicates return `1` or `0` as an `i64`.

## Constraints

- Immediate integers carry a 61-bit payload; values beyond that truncate.
- `ex.term.map_from_list` requires an even-length list.
- `ex.term.binary_from_list` reads each segment's integer payload as a byte.
- `ex.term.try_push` overflows at 16 nested try regions; deeper nesting aborts.
- An uncaught `ex.term.throw` panics.
- The runtime owns a fixed bump arena; GC is a later milestone.
- The mailbox is a fixed 64-slot FIFO for a single actor; blocking receives
  and `after` timeouts arrive with the scheduler.

## Building

```sh
zig build-lib native/term_runtime.zig -dynamic -O ReleaseSafe -femit-bin=priv/term_runtime/libterm_runtime.so -lc
```

`-lc` links libc so the runtime can reference `setjmp`/`longjmp` for
`try`/`throw`; `Batata.TermRuntime.ensure_built!/0` passes it automatically.

`Batata.TermRuntime.ensure_built!/0` wraps this and is used by
`Batata.execute/2` (JIT) through `shared_lib_paths`.
