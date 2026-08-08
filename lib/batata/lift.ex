defmodule Batata.Lift do
  @moduledoc """
  Lifts a `Batata.Frontend` module snapshot into `ex` dialect IR.

  The scalar slice supports integer literals, `+`/`-`/`*`, `=` bindings, local
  calls, comparisons (`==`/`!=`/`<`/`<=`/`>`/`>=`), `case` with integer literal
  or catch-all patterns and optional guards, and functions with integer
  parameters. Multi-clause functions (single argument) dispatch on the
  argument with `ex.case`; the final clause must be a catch-all. The term
  slice adds tuple, list, map and binary literals plus the `Kernel` term
  predicates, and tail-recursive binary scanners (fixed-width head segments +
  rest, `± delta` accumulator) compile to `scf.while` cursor loops instead of
  recursion.
  (`is_atom`/`is_binary`/`is_list`/`is_tuple`/`is_map`/`is_integer`); they
  lift to the `ex.tuple`/`ex.list`/`ex.map`/`ex.binary`/`ex.is_*` ops and are
  lowered through the Zig term runtime ABI. Bindings lower directly to SSA:
  `ex.var`/`ex.bind` are term-universe bookkeeping and stay out of the typed
  scalar slice (they are erased by
  `Beaver.MLIR.Dialect.Ex.MaterializeBoundVariables` on the term path). Anything
  outside the slice raises `Batata.Lift.Error` explicitly instead of being
  silently dropped.
  """

  alias Batata.Frontend
  alias Batata.Transform.PatternPlan
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.Ex
  alias Beaver.Walker

  defmodule Error do
    @moduledoc "Raised when the frontend encounters an unsupported AST form."
    defexception [:message]
  end

  @doc """
  Builds a `builtin.module` of `ex.func` operations for the snapshot.

  Returns a `Beaver.Deferred`; materialize it with `Beaver.Deferred.create/2`
  against the MLIR context.
  """
  def module_to_ir(%Frontend.Module{} = mod, opts) do
    Beaver.Deferred.from_opts(opts, fn ctx ->
      unless ex_dialect_loaded?(ctx) do
        Beaver.Slang.load(ctx, Ex)
      end

      module = MLIR.Module.create!("module {}", ctx: ctx)
      body = MLIR.CAPI.mlirModuleGetBody(module)

      mod.definitions
      |> extract_all_fns()
      |> then(&append_dispatch(&1))
      |> Enum.group_by(&{&1.name, &1.arity})
      |> Enum.each(fn {_key, definitions} ->
        lift_definitions(definitions, ctx, body)
      end)

      module
    end)
  end

  # `Beaver.Slang.load/2` registers the IRDL ops in the context and is not
  # idempotent: loading the same dialect twice crashes MLIR with an operation
  # registration assertion. Skip it when the ex dialect is already present.
  defp ex_dialect_loaded?(ctx) do
    ctx
    |> MLIR.CAPI.mlirContextIsRegisteredOperation(MLIR.StringRef.create("ex.box"))
    |> Beaver.Native.to_term()
  end

  # Extracts anonymous-function literals from every definition body into
  # synthetic `defp` definitions, replacing the literal with a
  # `{:__fn_ref__, _, [fn_idx, name, arity, captured]}` marker. The synthetic
  # definition uses the fixed closure ABI: four captured-value slots followed
  # by four argument slots. The application site threads the captured values
  # from the outer env.
  defp extract_all_fns(definitions) do
    {defs, {synthetic, _counter}} =
      definitions
      |> Enum.map_reduce({[], 0}, fn defn, {synthetic, counter} ->
        {clauses, {synthetic, counter}} =
          defn.clauses
          |> Enum.map_reduce({synthetic, counter}, fn clause, {synthetic, counter} ->
            {body_ast, {synthetic, counter}} =
              extract_fns(clause.body_ast, defn.name, {synthetic, counter})

            {%{clause | body_ast: body_ast}, {synthetic, counter}}
          end)

        {%{defn | clauses: clauses}, {synthetic, counter}}
      end)

    defs ++ synthetic
  end

  defp extract_fns({:fn, _, [{:->, _, [args, body]}]}, parent, {synthetic, counter}) do
    name = :"__fn_#{parent}_#{counter}"
    arity = length(args)
    bound = Enum.map(args, &param_name/1)
    captured = body |> free_vars(bound) |> Enum.uniq() |> Enum.sort()

    unless arity <= 4 and length(captured) <= 4 do
      raise Error,
            "anonymous functions are limited to 4 arguments and 4 captured variables: " <>
              "#{arity} arguments, #{length(captured)} captured"
    end

    patterns =
      ((captured ++ List.duplicate(nil, 4 - length(captured))) ++
         bound ++ List.duplicate(nil, 4 - arity))
      |> Enum.map(fn
        nil -> {:_, [], nil}
        name -> {name, [], nil}
      end)

    fn_def = %Frontend.Definition{
      kind: :defp,
      name: name,
      arity: 8,
      clauses: [%Frontend.Clause{patterns: patterns, body_ast: body}]
    }

    marker = {:__fn_ref__, [], [counter, name, arity, captured]}
    {marker, {synthetic ++ [fn_def], counter + 1}}
  end

  defp extract_fns(tuple, parent, acc) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map_reduce(acc, &extract_fns(&1, parent, &2))
    |> then(fn {elements, acc} -> {List.to_tuple(elements), acc} end)
  end

  defp extract_fns([head | tail], parent, acc) do
    {head, acc} = extract_fns(head, parent, acc)
    {tail, acc} = extract_fns(tail, parent, acc)
    {[head | tail], acc}
  end

  defp extract_fns(other, _parent, acc), do: {other, acc}

  # Collects variable references in an AST that are not bound by `bound`.
  # Nested fn literals are skipped: their bodies bind and reference variables
  # in their own scope, and each literal is extracted independently.
  defp free_vars({:fn, _, _}, _bound), do: []

  defp free_vars({var, _, nil}, bound) when is_atom(var) do
    if var == :_ or var in bound, do: [], else: [var]
  end

  defp free_vars({{:., _, [fun]}, _, args}, bound) when is_list(args) do
    free_vars(fun, bound) ++ Enum.flat_map(args, &free_vars(&1, bound))
  end

  defp free_vars({_name, _, args}, bound) when is_list(args) do
    Enum.flat_map(args, &free_vars(&1, bound))
  end

  defp free_vars(tuple, bound) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.flat_map(&free_vars(&1, bound))
  end

  defp free_vars(list, bound) when is_list(list) do
    Enum.flat_map(list, &free_vars(&1, bound))
  end

  defp free_vars(_other, _bound), do: []

  # Appends the closure dispatch function: it reads the function index and
  # env words from a closure (via the Zig runtime) and jumps to the matching
  # `__fn_*` with the fixed 8-slot ABI. Built only when at least one
  # anonymous function exists.
  defp append_dispatch(definitions) do
    fns =
      definitions
      |> Enum.filter(&fn_definition?(&1))
      |> Enum.map(fn defn ->
        idx =
          defn.name
          |> Atom.to_string()
          |> String.split("_")
          |> List.last()
          |> String.to_integer()

        {idx, defn.name}
      end)
      |> Enum.sort()

    case fns do
      [] -> definitions
      _ -> definitions ++ [dispatch_definition(fns)]
    end
  end

  defp fn_definition?(%Frontend.Definition{name: name}) do
    name |> Atom.to_string() |> String.starts_with?("__fn_")
  end

  defp dispatch_definition([{_, first_name} | _] = fns) do
    vars = [:idx, :e0, :e1, :e2, :e3, :a0, :a1, :a2, :a3]
    call_args = Enum.map(tl(vars), &{&1, [], nil})
    zero_args = List.duplicate(0, 8)

    clauses =
      Enum.map(fns, fn {idx, name} ->
        {:->, [], [[idx], {name, [], call_args}]}
      end) ++ [{:->, [], [[{:_, [], nil}], {first_name, [], zero_args}]}]

    %Frontend.Definition{
      kind: :defp,
      name: :__fn_dispatch,
      arity: length(vars),
      clauses: [
        %Frontend.Clause{
          patterns: Enum.map(vars, &{&1, [], nil}),
          body_ast: {:case, [], [{:idx, [], nil}, [do: clauses]]}
        }
      ]
    }
  end

  defp lift_definition(
         %Frontend.Definition{kind: kind, name: name, arity: arity, clauses: clauses},
         ctx,
         ip
       ) do
    unless kind in [:def, :defp] do
      raise Error, "unsupported definition kind: #{inspect(kind)}"
    end

    unless length(clauses) == 1 do
      raise Error, "multiple clauses are unsupported in the scalar slice: #{name}/#{arity}"
    end

    [%Frontend.Clause{patterns: patterns, body_ast: body_ast}] = clauses

    region = MLIR.CAPI.mlirRegionCreate()
    arg_types = List.duplicate(integer_type(ctx), length(patterns))
    arg_locs = List.duplicate(MLIR.Location.unknown(ctx: ctx), length(patterns))
    block = MLIR.Block.create(arg_types, arg_locs)
    MLIR.CAPI.mlirRegionAppendOwnedBlock(region, block)

    env =
      block
      |> Walker.arguments()
      |> Enum.to_list()
      |> Enum.zip(patterns)
      |> Enum.reduce(%{}, fn {value, pattern}, env ->
        Map.put(env, param_name(pattern), value)
      end)

    # The entry function starts a fresh actor: reset the mailbox so each
    # program run observes an empty message queue.
    if name == :main and uses_mailbox?(body_ast) do
      create_op("ex.mailbox_clear", [], [ex_type("dyn", ctx)], ctx, block)
    end

    {return_value, env} = lift_block(List.wrap(body_ast), ctx, block, env)
    insert_return(return_value, ctx, block, env)

    %Beaver.SSA{
      op: "ex.func",
      ip: ip,
      ctx: ctx,
      arguments: [sym_name: MLIR.Attribute.string(to_string(name))],
      results: [],
      filler: fn -> [region] end
    }
    |> MLIR.Operation.create()
  end

  defp uses_mailbox?(ast) do
    ast
    |> Macro.prewalk(false, fn
      node, true ->
        {node, true}

      {:receive, _, _}, _ ->
        {nil, true}

      {:send, _, _}, _ ->
        {nil, true}

      {:self, _, []}, _ ->
        {nil, true}

      node, acc ->
        {node, acc}
    end)
    |> elem(1)
  end

  defp lift_definitions([definition], ctx, ip) do
    lift_definition(definition, ctx, ip)
  end

  # Multiple `def` forms with the same name/arity become one ex.func whose
  # body dispatches on the argument with ex.case, matching each clause's
  # pattern (the cursor-loop foundation for recursive scanners). M2 requires
  # a single argument and a final catch-all clause.
  defp lift_definitions(definitions, ctx, ip) do
    %Frontend.Definition{kind: kind, name: name, arity: arity} = hd(definitions)

    unless kind in [:def, :defp] do
      raise Error, "unsupported definition kind: #{inspect(kind)}"
    end

    clauses = Enum.flat_map(definitions, & &1.clauses)

    cond do
      arity == 1 ->
        case detect_scanner(name, clauses) do
          {:ok, scanner} -> lift_scanner_loop(name, scanner, ctx, ip)
          :skip -> lift_multi_clause_dispatch(name, clauses, ctx, ip)
        end

      arity >= 2 ->
        case detect_accumulator_scanner(name, clauses) do
          {:ok, scanner} -> lift_reduce_loop(name, scanner, ctx, ip)
          :skip -> lift_multi_arg_dispatch(name, arity, clauses, ctx, ip)
        end

      true ->
        raise Error, "unsupported function arity: #{name}/#{arity}"
    end
  end

  defp lift_multi_clause_dispatch(name, clauses, ctx, ip) do
    region = MLIR.CAPI.mlirRegionCreate()

    # The argument is a scalar word (like single-clause functions); the term
    # path re-types it with ex.to_word when term reads are involved.
    arg_locs = [MLIR.Location.unknown(ctx: ctx)]
    block = MLIR.Block.create([integer_type(ctx)], arg_locs)
    MLIR.CAPI.mlirRegionAppendOwnedBlock(region, block)
    [arg] = block |> Walker.arguments() |> Enum.to_list()

    clause_asts =
      Enum.map(clauses, fn %Frontend.Clause{patterns: [pattern], body_ast: body_ast} ->
        {:->, [], [[pattern], body_ast]}
      end)

    return_value =
      lift_case(clause_asts, arg, %{}, ctx, block, relax_types: true, box_scrutinee: false)

    insert_return(return_value, ctx, block, %{})

    %Beaver.SSA{
      op: "ex.func",
      ip: ip,
      ctx: ctx,
      arguments: [sym_name: MLIR.Attribute.string(to_string(name))],
      results: [],
      filler: fn -> [region] end
    }
    |> MLIR.Operation.create()
  end

  # Multi-argument multi-clause functions (e.g. `reduce(binary, acc)`): the
  # first argument dispatches with `ex.case`; the trailing arguments must be
  # bound as variables and are threaded through the clause environments.
  defp lift_multi_arg_dispatch(name, arity, clauses, ctx, ip) do
    tail_names = validate_multi_arg_clauses!(arity, clauses)

    region = MLIR.CAPI.mlirRegionCreate()
    loc = MLIR.Location.unknown(ctx: ctx)
    i64 = integer_type(ctx)
    locs = List.duplicate(loc, arity)

    block = MLIR.Block.create(List.duplicate(i64, arity), locs)
    MLIR.CAPI.mlirRegionAppendOwnedBlock(region, block)
    [arg1 | tail_args] = block |> Walker.arguments() |> Enum.to_list()
    tail_env = Map.new(Enum.zip(tail_names, tail_args))

    clause_asts =
      Enum.map(clauses, fn %Frontend.Clause{patterns: [first | _], body_ast: body_ast} ->
        {:->, [], [[first], body_ast]}
      end)

    return_value =
      lift_case(clause_asts, arg1, tail_env, ctx, block, relax_types: true, box_scrutinee: false)

    insert_return(return_value, ctx, block, tail_env)

    %Beaver.SSA{
      op: "ex.func",
      ip: ip,
      ctx: ctx,
      arguments: [sym_name: MLIR.Attribute.string(to_string(name))],
      results: [],
      filler: fn -> [region] end
    }
    |> MLIR.Operation.create()
  end

  defp validate_multi_arg_clauses!(arity, clauses) do
    tail_names =
      clauses
      |> Enum.map(fn %Frontend.Clause{patterns: patterns} ->
        unless length(patterns) == arity do
          raise Error, "clause arity mismatch for a multi-clause function"
        end

        {_first, tails} = Enum.split(patterns, 1)

        Enum.map(tails, fn
          {name, _, nil} when is_atom(name) and name != :_ ->
            name

          other ->
            raise Error, "multi-clause trailing arguments must be variables: #{inspect(other)}"
        end)
      end)
      |> Enum.uniq()

    case tail_names do
      [names] ->
        names

      _ ->
        raise Error,
              "multi-clause multi-argument functions must use the same trailing argument names"
    end
  end

  # Accumulator-scanner detection (the `reduce(binary, acc)` shape): a
  # two-argument function whose recursive clause matches fixed-width binary
  # head segments plus a rest and calls itself with the rest slice and an
  # accumulator step (`acc + delta`), and whose other clauses return the
  # accumulator unchanged.
  defp detect_accumulator_scanner(name, clauses) do
    parsed = Enum.map(clauses, &accumulator_scanner_clause(&1, name))

    with [%{delta: delta, head_width: width}] <-
           Enum.filter(parsed, &match?(%{kind: :recursive}, &1)),
         {:ok, acc_name} <- common_acc_name(parsed),
         true <- terminating_returns_acc?(parsed, acc_name) do
      {:ok, %{delta: delta, head_width: width, acc_name: acc_name}}
    else
      _ -> :skip
    end
  end

  defp accumulator_scanner_clause(
         %Frontend.Clause{patterns: [p1, acc_pat], body_ast: body_ast},
         name
       ) do
    case binary_segments(p1) do
      {:ok, width, rest} ->
        case reduce_accumulator(body_ast, name, rest, acc_pat) do
          {:ok, delta} -> %{kind: :recursive, delta: delta, head_width: width, acc: acc_pat}
          :skip -> %{kind: :terminating, body: body_ast, acc: acc_pat}
        end

      :skip ->
        %{kind: :terminating, body: body_ast, acc: acc_pat}
    end
  end

  defp reduce_accumulator({name, _, [var_ast, acc_expr]}, name, rest, acc_pat)
       when is_atom(name) do
    if var_name(var_ast) == var_name(rest) do
      acc_step(acc_expr, acc_pat)
    else
      :skip
    end
  end

  defp reduce_accumulator(_body_ast, _name, _rest, _acc_pat), do: :skip

  defp acc_step({acc, _, nil} = acc_ast, acc_pat) when is_atom(acc) do
    if var_name(acc_ast) == var_name(acc_pat), do: {:ok, 0}, else: :skip
  end

  defp acc_step({:+, _, [acc_ast, delta]}, acc_pat) when is_integer(delta) do
    if var_name(acc_ast) == var_name(acc_pat), do: {:ok, delta}, else: :skip
  end

  defp acc_step({:+, _, [delta, acc_ast]}, acc_pat) when is_integer(delta) do
    if var_name(acc_ast) == var_name(acc_pat), do: {:ok, delta}, else: :skip
  end

  defp acc_step({:-, _, [acc_ast, delta]}, acc_pat) when is_integer(delta) do
    if var_name(acc_ast) == var_name(acc_pat), do: {:ok, -delta}, else: :skip
  end

  defp acc_step(_acc_expr, _acc_pat), do: :skip

  defp common_acc_name(parsed) do
    case parsed |> Enum.map(&var_name(&1.acc)) |> Enum.uniq() do
      [acc_name] when is_atom(acc_name) -> {:ok, acc_name}
      _ -> :skip
    end
  end

  defp terminating_returns_acc?(parsed, acc_name) do
    parsed
    |> Enum.reject(&match?(%{kind: :recursive}, &1))
    |> Enum.all?(fn clause -> var_name(clause.body) == acc_name end)
  end

  # Cursor-loop optimization (expandable d95fd36/f62b38b route): a
  # tail-recursive binary scanner — one clause whose pattern is fixed-width
  # binary segments plus a rest, whose body accumulates `± delta` around the
  # self call, and whose other clauses return a common constant base —
  # compiles to a cf loop over the original binary with a cursor and
  # accumulator, avoiding per-step slice materialization and call overhead.
  defp detect_scanner(name, clauses) do
    parsed =
      Enum.map(clauses, fn clause ->
        scanner_clause(clause, name)
      end)

    with [%{delta: delta, head_width: width}] <-
           Enum.filter(parsed, &match?(%{kind: :recursive}, &1)),
         {:ok, base} <- common_base(parsed) do
      {:ok, %{base: base, delta: delta, head_width: width}}
    else
      _ -> :skip
    end
  end

  defp scanner_clause(%Frontend.Clause{patterns: [pattern], body_ast: body_ast}, name) do
    case binary_segments(pattern) do
      {:ok, _width, nil} ->
        %{kind: :terminating, body: body_ast}

      {:ok, width, rest} ->
        case accumulator(body_ast, name, rest) do
          {:ok, delta} -> %{kind: :recursive, delta: delta, head_width: width}
          :skip -> %{kind: :terminating, body: body_ast}
        end

      :skip ->
        %{kind: :terminating, body: body_ast}
    end
  end

  defp binary_segments({:<<>>, _, segments}) do
    {bytes, rest} =
      Enum.split_while(segments, &(not match?({:"::", _, [_, {:binary, _, nil}]}, &1)))

    if Enum.all?(bytes, &byte_segment?/1) do
      case rest do
        [] ->
          {:ok, length(bytes), nil}

        [{:"::", _, [rest_pat, {:binary, _, nil}]}] ->
          case rest_pat do
            {name, _, nil} when is_atom(name) and name != :_ -> {:ok, length(bytes), rest_pat}
            _ -> :skip
          end

        _ ->
          :skip
      end
    else
      :skip
    end
  end

  defp binary_segments(_), do: :skip

  defp byte_segment?({:"::", _, [_, 8]}), do: true
  defp byte_segment?(pat) when is_integer(pat), do: true
  defp byte_segment?({_, _, nil}), do: true
  defp byte_segment?(_), do: false

  # `count(t)`, `delta + count(t)`, `count(t) + delta`, `count(t) - delta`
  # where `t` is the rest-segment bind.
  defp accumulator({name, _, [var_ast]}, name, rest) when is_atom(name) do
    if var_name(var_ast) == var_name(rest), do: {:ok, 0}, else: :skip
  end

  defp accumulator({:+, _, [delta, {name, _, [var_ast]}]}, name, rest)
       when is_integer(delta) and is_atom(name) do
    if var_name(var_ast) == var_name(rest), do: {:ok, delta}, else: :skip
  end

  defp accumulator({:+, _, [{name, _, [var_ast]}, delta]}, name, rest)
       when is_integer(delta) and is_atom(name) do
    if var_name(var_ast) == var_name(rest), do: {:ok, delta}, else: :skip
  end

  defp accumulator({:-, _, [{name, _, [var_ast]}, delta]}, name, rest)
       when is_integer(delta) and is_atom(name) do
    if var_name(var_ast) == var_name(rest), do: {:ok, -delta}, else: :skip
  end

  defp accumulator(_body_ast, _name, _rest), do: :skip

  defp var_name({name, _, nil}) when is_atom(name), do: name
  defp var_name(_), do: nil

  defp common_base(parsed) do
    bases =
      parsed
      |> Enum.reject(&match?(%{kind: :recursive}, &1))
      |> Enum.map(&terminator_base(&1.body))

    case Enum.uniq(bases) do
      [base] when is_integer(base) -> {:ok, base}
      _ -> :skip
    end
  end

  defp terminator_base(body) when is_integer(body), do: body
  defp terminator_base(_body), do: :skip

  defp lift_scanner_loop(name, %{base: base, delta: delta, head_width: width}, ctx, ip) do
    region = MLIR.CAPI.mlirRegionCreate()
    loc = MLIR.Location.unknown(ctx: ctx)
    i64 = integer_type(ctx)

    block = MLIR.Block.create([i64], [loc])
    MLIR.CAPI.mlirRegionAppendOwnedBlock(region, block)
    [arg] = block |> Walker.arguments() |> Enum.to_list()

    base_val = lit(base, ctx, block)
    acc_result = emit_cursor_while(block, arg, base_val, width, delta, ctx)
    create_op("ex.return", [acc_result, operandSegmentSizes: segment_sizes([1])], [], ctx, block)

    %Beaver.SSA{
      op: "ex.func",
      ip: ip,
      ctx: ctx,
      arguments: [sym_name: MLIR.Attribute.string(to_string(name))],
      results: [],
      filler: fn -> [region] end
    }
    |> MLIR.Operation.create()
  end

  defp lift_reduce_loop(name, %{delta: delta, head_width: width}, ctx, ip) do
    region = MLIR.CAPI.mlirRegionCreate()
    loc = MLIR.Location.unknown(ctx: ctx)

    block = MLIR.Block.create([integer_type(ctx), integer_type(ctx)], [loc, loc])
    MLIR.CAPI.mlirRegionAppendOwnedBlock(region, block)
    [arg, acc0] = block |> Walker.arguments() |> Enum.to_list()

    acc_result = emit_cursor_while(block, arg, acc0, width, delta, ctx)
    create_op("ex.return", [acc_result, operandSegmentSizes: segment_sizes([1])], [], ctx, block)

    %Beaver.SSA{
      op: "ex.func",
      ip: ip,
      ctx: ctx,
      arguments: [sym_name: MLIR.Attribute.string(to_string(name))],
      results: [],
      filler: fn -> [region] end
    }
    |> MLIR.Operation.create()
  end

  # scf.while keeps the ex.func body to a single block: the before region
  # carries (arg, acc, cursor) and conditions on the next head segment
  # existing; the after region advances the accumulator and cursor.
  defp emit_cursor_while(block, arg, acc, width, delta, ctx) do
    i64 = integer_type(ctx)

    locs = [
      MLIR.Location.unknown(ctx: ctx),
      MLIR.Location.unknown(ctx: ctx),
      MLIR.Location.unknown(ctx: ctx)
    ]

    before = MLIR.CAPI.mlirRegionCreate()
    before_block = MLIR.Block.create([i64, i64, i64], locs)
    MLIR.CAPI.mlirRegionAppendOwnedBlock(before, before_block)

    after_region = MLIR.CAPI.mlirRegionCreate()
    after_block = MLIR.Block.create([i64, i64, i64], locs)
    MLIR.CAPI.mlirRegionAppendOwnedBlock(after_region, after_block)

    [b_arg, b_acc, b_cursor] = before_block |> Walker.arguments() |> Enum.to_list()
    word = create_op("ex.to_word", [b_arg], [ex_type("dyn", ctx)], ctx, before_block)
    len = create_op("ex.binary_length", [word], [i64], ctx, before_block)

    next_cursor =
      create_op("ex.add", [b_cursor, lit(width, ctx, before_block)], [i64], ctx, before_block)

    cond = cmp(len, next_cursor, "sge", ctx, before_block)
    cond_i1 = create_op("arith.trunci", [cond], [MLIR.Type.i1()], ctx, before_block)
    create_op("scf.condition", [cond_i1, b_arg, b_acc, b_cursor], [], ctx, before_block)

    [a_arg, a_acc, a_cursor] = after_block |> Walker.arguments() |> Enum.to_list()
    acc_next = create_op("ex.add", [a_acc, lit(delta, ctx, after_block)], [i64], ctx, after_block)

    cursor_next =
      create_op("ex.add", [a_cursor, lit(width, ctx, after_block)], [i64], ctx, after_block)

    create_op("scf.yield", [a_arg, acc_next, cursor_next], [], ctx, after_block)

    cursor0 = lit(0, ctx, block)

    while_op =
      %Beaver.SSA{
        op: "scf.while",
        ip: block,
        ctx: ctx,
        arguments: [arg, acc, cursor0],
        results: [i64, i64, i64],
        loc: MLIR.Location.unknown(),
        filler: fn -> [before, after_region] end
      }
      |> MLIR.Operation.create()

    while_op |> MLIR.Operation.results() |> Enum.to_list() |> Enum.at(1)
  end

  defp lift_block(expressions, ctx, block, env) do
    {values, env} =
      Enum.map_reduce(expressions, env, fn expression, env ->
        lift_expr(expression, ctx, block, env)
      end)

    {List.last(values), env}
  end

  defp lift_expr(integer, ctx, block, env) when is_integer(integer) do
    {
      create_op(
        "ex.lit",
        [value: MLIR.Attribute.integer(MLIR.Type.i64(), integer)],
        [MLIR.Type.i64()],
        ctx,
        block
      ),
      env
    }
  end

  defp lift_expr({:+, _, [left, right]}, ctx, block, env) do
    {left_value, env} = lift_expr(left, ctx, block, env)
    {right_value, env} = lift_expr(right, ctx, block, env)

    {
      create_op("ex.add", [left_value, right_value], [MLIR.Type.i64()], ctx, block),
      env
    }
  end

  defp lift_expr({:-, _, [left, right]}, ctx, block, env) do
    {left_value, env} = lift_expr(left, ctx, block, env)
    {right_value, env} = lift_expr(right, ctx, block, env)

    {
      create_op("ex.sub", [left_value, right_value], [MLIR.Type.i64()], ctx, block),
      env
    }
  end

  defp lift_expr({:*, _, [left, right]}, ctx, block, env) do
    {left_value, env} = lift_expr(left, ctx, block, env)
    {right_value, env} = lift_expr(right, ctx, block, env)

    {
      create_op("ex.mul", [left_value, right_value], [MLIR.Type.i64()], ctx, block),
      env
    }
  end

  defp lift_expr(binary, ctx, block, env) when is_binary(binary) do
    {values, env} =
      binary
      |> :binary.bin_to_list()
      |> Enum.map_reduce(env, fn byte, env ->
        {value, env} = lift_expr(byte, ctx, block, env)
        {box_term(value, ctx, block), env}
      end)

    {create_term_op("ex.binary", values, ctx, block), env}
  end

  defp lift_expr([], ctx, block, env) do
    {create_term_op("ex.list", [], ctx, block), env}
  end

  defp lift_expr(elements, ctx, block, env) when is_list(elements) do
    {values, env} = lift_operands_boxed(elements, ctx, block, env)
    {create_term_op("ex.list", values, ctx, block), env}
  end

  defp lift_expr({:%{}, _, entries}, ctx, block, env) do
    {values, env} = lift_map_entries(entries, ctx, block, env)
    {create_term_op("ex.map", values, ctx, block), env}
  end

  defp lift_expr({:<<>>, _, segments}, ctx, block, env) do
    {values, env} = lift_operands_boxed(segments, ctx, block, env)
    {create_term_op("ex.binary", values, ctx, block), env}
  end

  defp lift_expr({name, _, [arg]}, ctx, block, env)
       when name in [:is_atom, :is_binary, :is_list, :is_tuple, :is_map, :is_integer] do
    {value, env} = lift_expr(arg, ctx, block, env)
    {create_op("ex.#{name}", [box_term(value, ctx, block)], [MLIR.Type.i64()], ctx, block), env}
  end

  defp lift_expr({op, _, [left, right]}, ctx, block, env)
       when op in [:==, :!=, :<, :<=, :>, :>=] do
    {left_value, env} = lift_expr(left, ctx, block, env)
    {right_value, env} = lift_expr(right, ctx, block, env)

    if term_operand?(left_value) or term_operand?(right_value) do
      unless op in [:==, :!=] do
        raise Error, "ordering comparisons on terms are unsupported: #{inspect(op)}"
      end

      eq =
        create_op(
          "ex.term_eq",
          [box_if_scalar(left_value, ctx, block), box_if_scalar(right_value, ctx, block)],
          [MLIR.Type.i64()],
          ctx,
          block
        )

      if op == :== do
        {eq, env}
      else
        {create_op(
           "ex.cmp",
           [eq, lit(0, ctx, block), predicate: MLIR.Attribute.string("eq")],
           [MLIR.Type.i64()],
           ctx,
           block
         ), env}
      end
    else
      {
        create_op(
          "ex.cmp",
          [left_value, right_value, predicate: MLIR.Attribute.string(cmp_predicate(op))],
          [MLIR.Type.i64()],
          ctx,
          block
        ),
        env
      }
    end
  end

  defp lift_expr({:case, _, [scrutinee_ast, [do: clauses]]}, ctx, block, env) do
    {scrutinee, env} = lift_expr(scrutinee_ast, ctx, block, env)
    {lift_case(clauses, scrutinee, env, ctx, block), env}
  end

  defp lift_expr({:__block__, _, expressions}, ctx, block, env) do
    lift_block(expressions, ctx, block, env)
  end

  defp lift_expr({:=, _, [{var, _, nil}, rhs]}, ctx, block, env) when is_atom(var) do
    {value, env} = lift_expr(rhs, ctx, block, env)
    {value, Map.put(env, var, value)}
  end

  # Anonymous-function marker produced by `extract_all_fns/1`: the literal
  # becomes a compile-time function reference. It is materialized into a
  # first-class closure word only when it crosses into a value context; a
  # direct `.()` application calls the extracted ex.func directly.
  defp lift_expr({:__fn_ref__, _, [fn_idx, name, arity, captured]}, _ctx, _block, env) do
    {{:fn_ref, fn_idx, name, arity, captured}, env}
  end

  # Anonymous-function application: `f.(args)` / `(fn ... end).(args)`.
  defp lift_expr({{:., _, [fun_ast]}, _, args}, ctx, block, env) do
    case resolve_fun_ref(fun_ast, env) do
      {:ok, _fn_idx, name, arity, captured} ->
        unless length(args) == arity do
          raise Error,
                "anonymous function application arity mismatch: expected #{arity}, got #{length(args)}"
        end

        {arg_values, env} =
          Enum.map_reduce(args, env, fn arg, env ->
            lift_expr(arg, ctx, block, env)
          end)

        captured_values = resolve_captured(captured, env)
        captured_values = Enum.map(captured_values, &lift_value(&1, ctx, block, env))
        arg_values = Enum.map(arg_values, &lift_value(&1, ctx, block, env))

        # The extracted fn uses the fixed 8-slot closure ABI: four captured
        # slots followed by four argument slots.
        call_args =
          captured_values ++
            List.duplicate(zero_i64(ctx, block), 4 - length(captured_values)) ++
            arg_values ++ List.duplicate(zero_i64(ctx, block), 4 - length(arg_values))

        {
          create_op(
            "ex.call",
            call_args ++
              [
                callee: MLIR.Attribute.string(to_string(name)),
                arity: MLIR.Attribute.integer(MLIR.Type.i64(), 8),
                operandSegmentSizes: segment_sizes(arg_segment_sizes(8))
              ],
            [ex_type("dyn", ctx)],
            ctx,
            block
          ),
          env
        }

      {:dynamic, closure} ->
        unless length(args) <= 4 do
          raise Error,
                "dynamic anonymous function application supports at most 4 arguments, got #{length(args)}"
        end

        {arg_values, env} =
          Enum.map_reduce(args, env, fn arg, env ->
            lift_expr(arg, ctx, block, env)
          end)

        closure_word = create_op("ex.to_word", [closure], [ex_type("dyn", ctx)], ctx, block)

        {
          create_op(
            "ex.apply",
            [closure_word] ++
              arg_values ++
              [
                arg_count: MLIR.Attribute.integer(MLIR.Type.i64(), length(args)),
                operandSegmentSizes:
                  segment_sizes(
                    [1 | List.duplicate(1, length(args))] ++
                      List.duplicate(0, 4 - length(args))
                  )
              ],
            [ex_type("dyn", ctx)],
            ctx,
            block
          ),
          env
        }

      :error ->
        raise Error,
              "anonymous function application requires a fn literal or a bound function: " <>
                inspect(fun_ast)
    end
  end

  # Remote stdlib call: `Kernel.length(x)` / `List.first(x)` / `Enum.count(x)`.
  # Module-qualified calls resolve through the stdlib domain registry; anything
  # outside the declared surface raises explicitly.
  defp lift_expr({{:., _, [mod_ast, fun]}, _, args}, ctx, block, env)
       when is_atom(fun) and is_list(args) do
    case module_ref(mod_ast) do
      {:ok, module} ->
        lift_stdlib_call(module, fun, args, ctx, block, env)

      :error ->
        raise Error, "unsupported AST in the current slice: #{inspect(mod_ast)}.#{fun}"
    end
  end

  defp lift_expr({:self, _, []}, ctx, block, env) do
    {create_op("ex.self", [], [ex_type("dyn", ctx)], ctx, block), env}
  end

  defp lift_expr({:send, _, [pid_ast, msg_ast]}, ctx, block, env) do
    {pid_value, env} = lift_expr(pid_ast, ctx, block, env)
    {msg_value, env} = lift_expr(msg_ast, ctx, block, env)

    pid_word = box_term(lift_value(pid_value, ctx, block, env), ctx, block)
    msg_word = box_term(lift_value(msg_value, ctx, block, env), ctx, block)

    {
      create_op("ex.send", [pid_word, msg_word], [ex_type("dyn", ctx)], ctx, block),
      env
    }
  end

  # `receive do pattern -> body end`: pops one message from the actor
  # mailbox and matches it with a term case. Empty or non-matching messages
  # fall through to a catch-all that returns the popped word; `after`
  # clauses are a later milestone.
  defp lift_expr({:receive, _, [options]}, ctx, block, env) do
    if Keyword.has_key?(options, :after) do
      raise Error, "receive after clauses are unsupported in the current slice"
    end

    clauses = Keyword.fetch!(options, :do)
    clauses = ensure_receive_catch_all(clauses)
    msg = create_op("ex.receive", [], [ex_type("dyn", ctx)], ctx, block)
    {lift_term_case(clauses, msg, env, ctx, block, untag_int_binds: true), env}
  end

  defp lift_expr({:throw, _, [value_ast]}, ctx, block, env) do
    {value, env} = lift_expr(value_ast, ctx, block, env)
    value = box_term(lift_value(value, ctx, block, env), ctx, block)
    {create_op("ex.throw", [value], [ex_type("dyn", ctx)], ctx, block), env}
  end

  # `try do body catch pattern -> handler end`: the body region runs normally;
  # a `throw` longjmps back and the catch region matches the thrown value.
  defp lift_expr({:try, _, [options]}, ctx, block, env) do
    if Enum.any?([:rescue, :after, :else], &Keyword.has_key?(options, &1)) do
      raise Error, "only try/catch is supported in the current slice"
    end

    body = Keyword.fetch!(options, :do)
    catch_clauses = Keyword.fetch!(options, :catch) |> ensure_receive_catch_all()

    body_region = MLIR.CAPI.mlirRegionCreate()
    body_block = MLIR.Block.create([], [])
    MLIR.CAPI.mlirRegionAppendOwnedBlock(body_region, body_block)

    {body_value, body_env} = lift_block(List.wrap(body), ctx, body_block, env)
    body_value = lift_value(body_value, ctx, body_block, body_env)

    create_op(
      "ex.yield",
      [body_value, operandSegmentSizes: segment_sizes([1])],
      [],
      ctx,
      body_block
    )

    catch_region = MLIR.CAPI.mlirRegionCreate()
    catch_block = MLIR.Block.create([], [])
    MLIR.CAPI.mlirRegionAppendOwnedBlock(catch_region, catch_block)

    thrown = create_op("ex.catch_value", [], [ex_type("dyn", ctx)], ctx, catch_block)

    catch_value =
      lift_term_case(catch_clauses, thrown, env, ctx, catch_block, untag_int_binds: true)

    create_op(
      "ex.yield",
      [catch_value, operandSegmentSizes: segment_sizes([1])],
      [],
      ctx,
      catch_block
    )

    try_op =
      %Beaver.SSA{
        op: "ex.try",
        ip: block,
        ctx: ctx,
        results: [ex_type("dyn", ctx)],
        loc: MLIR.Location.unknown(),
        filler: fn -> [body_region, catch_region] end
      }
      |> MLIR.Operation.create()

    {try_op |> MLIR.Operation.results() |> Enum.to_list() |> hd(), env}
  end

  defp lift_expr({name, _, args}, ctx, block, env) when is_atom(name) and is_list(args) do
    if Batata.Stdlib.class({Kernel, name, length(args)}) == :native_term do
      # Kernel auto-imported BIFs (length/1, hd/1, ...) resolve through the
      # stdlib registry; user definitions of the same name are not visible in
      # this slice, matching the existing self/0 and send/2 special cases.
      lift_stdlib_call(Kernel, name, args, ctx, block, env)
    else
      {arg_values, env} =
        Enum.map_reduce(args, env, fn arg, env ->
          {value, env} = lift_expr(arg, ctx, block, env)
          {lift_value(value, ctx, block, env), env}
        end)

      {
        create_op(
          "ex.call",
          arg_values ++
            [
              callee: MLIR.Attribute.string(to_string(name)),
              arity: MLIR.Attribute.integer(MLIR.Type.i64(), length(args)),
              operandSegmentSizes: segment_sizes(arg_segment_sizes(length(args)))
            ],
          [ex_type("dyn", ctx)],
          ctx,
          block
        ),
        env
      }
    end
  end

  defp lift_expr({name, _, nil}, _ctx, _block, env) when is_atom(name) do
    case Map.fetch(env, name) do
      {:ok, value} -> {value, env}
      :error -> raise Error, "unbound variable reference: #{inspect(name)}"
    end
  end

  # Tuple literals: calls, operators and variables are 3-tuples in the AST and
  # are handled above, so every other tuple shape is a literal tuple.
  defp lift_expr(tuple, ctx, block, env) when is_tuple(tuple) and tuple_size(tuple) != 3 do
    lift_tuple_literal(tuple, ctx, block, env)
  end

  defp lift_expr({a, b, c}, ctx, block, env)
       when not (is_atom(a) and is_list(b) and is_list(c)) do
    lift_tuple_literal({a, b, c}, ctx, block, env)
  end

  defp lift_expr(ast, _ctx, _block, _env) do
    raise Error, "unsupported AST in the current slice: #{inspect(ast)}"
  end

  defp ensure_receive_catch_all(clauses) do
    if catch_all_clause?(List.last(clauses)) do
      clauses
    else
      clauses ++ [{:->, [], [[{:_, [], nil}], 0]}]
    end
  end

  defp catch_all_clause?({:->, _, [[pattern], _body]}) do
    match?({name, _, nil} when is_atom(name), pattern)
  end

  defp catch_all_clause?(_clause), do: false

  defp module_ref({:__aliases__, _, [module]}) when is_atom(module),
    do: {:ok, Module.concat([module])}

  defp module_ref({:__aliases__, _, [:"Elixir", module]}) when is_atom(module),
    do: {:ok, Module.concat([:"Elixir", module])}

  defp module_ref(module) when is_atom(module), do: {:ok, module}
  defp module_ref(_), do: :error

  # Resolves a module-qualified stdlib call through the domain registry.
  defp lift_stdlib_call(module, fun, args, ctx, block, env) do
    case Batata.Stdlib.class({module, fun, length(args)}) do
      :native_term ->
        {values, env} = lift_operands_boxed(args, ctx, block, env)
        {native_term_call(module, fun, values, ctx, block), env}

      :beamer_callback ->
        raise Error,
              "stdlib call #{inspect(module)}.#{fun}/#{length(args)} requires BEAM callback " <>
                "interop (protocol consolidation), not yet supported"

      :unsupported ->
        raise Error,
              "stdlib call #{inspect(module)}.#{fun}/#{length(args)} is declared but not yet " <>
                "supported in this slice"

      nil ->
        raise Error,
              "unsupported stdlib call: #{inspect(module)}.#{fun}/#{length(args)}"
    end
  end

  # Lowering for `:native_term` registry entries: operands arrive boxed as
  # `!ex.dyn` words, results are either scalar i64 or `!ex.dyn`.
  defp native_term_call(_module, :length, [value], ctx, block),
    do: create_op("ex.list_length", [value], [MLIR.Type.i64()], ctx, block)

  defp native_term_call(_module, :hd, [value], ctx, block),
    do: create_op("ex.list_head", [value], [ex_type("dyn", ctx)], ctx, block)

  defp native_term_call(_module, :tl, [value], ctx, block),
    do: create_op("ex.list_tail", [value], [ex_type("dyn", ctx)], ctx, block)

  defp native_term_call(_module, :tuple_size, [value], ctx, block),
    do: create_op("ex.tuple_length", [value], [MLIR.Type.i64()], ctx, block)

  defp native_term_call(Map, :size, [value], ctx, block),
    do: create_op("ex.map_length", [value], [MLIR.Type.i64()], ctx, block)

  defp native_term_call(Tuple, :size, [value], ctx, block),
    do: create_op("ex.tuple_length", [value], [MLIR.Type.i64()], ctx, block)

  defp native_term_call(_module, :byte_size, [value], ctx, block),
    do: create_op("ex.binary_length", [value], [MLIR.Type.i64()], ctx, block)

  defp native_term_call(_module, :map_size, [value], ctx, block),
    do: create_op("ex.map_length", [value], [MLIR.Type.i64()], ctx, block)

  defp native_term_call(_module, :elem, [tuple, index], ctx, block) do
    index_int = create_op("ex.to_int", [index], [MLIR.Type.i64()], ctx, block)
    index0 = create_op("ex.sub", [index_int, lit(1, ctx, block)], [MLIR.Type.i64()], ctx, block)
    create_op("ex.tuple_get", [tuple, index0], [ex_type("dyn", ctx)], ctx, block)
  end

  defp native_term_call(_module, :is_atom, [value], ctx, block),
    do: create_op("ex.is_atom", [value], [MLIR.Type.i64()], ctx, block)

  defp native_term_call(_module, :is_binary, [value], ctx, block),
    do: create_op("ex.is_binary", [value], [MLIR.Type.i64()], ctx, block)

  defp native_term_call(_module, :is_integer, [value], ctx, block),
    do: create_op("ex.is_integer", [value], [MLIR.Type.i64()], ctx, block)

  defp native_term_call(_module, :is_list, [value], ctx, block),
    do: create_op("ex.is_list", [value], [MLIR.Type.i64()], ctx, block)

  defp native_term_call(_module, :is_map, [value], ctx, block),
    do: create_op("ex.is_map", [value], [MLIR.Type.i64()], ctx, block)

  defp native_term_call(_module, :is_tuple, [value], ctx, block),
    do: create_op("ex.is_tuple", [value], [MLIR.Type.i64()], ctx, block)

  defp native_term_call(_module, :first, [value], ctx, block),
    do: create_op("ex.list_head", [value], [ex_type("dyn", ctx)], ctx, block)

  defp native_term_call(_module, :self, [], ctx, block),
    do: create_op("ex.self", [], [ex_type("dyn", ctx)], ctx, block)

  defp native_term_call(_module, :send, [pid, msg], ctx, block),
    do: create_op("ex.send", [pid, msg], [ex_type("dyn", ctx)], ctx, block)

  defp native_term_call(module, fun, _args, _ctx, _block) do
    raise Error, "no native_term lowering for #{inspect(module)}.#{fun}"
  end

  defp resolve_fun_ref({name, _, nil}, env) when is_atom(name) do
    case Map.get(env, name) do
      {:fn_ref, fn_idx, fn_name, arity, captured} -> {:ok, fn_idx, fn_name, arity, captured}
      nil -> :error
      value -> {:dynamic, value}
    end
  end

  defp resolve_fun_ref({:__fn_ref__, _, [fn_idx, name, arity, captured]}, _env),
    do: {:ok, fn_idx, name, arity, captured}

  defp resolve_fun_ref(_ast, _env), do: :error

  # Reads the captured variable values of a compile-time function reference
  # from the current env.
  defp resolve_captured(captured, env) do
    Enum.map(captured, fn var ->
      case Map.fetch(env, var) do
        {:ok, value} -> value
        :error -> raise Error, "unbound variable reference: #{inspect(var)}"
      end
    end)
  end

  # Materializes a compile-time function reference into a first-class closure
  # word; all other values pass through unchanged.
  defp lift_value({:fn_ref, fn_idx, _name, _arity, captured}, ctx, block, env) do
    env_values = resolve_captured(captured, env)

    unless length(env_values) <= 4 do
      raise Error, "anonymous function capture exceeds 4 slots: #{length(env_values)}"
    end

    create_op(
      "ex.make_fun",
      env_values ++
        [
          fn_idx: MLIR.Attribute.integer(MLIR.Type.i64(), fn_idx),
          env_len: MLIR.Attribute.integer(MLIR.Type.i64(), length(captured)),
          operandSegmentSizes:
            segment_sizes(
              List.duplicate(1, length(captured)) ++ List.duplicate(0, 4 - length(captured))
            )
        ],
      [ex_type("dyn", ctx)],
      ctx,
      block
    )
  end

  defp lift_value(value, _ctx, _block, _env), do: value

  defp zero_i64(ctx, block) do
    create_op(
      "ex.lit",
      [value: MLIR.Attribute.integer(MLIR.Type.i64(), 0)],
      [MLIR.Type.i64()],
      ctx,
      block
    )
  end

  defp lift_tuple_literal(tuple, ctx, block, env) do
    {values, env} = lift_operands_boxed(Tuple.to_list(tuple), ctx, block, env)
    {create_term_op("ex.tuple", values, ctx, block), env}
  end

  defp cmp_predicate(:==), do: "eq"
  defp cmp_predicate(:!=), do: "ne"
  defp cmp_predicate(:<), do: "slt"
  defp cmp_predicate(:<=), do: "sle"
  defp cmp_predicate(:>), do: "sgt"
  defp cmp_predicate(:>=), do: "sge"

  defp lift_case(clauses, scrutinee, env, ctx, block, opts \\ []) do
    if Enum.any?(clauses, &(clause_pattern(&1) |> term_pattern?())) do
      lift_term_case(clauses, scrutinee, env, ctx, block, opts)
    else
      lift_scalar_case(clauses, scrutinee, env, ctx, block, opts)
    end
  end

  defp lift_scalar_case(clauses, scrutinee, env, ctx, block, opts) do
    parsed = Enum.map(clauses, &parse_clause/1)

    unless parsed |> List.last() |> Map.fetch!(:patterns) == [] do
      raise Error, "case requires a final catch-all clause"
    end

    guards =
      Enum.map(parsed, fn clause ->
        case clause.guard do
          nil -> nil
          guard_ast -> lift_guard(guard_ast, clause.vars, scrutinee, env, ctx, block)
        end
      end)

    region = MLIR.CAPI.mlirRegionCreate()

    yield_types =
      parsed
      |> Enum.zip(guards)
      |> Enum.map(fn {clause, guard} ->
        add_clause_block(clause, guard, scrutinee, env, ctx, region)
      end)

    [first_type | rest_types] = yield_types

    unless Keyword.get(opts, :relax_types, false) or
             Enum.all?(rest_types, &MLIR.equal?(first_type, &1)) do
      raise Error, "case clauses must yield the same type"
    end

    result_type = first_type

    case_op =
      %Beaver.SSA{
        op: "ex.case",
        ip: block,
        ctx: ctx,
        arguments: [scrutinee, operandSegmentSizes: segment_sizes([1])],
        results: [result_type],
        loc: MLIR.Location.unknown(),
        filler: fn -> [region] end
      }
      |> MLIR.Operation.create()

    case_op |> MLIR.Operation.results() |> Enum.to_list() |> hd()
  end

  defp lift_term_case(clauses, scrutinee, env, ctx, block, opts) do
    parsed = Enum.map(clauses, &parse_term_clause/1)

    unless match?(
             {name, _, nil} when is_atom(name),
             parsed |> List.last() |> Map.fetch!(:pattern)
           ) do
      raise Error, "case requires a final catch-all clause"
    end

    # term reads require a tagged word, so box the scrutinee once up front
    # (a no-op for values that already are terms). Multi-clause function
    # arguments already carry the tagged word, so they are re-typed with
    # ex.to_word instead (pure passthrough, no re-tagging).
    scrutinee =
      if Keyword.get(opts, :box_scrutinee, true) do
        box_term(scrutinee, ctx, block)
      else
        create_op("ex.to_word", [scrutinee], [ex_type("dyn", ctx)], ctx, block)
      end

    {guards, bindss} =
      parsed
      |> Enum.map(fn clause ->
        {match_cond, binds} =
          build_match(clause.pattern, scrutinee, ctx, block, clause.guard == nil)

        cond =
          case clause.guard do
            nil ->
              match_cond

            guard_ast ->
              guard_cond = lift_term_guard(guard_ast, binds, env, ctx, block)
              combine([match_cond, guard_cond], ctx, block)
          end

        {cond, binds}
      end)
      |> Enum.unzip()

    # `receive` clauses guard integer messages with `is_integer(x)`; the
    # bound word is untagged so the clause body can use it in scalar
    # arithmetic.
    bindss =
      if Keyword.get(opts, :untag_int_binds, false) do
        untag_int_binds(parsed, bindss, ctx, block)
      else
        bindss
      end

    region = MLIR.CAPI.mlirRegionCreate()

    yield_types =
      parsed
      |> Enum.zip(guards)
      |> Enum.zip(bindss)
      |> Enum.map(fn {{clause, guard}, binds} ->
        add_term_clause_block(clause, guard, binds, env, ctx, region)
      end)

    [first_type | rest_types] = yield_types

    unless Keyword.get(opts, :relax_types, false) or
             Enum.all?(rest_types, &MLIR.equal?(first_type, &1)) do
      raise Error, "case clauses must yield the same type"
    end

    case_op =
      %Beaver.SSA{
        op: "ex.case",
        ip: block,
        ctx: ctx,
        arguments: [scrutinee, operandSegmentSizes: segment_sizes([1])],
        results: [first_type],
        loc: MLIR.Location.unknown(),
        filler: fn -> [region] end
      }
      |> MLIR.Operation.create()

    case_op |> MLIR.Operation.results() |> Enum.to_list() |> hd()
  end

  defp untag_int_binds(parsed, bindss, ctx, block) do
    parsed
    |> Enum.zip(bindss)
    |> Enum.map(fn {%{guard: guard}, binds} ->
      case integer_guard_var(guard) do
        nil ->
          binds

        var ->
          Enum.map(binds, fn
            {^var, value} ->
              {var, create_op("ex.to_int", [value], [MLIR.Type.i64()], ctx, block)}

            other ->
              other
          end)
      end
    end)
  end

  defp integer_guard_var({:is_integer, _, [{var, _, nil}]}) when is_atom(var), do: var
  defp integer_guard_var(_guard), do: nil

  # The match condition and the bound values of one term pattern are computed
  # eagerly before `ex.case`: predicates and reads are pure and safe on the
  # wrong term kind (reads return nil), so a non-matching clause's eager
  # values are simply unused. The combined condition becomes the clause guard.
  # `defer_rest?` moves the rest-slice materialization of a top-level binary
  # pattern into the clause body (expandable 210418e): without a guard, the
  # slice is only needed when the clause matches, so a rejected clause never
  # allocates it.
  defp build_match(pattern, value, ctx, block, defer_rest?) do
    case pattern do
      {:<<>>, _, segments} -> build_binary_match(segments, value, ctx, block, defer_rest?)
      _ -> do_build_match(pattern, value, ctx, block)
    end
  end

  defp do_build_match({name, _, nil}, value, _ctx, _block) when is_atom(name) do
    if name == :_ do
      {nil, []}
    else
      {nil, [{name, value}]}
    end
  end

  defp do_build_match(integer, value, ctx, block) when is_integer(integer) do
    lit =
      create_op(
        "ex.lit",
        [value: MLIR.Attribute.integer(MLIR.Type.i64(), integer)],
        [MLIR.Type.i64()],
        ctx,
        block
      )

    boxed = box_term(lit, ctx, block)
    {create_op("ex.term_eq", [value, boxed], [MLIR.Type.i64()], ctx, block), []}
  end

  defp do_build_match(tuple, value, ctx, block) when is_tuple(tuple) and tuple_size(tuple) != 3 do
    build_tuple_match(Tuple.to_list(tuple), value, ctx, block)
  end

  defp do_build_match({a, b, c}, value, ctx, block)
       when not (is_atom(a) and is_list(b) and is_list(c)) do
    build_tuple_match([a, b, c], value, ctx, block)
  end

  defp do_build_match({:{}, _, elements}, value, ctx, block) do
    build_tuple_match(elements, value, ctx, block)
  end

  defp do_build_match({:<<>>, _, segments}, value, ctx, block) do
    build_binary_match(segments, value, ctx, block)
  end

  defp do_build_match([], value, ctx, block) do
    cond_list =
      create_op("ex.is_list", [box_term(value, ctx, block)], [MLIR.Type.i64()], ctx, block)

    cond_len =
      cmp(
        create_op("ex.list_length", [value], [MLIR.Type.i64()], ctx, block),
        0,
        "eq",
        ctx,
        block
      )

    {combine([cond_list, cond_len], ctx, block), []}
  end

  defp do_build_match([{:|, _, [head, tail]}], value, ctx, block) do
    cond_list =
      create_op("ex.is_list", [box_term(value, ctx, block)], [MLIR.Type.i64()], ctx, block)

    cond_nonempty =
      cmp(
        create_op("ex.list_length", [value], [MLIR.Type.i64()], ctx, block),
        0,
        "ne",
        ctx,
        block
      )

    head_value = create_op("ex.list_head", [value], [ex_type("dyn", ctx)], ctx, block)
    tail_value = create_op("ex.list_tail", [value], [ex_type("dyn", ctx)], ctx, block)
    {head_cond, head_binds} = do_build_match(head, head_value, ctx, block)
    {tail_cond, tail_binds} = do_build_match(tail, tail_value, ctx, block)

    {combine([cond_list, cond_nonempty, head_cond, tail_cond], ctx, block),
     head_binds ++ tail_binds}
  end

  defp do_build_match(elements, value, ctx, block) when is_list(elements) do
    cond_list =
      create_op("ex.is_list", [box_term(value, ctx, block)], [MLIR.Type.i64()], ctx, block)

    cond_len =
      cmp(
        create_op("ex.list_length", [value], [MLIR.Type.i64()], ctx, block),
        length(elements),
        "eq",
        ctx,
        block
      )

    {elem_conds, binds} = list_elements_match(elements, value, ctx, block, [])
    {combine([cond_list, cond_len | elem_conds], ctx, block), binds}
  end

  defp do_build_match(other, _value, _ctx, _block) do
    raise Error, "unsupported term pattern: #{inspect(other)}"
  end

  defp build_tuple_match(elements, value, ctx, block) do
    cond_tuple =
      create_op("ex.is_tuple", [box_term(value, ctx, block)], [MLIR.Type.i64()], ctx, block)

    cond_len =
      cmp(
        create_op("ex.tuple_length", [value], [MLIR.Type.i64()], ctx, block),
        length(elements),
        "eq",
        ctx,
        block
      )

    {elem_conds, binds} =
      elements
      |> Enum.with_index()
      |> Enum.map_reduce([], fn {element, index}, binds ->
        element_value =
          create_op(
            "ex.tuple_get",
            [value, lit(index, ctx, block)],
            [ex_type("dyn", ctx)],
            ctx,
            block
          )

        {cond, element_binds} = do_build_match(element, element_value, ctx, block)
        {cond, element_binds ++ binds}
      end)

    {combine([cond_tuple, cond_len | elem_conds], ctx, block), Enum.reverse(binds)}
  end

  defp list_elements_match([], _value, _ctx, _block, binds), do: {[], binds}

  defp list_elements_match([element | rest], value, ctx, block, binds) do
    head_value = create_op("ex.list_head", [value], [ex_type("dyn", ctx)], ctx, block)
    tail_value = create_op("ex.list_tail", [value], [ex_type("dyn", ctx)], ctx, block)
    {head_cond, head_binds} = do_build_match(element, head_value, ctx, block)
    {tail_conds, tail_binds} = list_elements_match(rest, tail_value, ctx, block, binds)
    {[head_cond | tail_conds], head_binds ++ tail_binds}
  end

  defp build_binary_match(segments, value, ctx, block, defer_rest? \\ false) do
    {segs, rest} = parse_binary_segments(segments)

    cond_bin =
      create_op("ex.is_binary", [box_term(value, ctx, block)], [MLIR.Type.i64()], ctx, block)

    {conds, binds, offset} =
      Enum.reduce(segs, {[], [], lit(0, ctx, block)}, fn seg, {conds, binds, offset} ->
        case seg do
          {:byte, pat} ->
            byte_value =
              create_op(
                "ex.binary_get",
                [value, offset],
                [ex_type("dyn", ctx)],
                ctx,
                block
              )

            {cond, pat_binds} = do_build_match(pat, byte_value, ctx, block)

            next =
              create_op("ex.add", [offset, lit(1, ctx, block)], [MLIR.Type.i64()], ctx, block)

            {[cond | conds], pat_binds ++ binds, next}

          {:utf8, pat} ->
            width =
              create_op("ex.binary_utf8_width", [value, offset], [MLIR.Type.i64()], ctx, block)

            codepoint =
              create_op("ex.binary_utf8_get", [value, offset], [ex_type("dyn", ctx)], ctx, block)

            cond_w = cmp(width, 0, "ne", ctx, block)
            {pat_cond, pat_binds} = do_build_match(pat, codepoint, ctx, block)
            next = create_op("ex.add", [offset, width], [MLIR.Type.i64()], ctx, block)
            {[cond_w, pat_cond | conds], pat_binds ++ binds, next}
        end
      end)

    {rest_cond, rest_binds} = build_rest_bind(rest, value, offset, ctx, block, defer_rest?)

    cond_len =
      cmp(
        create_op("ex.binary_length", [value], [MLIR.Type.i64()], ctx, block),
        offset,
        if(rest == nil, do: "eq", else: "sge"),
        ctx,
        block
      )

    {combine([cond_bin, cond_len | Enum.reverse(conds) ++ [rest_cond]], ctx, block),
     Enum.reverse(binds) ++ rest_binds}
  end

  defp build_rest_bind(nil, _value, _offset, _ctx, _block, _defer_rest?), do: {nil, []}

  defp build_rest_bind({name, _, nil}, value, offset, ctx, _block, true)
       when is_atom(name) and name != :_ do
    slice = fn clause_block ->
      create_op("ex.binary_slice", [value, offset], [ex_type("dyn", ctx)], ctx, clause_block)
    end

    {nil, [{name, {:deferred, slice}}]}
  end

  defp build_rest_bind(rest_pat, value, offset, ctx, block, _defer_rest?) do
    rest_value =
      create_op("ex.binary_slice", [value, offset], [ex_type("dyn", ctx)], ctx, block)

    do_build_match(rest_pat, rest_value, ctx, block)
  end

  defp parse_binary_segments(segments) do
    {segs, rest} =
      Enum.split_while(segments, &(not match?({:"::", _, [_, {:binary, _, nil}]}, &1)))

    case rest do
      [] ->
        {Enum.map(segs, &binary_segment!/1), nil}

      [{:"::", _, [rest_pat, {:binary, _, nil}]}] ->
        {Enum.map(segs, &binary_segment!/1), rest_pat}

      _ ->
        raise Error, "binary rest segment must be the last segment: #{inspect(segments)}"
    end
  end

  defp binary_segment!({:"::", _, [pat, 8]}), do: {:byte, pat}
  defp binary_segment!({:"::", _, [pat, {:utf8, _, nil}]}), do: {:utf8, pat}
  defp binary_segment!(pat) when is_integer(pat), do: {:byte, pat}
  defp binary_segment!({name, _, nil} = pat) when is_atom(name), do: {:byte, pat}

  defp binary_segment!(segment) do
    raise Error, "unsupported binary segment: #{inspect(segment)}"
  end

  defp add_term_clause_block(clause, guard, binds, env, ctx, region) do
    block = MLIR.Block.create([], [])
    MLIR.CAPI.mlirRegionAppendOwnedBlock(region, block)

    clause_args = if guard, do: [guard], else: []
    create_op("ex.clause", clause_args ++ [patterns: pattern_attr([])], [], ctx, block)

    clause_env =
      Enum.reduce(binds, env, fn
        {var, {:deferred, fun}}, acc -> Map.put(acc, var, fun.(block))
        {var, value}, acc -> Map.put(acc, var, value)
      end)

    {value, clause_env} = lift_block(List.wrap(clause.body), ctx, block, clause_env)
    value = lift_value(value, ctx, block, clause_env)
    create_op("ex.yield", [value, operandSegmentSizes: segment_sizes([1])], [], ctx, block)
    MLIR.Value.type(value)
  end

  defp combine(conds, ctx, block) do
    conds
    |> Enum.reject(&is_nil/1)
    |> case do
      [] ->
        nil

      [single] ->
        single

      many ->
        Enum.reduce(many, fn cond, acc ->
          create_op("arith.andi", [acc, cond], [MLIR.Type.i64()], ctx, block)
        end)
    end
  end

  # Term-pattern guards are evaluated eagerly against the (nil-safe) bound
  # values, so they must be composed of term-safe predicates only: `is_*`
  # calls on bound or outer variables. Comparisons and arithmetic on terms
  # are rejected explicitly.
  defp lift_term_guard(guard_ast, binds, env, ctx, block) do
    unless supported_term_guard?(guard_ast) do
      raise Error,
            "unsupported guard on term pattern (only is_* predicates on bound or outer variables): " <>
              inspect(guard_ast)
    end

    guard_env = Map.merge(env, Map.new(binds))
    {value, _env} = lift_expr(guard_ast, ctx, block, guard_env)
    value
  end

  defp supported_term_guard?({predicate, _, [var_ast]})
       when predicate in [:is_integer, :is_atom, :is_binary, :is_list, :is_tuple, :is_map] do
    match?({name, _, nil} when is_atom(name), var_ast)
  end

  defp supported_term_guard?({op, _, [left, right]}) when op in [:==, :!=] do
    guard_operand?(left) and guard_operand?(right)
  end

  defp supported_term_guard?(_guard_ast), do: false

  defp guard_operand?(value) when is_integer(value), do: true
  defp guard_operand?(value) when is_binary(value), do: true
  defp guard_operand?({name, _, nil}) when is_atom(name), do: true
  defp guard_operand?({:<<>>, _, _}), do: true
  defp guard_operand?({:%{}, _, _}), do: true
  defp guard_operand?(tuple) when is_tuple(tuple) and tuple_size(tuple) != 3, do: true

  defp guard_operand?(_), do: false

  defp term_operand?(value) do
    value
    |> MLIR.Value.type()
    |> MLIR.to_string()
    |> then(&(&1 in ["!ex.dyn", "!ex.bound", "!ex.unbound"]))
  end

  defp box_if_scalar(value, ctx, block) do
    if term_operand?(value), do: value, else: box_term(value, ctx, block)
  end

  defp lit(value, ctx, block) do
    create_op(
      "ex.lit",
      [value: MLIR.Attribute.integer(MLIR.Type.i64(), value)],
      [MLIR.Type.i64()],
      ctx,
      block
    )
  end

  defp cmp(left, right, predicate, ctx, block) do
    right = if is_integer(right), do: lit(right, ctx, block), else: right

    create_op(
      "ex.cmp",
      [left, right, predicate: MLIR.Attribute.string(predicate)],
      [MLIR.Type.i64()],
      ctx,
      block
    )
  end

  defp clause_pattern({:->, _, [args, _body]}) when is_list(args) do
    case args do
      [{:when, _, [pattern, _guard]}] -> pattern
      [pattern] -> pattern
      _ -> raise Error, "case clauses with multiple patterns are unsupported: #{inspect(args)}"
    end
  end

  defp parse_term_clause({:->, _, [args, body]}) when is_list(args) do
    {pattern, guard} =
      case args do
        [{:when, _, [pattern, guard]}] -> {pattern, guard}
        [pattern] -> {pattern, nil}
        _ -> raise Error, "case clauses with multiple patterns are unsupported: #{inspect(args)}"
      end

    %{pattern: pattern, guard: guard, body: body}
  end

  defp term_pattern?(pattern) do
    pattern
    |> PatternPlan.lower_pattern()
    |> Enum.any?(&(&1.op in [:tuple, :list_exact, :list_cons, :binary]))
  end

  defp parse_clause({:->, _, [args, body]}) when is_list(args) do
    {pattern, guard} =
      case args do
        [{:when, _, [pattern, guard]}] -> {pattern, guard}
        [pattern] -> {pattern, nil}
        _ -> raise Error, "case clauses with multiple patterns are unsupported: #{inspect(args)}"
      end

    {patterns, vars} = parse_pattern(pattern)
    %{pattern: pattern, patterns: patterns, vars: vars, guard: guard, body: body}
  end

  defp parse_pattern(integer) when is_integer(integer), do: {[integer], []}

  defp parse_pattern({name, _, nil}) when is_atom(name) do
    if name == :_ do
      {[], []}
    else
      {[], [name]}
    end
  end

  defp parse_pattern(pattern) do
    raise Error, "unsupported case pattern: #{inspect(pattern)}"
  end

  defp lift_guard(guard_ast, vars, scrutinee, env, ctx, block) do
    guard_env = Enum.reduce(vars, env, fn var, acc -> Map.put(acc, var, scrutinee) end)
    {value, _env} = lift_expr(guard_ast, ctx, block, guard_env)
    value
  end

  defp add_clause_block(clause, guard, scrutinee, env, ctx, region) do
    block = MLIR.Block.create([], [])
    MLIR.CAPI.mlirRegionAppendOwnedBlock(region, block)

    clause_env = Enum.reduce(clause.vars, env, fn var, acc -> Map.put(acc, var, scrutinee) end)

    clause_attrs = [patterns: pattern_attr(clause.patterns)]
    clause_args = if guard, do: [guard], else: []
    create_op("ex.clause", clause_args ++ clause_attrs, [], ctx, block)

    {value, clause_env} = lift_block(List.wrap(clause.body), ctx, block, clause_env)
    value = lift_value(value, ctx, block, clause_env)
    create_op("ex.yield", [value, operandSegmentSizes: segment_sizes([1])], [], ctx, block)
    MLIR.Value.type(value)
  end

  defp pattern_attr(patterns) do
    MLIR.Attribute.dense_array(patterns, Beaver.Native.I64)
  end

  # Values crossing into a term-universe op are boxed with `ex.box`; the
  # conversion turns the box into a tagged word (and is a no-op for values
  # that already are terms).
  defp lift_operands_boxed(args, ctx, block, env) do
    Enum.map_reduce(args, env, fn arg, env ->
      {value, env} = lift_expr(arg, ctx, block, env)
      {box_term(lift_value(value, ctx, block, env), ctx, block), env}
    end)
  end

  defp box_term(value, ctx, block) do
    create_op("ex.box", [value], [ex_type("dyn", ctx)], ctx, block)
  end

  defp lift_map_entries(entries, ctx, block, env) do
    Enum.flat_map_reduce(entries, env, fn entry, env ->
      case entry do
        {key, _value} when is_atom(key) ->
          raise Error,
                "atom-keyed map entries are unsupported in the term slice: #{inspect(entry)}"

        {key, value} ->
          {key_value, env} = lift_expr(key, ctx, block, env)
          {value_value, env} = lift_expr(value, ctx, block, env)

          {
            [
              box_term(lift_value(key_value, ctx, block, env), ctx, block),
              box_term(lift_value(value_value, ctx, block, env), ctx, block)
            ],
            env
          }

        other ->
          {value, env} = lift_expr(other, ctx, block, env)
          {[box_term(lift_value(value, ctx, block, env), ctx, block)], env}
      end
    end)
  end

  defp create_term_op(op_name, args, ctx, block) do
    create_op(
      op_name,
      args ++ [operandSegmentSizes: segment_sizes([length(args)])],
      [ex_type("dyn", ctx)],
      ctx,
      block
    )
  end

  defp insert_return(nil, ctx, block, _env) do
    create_op("ex.return", [operandSegmentSizes: segment_sizes([0])], [], ctx, block)
    :ok
  end

  defp insert_return(value, ctx, block, env) do
    value = lift_value(value, ctx, block, env)
    create_op("ex.return", [value, operandSegmentSizes: segment_sizes([1])], [], ctx, block)
    :ok
  end

  defp create_op(op_name, arguments, result_types, ctx, block) do
    operation =
      %Beaver.SSA{
        op: op_name,
        ip: block,
        ctx: ctx,
        arguments: arguments,
        results: result_types,
        loc: MLIR.Location.unknown()
      }
      |> MLIR.Operation.create()

    case result_types do
      [] -> operation
      [_] -> operation |> MLIR.Operation.results() |> Enum.to_list() |> hd()
      _ -> operation |> MLIR.Operation.results() |> Enum.to_list()
    end
  end

  defp param_name({name, _, nil}) when is_atom(name), do: name
  defp param_name(pattern), do: raise(Error, "unsupported parameter pattern: #{inspect(pattern)}")

  defp integer_type(ctx), do: MLIR.Type.integer(64, ctx: ctx)

  defp ex_type(name, ctx) do
    Beaver.Slang.create_constrained_element(:type, "ex", name, [], ctx: ctx)
    |> Beaver.Deferred.create(ctx)
  end

  defp segment_sizes(sizes) do
    MLIR.Attribute.dense_array(sizes, Beaver.Native.I32)
  end

  # ex.call has eight optional argument slots (the closure ABI adds four);
  # encode which are filled.
  defp arg_segment_sizes(count) do
    unless count <= 8 do
      raise Error, "calls with more than 8 arguments are unsupported: #{count}"
    end

    List.duplicate(1, count) ++ List.duplicate(0, 8 - count)
  end
end
