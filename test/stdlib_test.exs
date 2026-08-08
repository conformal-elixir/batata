defmodule Batata.StdlibTest do
  use Batata.Case, async: true

  alias Batata
  alias Batata.Stdlib

  describe "domain registry" do
    test "classifies declared Kernel entries" do
      assert Stdlib.class({Kernel, :length, 1}) == :native_term
      assert Stdlib.class({Kernel, :hd, 1}) == :native_term
      assert Stdlib.class({Kernel, :elem, 2}) == :native_term
      assert Stdlib.class({Kernel, :map_size, 1}) == :native_term
      assert Stdlib.class({:erlang, :length, 1}) == :native_term
    end

    test "classifies declared domain modules" do
      assert Stdlib.class({List, :first, 1}) == :native_term
      assert Stdlib.class({Map, :size, 1}) == :native_term
      assert Stdlib.class({Tuple, :size, 1}) == :native_term
      assert Stdlib.class({Tuple, :delete_at, 2}) == :unsupported
      assert Stdlib.class({Enum, :map, 2}) == :beamer_callback
    end

    test "returns nil outside the declared surface" do
      assert Stdlib.class({Foo, :bar, 1}) == nil
      assert Stdlib.class({Kernel, :apply, 2}) == nil
    end
  end

  describe "execution" do
    test "resolves auto-imported Kernel BIFs", %{ctx: ctx} do
      assert 3 == execute("length([1, 2, 3])", ctx)
      assert 8 == execute("hd([1, 2, 3])", ctx)
      assert 16 == execute("hd(tl([1, 2, 3]))", ctx)
      assert 1 == execute("is_integer(5)", ctx)
    end

    test "resolves module-qualified stdlib calls", %{ctx: ctx} do
      assert 3 == execute("Kernel.length([1, 2, 3])", ctx)
      assert 56 == execute("List.first([7, 8])", ctx)
      assert 1 == execute("Kernel.is_list([1])", ctx)
    end

    test "reads term sizes and elements", %{ctx: ctx} do
      assert 2 == execute("tuple_size({10, 20})", ctx)
      assert 80 == execute("elem({10, 20}, 1)", ctx)
      assert 160 == execute("elem({10, 20}, 2)", ctx)
      assert 3 == execute("byte_size(<<1, 2, 3>>)", ctx)
      assert 1 == execute("map_size(%{1 => 2})", ctx)
      assert 2 == execute("Map.size(%{1 => 2, 3 => 4})", ctx)
      assert 2 == execute("Tuple.size({1, 2})", ctx)
    end

    test "rejects BEAM-callback stdlib calls explicitly", %{ctx: ctx} do
      error =
        assert_raise Batata.Lift.Error, fn ->
          execute("Enum.count([1, 2, 3])", ctx)
        end

      assert error.message =~ "requires BEAM callback interop"
    end

    test "rejects undeclared stdlib calls explicitly", %{ctx: ctx} do
      error =
        assert_raise Batata.Lift.Error, fn ->
          execute("Foo.bar(1)", ctx)
        end

      assert error.message =~ "unsupported stdlib call: Foo.bar/1"
    end

    test "rejects declared-but-unsupported stdlib calls explicitly", %{ctx: ctx} do
      error =
        assert_raise Batata.Lift.Error, fn ->
          execute("Tuple.delete_at({1, 2}, 1)", ctx)
        end

      assert error.message =~ "declared but not yet supported"
    end
  end

  defp execute(expr, ctx) do
    source = """
    defmodule Math do
      def main() do
        #{expr}
      end
    end
    """

    Batata.execute(source, ctx)
  end
end
