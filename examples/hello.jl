# msg = "Hello World"
# println(msg)

using ForwardDiff
using BenchmarkTools

function f(x)
    return [ x[0] , 5*x[1] ]
end

y = ForwardDiff.jacobian(f, [1.0 ,1.0])
println(y)


# f(x::Vector) = sum(sin, x) + prod(tan, x) * sum(sqrt, x);
# x = rand(5)
# g = x -> ForwardDiff.gradient(f, x);
# @btime g(x)

# println(y)