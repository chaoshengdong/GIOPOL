## iop_online
#  Chaosheng Dong
#  chaosheng@pitt.edu
#  All rights reserved 2017-03-17
# function
@everywhere using JuMP, Gurobi, Distributions, Clustering, Distances
@everywhere function f(Q,c,x)
    1/2*vecdot(x, Q*x) + vecdot(c,x)
end
@everywhere function qp_b_batch(m,n,M,Q,c,P,y,N)
    problem = Model( solver=GurobiSolver( OutputFlag = 1, MIPGap = 1e-6) )
    @variable(problem, -100 <= b[1:n+1] <= 0 )
    @variable(problem, x[1:n,1:N] )
    @variable(problem, u[1:m,1:N] >= 0 )
    @variable(problem, t[1:m,1:N], Bin )
    @variable(problem, eta[1:N] )
    @objective(problem, Min, sum(eta) )
    # @constraint(problem, c_t[1:2] .== [1;2])
    @constraint(problem, b[2:end] .== zeros(n,1) )
    for i = 1:N
        @constraint(problem, norm([2*(y[:,i] - x[:,i]);eta[i]-1]) <= eta[i] + 1 )
        @constraint(problem, [-P[i,:]';eye(n)]*x[:,i] .>= b )
        @constraint(problem, u[:,i] .<= M*t[:,i] )
        @constraint(problem, [-P[i,:]';eye(n)]*x[:,i] - b .<= M*( 1 - t[:,i] ) )
        @constraint(problem, Q*x[:,i] + c - [-P[i,:]';eye(n)]'*u[:,i] .== zeros(n) )
    end
    solve(problem)
    return getvalue(b)
end
@everywhere function GenerateData(n,Q,c,A,b,T)
    x = SharedArray{Float64,2}((n,T));
    @parallel (+) for i = 1:T
        genData = Model( solver=GurobiSolver( OutputFlag=0 ) )
        @variable(genData, y[1:n] )
        @objective(genData, Min, f(Q,c,y) )
        @constraint(genData, [A[i,:]';eye(n)]*y .>= b)
        solve(genData)
        x[:,i] = getvalue(y)
    end
    return x
end
@everywhere function GenerateData2(n,Q,c,A,b)
    genData = Model( solver=GurobiSolver( OutputFlag=0 ) )
    @variable(genData, y[1:n] )
    @objective(genData, Min, f(Q,c,y) )
    @constraint(genData, A*y .>= b)
    solve(genData)
    return getvalue(y)
end
n = 10; m = n+1;
srand(1);
Q = diagm(rand(Uniform(0,10),n));
# Q = rand(n,n); Q = Q*Q'; Q = Q + 1e0*eye(n);
srand(1);
c = -rand(Uniform(0,5),n); b = [-40;zeros(n,1)]; b_true = b;
T = 50; pmin = 5; pmax = 25; estimation_error = Float64[]; cum_risk = Float64[]; Time = Float64[];
for rep = 1:10
    # srand(1)
    P = rand(Uniform(pmin,pmax),T,n);
    x = GenerateData(n,Q,c,-P,b_true,T);
    # srand(1)
    err = rand(Uniform(-0.25,0.25),n,T);
    y = x + err; # add noise
    # writedlm("C:/Users/CHD58/Desktop/y.txt", y, ' ') # observations
    #########################################################################
    M = 1e4;
    println("rep = ", rep, "\n")
    tic()
    b = qp_b_batch(m,n,M,Q,c,P,y,T);
    ttime = toc()
    Time = push!(Time, ttime)
    estimation_error = push!(estimation_error, norm(b - b_true,2) )
    println("estimation_error = ", norm(b - b_true,2), "\n")
end
# writedlm("C:/Users/CHD58/Desktop/estimation_error.txt", estimation_error, ' ')
# writedlm("C:/Users/CHD58/Desktop/Time_b_batch50.txt", Time, ' ')
#######################################################################
