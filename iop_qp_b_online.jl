## iop_online
#  Chaosheng Dong
#  chaosheng@pitt.edu
#  All rights reserved 2017-03-17
# function
@everywhere using JuMP, Gurobi, Distributions, Clustering, Distances
@everywhere function f(Q,c,x)
    1/2*vecdot(x, Q*x) + vecdot(c,x)
end
@everywhere function qp_b_online(m,n,M,lambda,Q,c,A,b,y)
    problem = Model( solver=GurobiSolver( OutputFlag = 0, MIPGap = 1e-6, TimeLimit = 300) )
    @variable(problem, -100 <= b_t[1:n+1] <= 0)
    @variable(problem, x[1:n] )
    @variable(problem, u[1:m] >= 0 )
    @variable(problem, t[1:m], Bin )
    @variable(problem, eta[1:n] >= 0 )
    @variable(problem, eta1)
    @variable(problem, eta2)
    @objective(problem, Min, eta1/2 + lambda*eta2 )
    @constraint(problem, b_t[2:end] .== zeros(n,1))
    @constraint(problem, norm([2*(b - b_t);eta1-1]) <= eta1 + 1 )
    @constraint(problem, norm([2*(y - x);eta2-1]) <= eta2 + 1 )
    @constraint(problem, A*x .>= b_t )
    @constraint(problem, u .<= M*t )
    @constraint(problem, A*x - b_t .<= M*( 1 - t ) )
    @constraint(problem, Q*x + c - A'*u .== zeros(n) )
    solve(problem)
    return getvalue(b_t)
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
for rep = 1:100
    # srand(1)
    P = rand(Uniform(pmin,pmax),T,n);
    x = GenerateData(n,Q,c,-P,b_true,T);
    # srand(1)
    err = rand(Uniform(-0.25,0.25),n,T);
    # err = rand(Normal(0,0.1),n,T);
    y = x + err; # add noise
    # writedlm("C:/Users/CHD58/Desktop/y.txt", y, ' ') # observations
    #########################################################################
    M = 1e4; lambda = 100;
    b = zeros(n+1,1);
    tic()
    for t = 1:T
        println("rep = ", rep, ",", "t  = ", t, "\n")
        estimation_error = push!(estimation_error, norm(b - b_true,2) )
        println("estimation_error = ", norm(b - b_true,2), "\n")
        xx = GenerateData2(n,Q,c,[-P[t,:]';eye(n)],b);
        cum_risk = push!(cum_risk, sqeuclidean(y[:,t], xx))
        b = qp_b_online(m,n,M,lambda*t^(-1/2),Q,c,[-P[t,:]';eye(n)],b,y[:,t]);
    end
    t = toc()
    Time = push!(Time, t)
end
# writedlm("C:/Users/CHD58/Desktop/estimation_error.txt", estimation_error, ' ')
# writedlm("C:/Users/CHD58/Desktop/cum_risk.txt", cum_risk, ' ')
# writedlm("C:/Users/CHD58/Desktop/Time_b_.txt", Time, ' ')

#######################################################################
