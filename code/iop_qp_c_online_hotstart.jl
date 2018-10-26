## iop_online
#  Chaosheng Dong
#  chaosheng@pitt.edu
#  All rights reserved 2017-03-17
# function
@everywhere using JuMP, Gurobi, Distributions, Clustering, Distances
@everywhere function f(Q,c,x)
    1/2*vecdot(x, Q*x) + vecdot(c,x)
end
@everywhere function qp_c_online(m,n,M,lambda,Q,c,A,b,y)
    problem = Model( solver=GurobiSolver( OutputFlag = 0, MIPGap = 1e-6, TimeLimit = 300) )
    @variable(problem, -5 <= c_t[1:n] <= 0)
    @variable(problem, x[1:n] )
    @variable(problem, u[1:m] >= 0 )
    @variable(problem, t[1:m], Bin )
    @variable(problem, eta1)
    @variable(problem, eta2)
    @objective(problem, Min, eta1/2 + lambda*eta2 )
    # @constraint(problem, c_t[1:2] .== [1;2])
    @constraint(problem, norm([2*(c - c_t);eta1-1]) <= eta1 + 1 )
    @constraint(problem, norm([2*(y - x);eta2-1]) <= eta2 + 1 )
    @constraint(problem, A*x .>= b )
    @constraint(problem, u .<= M*t )
    @constraint(problem, A*x - b .<= M*( 1 - t ) )
    @constraint(problem, Q*x + c_t - A'*u .== zeros(n) )
    solve(problem)
    return getvalue(c_t)
end
@everywhere function qp_c_batch(m,n,M,Q,P,b,y,N)
    problem = Model( solver=GurobiSolver( OutputFlag = 0, MIPGap = 1e-3, ConcurrentMIP = 4) )
    @variable(problem, -5 <= c[1:n] <= 0)
    # @variable(problem, x[1:n,1:N] )
    @variable(problem, u[1:m,1:N] >= 0 )
    # @variable(problem, t[1:m,1:N], Bin )
    @variable(problem, eta2[1:N] )
    @variable(problem, eta3[1:N] )
    @objective(problem, Min, sum(eta2) + sum(eta3) )
    # @constraint(problem, c_t[1:2] .== [1;2])
    for i = 1:N
        # @constraint(problem, norm([2*(y[:,i] - x[:,i]);eta[i]-1]) <= eta[i] + 1 )
        # @constraint(problem, [-P[i,:]';eye(n)]*x[:,i] .>= b )
        # @constraint(problem, u[:,i] .<= M*t[:,i] )
        # @constraint(problem, [-P[i,:]';eye(n)]*x[:,i] - b .<= M*( 1 - t[:,i] ) )
        # @constraint(problem, Q*x[:,i] + c - [-P[i,:]';eye(n)]'*u[:,i] .== zeros(n) )
        @constraint(problem, norm([2*(u[:,i]'*([-P[i,:]';eye(n)]*y[:,i] - b));eta3[i]-1]) <= eta3[i] + 1 )
        @constraint(problem, norm([2*(Q*y[:,i] + c - [-P[i,:]';eye(n)]'*u[:,i]);eta2[i]-1]) <= eta2[i] + 1 )
    end
    solve(problem)
    return getvalue(c)
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
c = -rand(Uniform(0,5),n); b = [-40;zeros(n,1)];
c_true = c;
T = 1000; pmin = 5; pmax = 25; estimation_error = Float64[]; cum_risk = Float64[]; Time = Float64[];
for rep = 1:100
    srand(rep)
    P = rand(Uniform(pmin,pmax),T,n);
    x = GenerateData(n,Q,c_true,-P,b,T);
    srand(rep)
    err = rand(Uniform(-0.25,0.25),n,T);
    y = x + err; # add noise
    # writedlm("C:/Users/CHD58/Desktop/y.txt", y, ' ') # observations
    #########################################################################
    M = 1e4; lambda = 5;
    ################ Initialization of theta ################################
    c = zeros(n,1);
    tic()
    c = qp_c_batch(m,n,M,Q,P[1:T,:],b,y[:,1:T],T);
    toc()
    ################ Initialization of theta ################################
    tic()
    for t = 1:T
        println("rep = ", rep, ",", "t  = ", t, "\n")
        estimation_error = push!(estimation_error, norm(c - c_true,2) )
        println("estimation_error = ", norm(c - c_true,2), "\n")
        xx = GenerateData2(n,Q,c,[-P[t,:]';eye(n)],b);
        cum_risk = push!(cum_risk, sqeuclidean(y[:,t], xx))
        c = qp_c_online(m,n,M,lambda*t^(-1/2),Q,c,[-P[t,:]';eye(n)],b,y[:,t]);
    end
    tt = toc()
    Time = push!(Time, tt)
end
# writedlm("C:/Users/CHD58/Desktop/estimation_error_hotstart.txt", estimation_error, ' ')
# writedlm("C:/Users/CHD58/Desktop/cum_risk.txt", cum_risk, ' ')
# writedlm("C:/Users/CHD58/Desktop/Time15.txt", Time, ' ')
#######################################################################
