## iop_online
#  Chaosheng Dong
#  chaosheng@pitt.edu
#  All rights reserved 2017-03-17
# function
@everywhere using JuMP, Gurobi, Distributions, Clustering, Distances
@everywhere function f(Q,c,x)
    1/2*vecdot(x, Q*x) + vecdot(c,x)
end
@everywhere function qp_c_online(m,n,M,lambda,Q,c,A,d,y,ymax,xmax,c_true,num_s,num_edge)
    problem = Model( solver=GurobiSolver( OutputFlag = 0, MIPGap = 1e-6, TimeLimit = 300) )
    @variable(problem, 0 <= c_t[1:n] <= 10 )
    @variable(problem, x[1:n] )
    @variable(problem, u[1:m] >= 0 )
    @variable(problem, t[1:m], Bin )
    @variable(problem, eta1 )
    @variable(problem, eta2 )
    @objective(problem, Min, eta1/2 + lambda*eta2 )
    @constraint(problem, c_t[1:4] .== c_true[1:4] )
    @constraint(problem, c_t[7:8] .== c_true[7:8] )
    @constraint(problem, c_t[3:8] .>= 1*ones(6) )
    # @constraint(problem, c_t .== c_true )
    @constraint(problem, norm([2*(c - c_t);eta1-1]) <= eta1 + 1 )
    @constraint(problem, norm([2*(y - x);eta2-1]) <= eta2 + 1 )
    @constraint(problem, A*x .>= [zeros(num_s);zeros(num_edge);x[1:num_s];d;-x[1:num_s];-d;-ymax;-xmax] )
    @constraint(problem, u .<= M*t )
    @constraint(problem, A*x - [zeros(num_s);zeros(num_edge);x[1:num_s];d;-x[1:num_s];-d;-ymax;-xmax] .<= M*( 1 - t ) )
    @constraint(problem, Q*x + c_t - A'*u .== zeros(n) )
    solve(problem)
    return getvalue(c_t)
end
@everywhere function GenerateData(n,Q,c,A,Dem,T,ymax,xmax,num_s,num_edge)
    x = SharedArray{Float64,2}((n,T));
    @parallel (+) for i = 1:T
    # b = [zeros(num_s);zeros(num_edge);y[1:num_s];Dem[i,:];-y[1:num_s];;-Dem[i,:];-ymax;-xmax];
    genData = Model( solver=GurobiSolver( OutputFlag=0 ) )
    @variable(genData, y[1:n] )
    @objective(genData, Min, f(Q,c,y) )
    @constraint(genData, A*y .>= [zeros(num_s);zeros(num_edge);y[1:num_s];Dem[i,:];-y[1:num_s];-Dem[i,:];-ymax;-xmax]  )
    solve(genData)
    x[:,i] = getvalue(y)
    end
    return x
end
@everywhere function GenerateData2(n,Q,c,A,d,ymax,xmax,num_s,num_edge)
    genData = Model( solver=GurobiSolver( OutputFlag=0 ) )
    @variable(genData, y[1:n] )
    @objective(genData, Min, f(Q,c,y) )
    @constraint(genData, A*y .>= [zeros(num_s);zeros(num_edge);y[1:num_s];d;-y[1:num_s];-d;-ymax;-xmax] )
    solve(genData)
    return getvalue(y)
end
# n = 5; m = 6; ymax = [3;1]; xmax = 2*ones(m);
# Q = diagm([2;1;zeros(6)]);
# srand(1)
# c = [zeros(2);rand(Uniform(2.5,7.5),m)]; c_true = c;
# A = [1 1 0 0 0 0;0 0 1 1 0 0;-1 -0 -1 0 1 1;0 -1 0 0 -1 0; 0 0 0 -1 0 -1];
# A = [eye(2) zeros(2,m);zeros(m,2) eye(m);zeros(n,2) A; zeros(n,2) -A; -eye(2) zeros(2,m);zeros(m,2) -eye(m)];
# (m,n) = size(A);

n = 5; m = 6; num_s = 2; num_d = 3; num_edge = 6; ymax = [3;1.5]; xmax = 1.3*ones(m);
Q = diagm([2;10;zeros(m)]);
srand(1)
c = [zeros(num_s);rand(Uniform(1,10),m)]; c_true = c;
A = [1 1 0 0 0 0;0 0 1 1 0 0;-1 -0 -1 0 1 1;0 -1 0 0 -1 0; 0 0 0 -1 0 -1]; A = [zeros(n,num_s) A] - [eye(num_s) zeros(num_s,m);zeros(num_d,num_s) zeros(num_d,m)];
A = [eye(num_s) zeros(num_s,m);zeros(m,num_s) eye(m);A; -A; -eye(num_s) zeros(num_s,m);zeros(m,num_s) -eye(m)];
(m,n) = size(A);
T = 1000; estimation_error = Float64[]; cum_risk = Float64[];
for rep = 1:10
    # srand(1)
    D = rand(Uniform(-1.25,0),T,3);
    x = GenerateData(n,Q,c_true,A,D,T,ymax,xmax,num_s,num_edge);
    # srand(1)
    err = rand(Uniform(-0.25,0.25),n,T);
    y = x + err; # add noise
    # writedlm("C:/Users/CHD58/Desktop/y.txt", y, ' ') # observations
    #########################################################################
    M = 1e3; lambda = 2;
    c = zeros(n,1); c[1:4] = c_true[1:4]; c[7:8] = c_true[7:8];
    # c[3:end] = 2*ones(6);
    for t = 1:T
        println("rep = ", rep, ",", "t  = ", t, "\n")
        # t = t + 1
        estimation_error = push!(estimation_error, norm(c - c_true,2) )
        println("estimation_error = ", norm(c - c_true,2), "\n")
        xx = GenerateData2(n,Q,c,A,D[t,:],ymax,xmax,num_s,num_edge);
        cum_risk = push!(cum_risk, sqeuclidean(y[:,t], xx));
        # b = [zeros(num_s);zeros(num_edge);y[:,t][1:num_s];D[t,:];-y[:,t][1:num_s];-D[t,:];-ymax;-xmax];
        tic()
        c = qp_c_online(m,n,M,lambda*t^(-1/2),Q,c,A,D[t,:],y[:,t],ymax,xmax,c_true,num_s,num_edge);
        toc()
    end
end
# writedlm("C:/Users/CHD58/Desktop/estimation_error.txt", estimation_error, ' ')
# writedlm("C:/Users/CHD58/Desktop/cum_risk.txt", cum_risk, ' ')
