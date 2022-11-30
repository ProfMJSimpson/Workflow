using Plots, DifferentialEquations
using .Threads 
using Interpolations, Random, Distributions
using Roots, NLopt
gr()
a=zeros(3)
λ =0.01; K=100.0; C0=10.0; t=0:100:1000;
tt=0:5:1000;

function DE!(dC,C,p,t)
λ,K=p
dC[1]=λ*C[1]*(1.0-C[1]/K);
end

function odesolver(t,λ,K,C0)
p=(λ,K)
tspan=(0.0,maximum(t));
prob=ODEProblem(DE!,[C0],tspan,p);
sol=solve(prob,saveat=t);
return sol[1,:];
end

function model(t,a)
y=zeros(length(t))
y=odesolver(t,a[1],a[2],a[3])
return y
end

data0=zeros(length(t));
data=zeros(length(t));
data0=model(t,[λ,K,C0]);
#data_dists=[Normal(mi,1) for mi in model(t,[λ,K,C0])];
data_dists=[Poisson(mi) for mi in model(t,[λ,K,C0])];
data=[rand(data_dist) for data_dist in data_dists];

function loglhood(data,a)
    y=zeros(length(t))
    y=model(t,a);
    e=0;
    data_dists=[Poisson(mi) for mi in model(t,a)];
    #data_dists=[Normal(mi,1.) for mi in model(t,a)];
    e+=sum([loglikelihood(data_dists[i],data[i]) for i in 1:length(data_dists)])
    return e
end

λmin=0.0
λmax=0.05
Kmin=50
Kmax=150
C0min=0
C0max=50




function optimise(fun,θ₀,lb,ub;
    dv = false,
    method = dv ? :LD_LBFGS : :LN_BOBYQA,
)

if dv || String(method)[2] == 'D'
    tomax = fun
else
    tomax = (θ,∂θ) -> fun(θ)
end

opt = Opt(method,length(θ₀))
opt.max_objective = tomax
opt.lower_bounds = lb       # Lower bound
opt.upper_bounds = ub       # Upper bound
opt.local_optimizer = Opt(:LN_NELDERMEAD, length(θ₀))
res = optimize(opt,θ₀)
return res[[2,1]]
end

a=zeros(3)
function funmle(a)
return loglhood(data,a)
end

θG = [λ,K,C0]
lb=[λmin,Kmin,C0min]
ub=[λmax,Kmax,C0max]
(xopt,fopt)  = optimise(funmle,θG,lb,ub)
fmle=fopt
λmle=xopt[1]; 
Kmle=xopt[2]; 
C0mle=xopt[3]; 
ymle(t) = Kmle*C0mle/((Kmle-C0mle)*exp(-λmle*t)+C0mle);
p1=plot(ymle,0,1000,color=:turquoise1,xlabel="t",ylabel="C(t)",legend=false,lw=4,xlims=(0,1100),ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100])
p1=scatter!(t,data,legend=false,msw=0,ms=7,color=:darkorange,msa=:darkorange)
display(p1)
savefig(p1, "mle.pdf")



λmin=0.0
λmax=0.05
Kmin=80
Kmax=120
C0min=0.0
C0max=30


#Let's compute and push forward from the full 3D likelihood function

        
df=3
llstar=-quantile(Chisq(df),0.95)/2

N=100000
λs=rand(Uniform(λmin,λmax),N);
Ks=rand(Uniform(Kmin,Kmax),N);
C0s=rand(Uniform(C0min,C0max),N);
lls=zeros(N)
for i in 1:N
lls[i]=loglhood(data,[λs[i],Ks[i],C0s[i]])-fmle
end
q1=scatter(lls,legend=false)
q1=hline!([llstar],lw=2)
display(q1)
M=0

for i in 1:N
    if lls[i] >= llstar
        M+=1
    end

end

λsampled=zeros(M)
Ksampled=zeros(M)
C0sampled=zeros(M)
CtraceF = zeros(length(tt),M)
CUF=zeros(length(tt))
CLF=zeros(length(tt))
j=0
for i in 1:N
    if lls[i] > llstar
    global j = j + 1
    λsampled[j]=λs[i]
    Ksampled[j]=Ks[i]
    C0sampled[j]=C0s[i]
    CtraceF[:,j]=model(tt,[λs[i],Ks[i],C0s[i]]);
    end
end


for i in 1:length(tt)
CUF[i] = maximum(CtraceF[i,:])
CLF[i] = minimum(CtraceF[i,:])
end





qq1=plot(tt,CtraceF[:,:],color=:grey,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
pp1=plot!(ymle,0,1000,lw=3,color=:turquoise1,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
qq1=plot!(tt,CUF,lw=3,color=:gold,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
qq1=plot!(tt,CLF,lw=3,color=:gold,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],xlabel="t",ylabel="C(t)",legend=false)


#Compute univariate and push forward
df=1
llstar=-quantile(Chisq(df),0.95)/2
    
function univariateλ(λ)   
function funλ(a)
return loglhood(data,[λ,a[1],a[2]])
end
θG=[K,C0]
lb=[Kmin,C0min]
ub=[Kmax,C0max]
(xopt,fopt)=optimise(funλ,θG,lb,ub)
llb=fopt-fmle
return llb,xopt
end 
f(x) = univariateλ(x)[1]
M=100;
λrange=LinRange(0.0,0.03,M)
ff=zeros(M)
for i in 1:M
    ff[i]=univariateλ(λrange[i])[1]
end

q1=plot(λrange,ff,ylims=(-3,0),xlims=(0.005,0.02),legend=false,lw=3)
q1=hline!([llstar],legend=false,lw=3)
q1=vline!([λmle],legend=false,xlabel="λ",ylabel="ll",lw=3)



g(x)=f(x)[1]-llstar
ϵ=(λmax-λmin)/10^6
x0=λmle
x1=λmin
x2=(x1+x0)/2;
while abs(x1-x0) > ϵ
x2=(x1+x0)/2;
if g(x0)*g(x2) < 0 
 x1=x2
 else
 x0=x2
 end
 end
λλmin = x2
x0=λmle
x1=λmax
x2=(x1+x0)/2;
while abs(x1-x0) > ϵ
x2=(x1+x0)/2;
if g(x0)*g(x2) < 0 
 x1=x2
 else
 x0=x2
 end
 end
λλmax = x2
N=100
λsampled=zeros(N)
Ksampled=zeros(N)
C0sampled=zeros(N)

λsampled=LinRange(λλmin,λλmax,N)
for i in 1:N
Ksampled[i]=univariateλ(λsampled[i])[2][1]
C0sampled[i]=univariateλ(λsampled[i])[2][2]
end

CUnivariatetrace1 = zeros(length(tt),N)
CU1=zeros(length(tt))
CL1=zeros(length(tt))
for i in 1:N
CUnivariatetrace1[:,i]=model(tt,[λsampled[i],Ksampled[i],C0sampled[i]]);
end



for i in 1:length(tt)
CU1[i] = maximum(CUnivariatetrace1[i,:])
CL1[i] = minimum(CUnivariatetrace1[i,:])
end
    


pp1=plot(tt,CUnivariatetrace1[:,:],color=:grey,xlims=(0,1000),legend=false)
pp1=plot!(tt,CU1,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
pp1=plot!(tt,CL1,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
pp1=plot!(ymle,0,1000,lw=3,color=:turquoise1,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
display(pp1)
savefig(pp1, "UnivatiatePredictionPoissonL.pdf")





  
function univariateK(K)
 a=zeros(2)    
 function funK(a)
 return loglhood(data,[a[1],K,a[2]])
 end
     θG=[λ,C0]
     lb=[λmin,C0min]
     ub=[λmax,C0max]
     (xopt,fopt)=optimise(funK,θG,lb,ub)
     llb=fopt-fmle
     N=xopt
     return llb,N
     end 
     f(x) = univariateK(x)[1]
     M=100;
     Krange=LinRange(Kmin,Kmax,M)
     ff=zeros(M)
     for i in 1:M
         ff[i]=univariateK(Krange[i])[1]
     end
     
     q2=plot(Krange,ff,ylims=(-3,0.),xlims=(80,120),legend=false,lw=3)
     q2=hline!([llstar],legend=false,lw=3)
     q2=vline!([Kmle],legend=false,xlabel="K",ylabel="ll",lw=3)



  
g(x)=f(x)[1]-llstar
ϵ=(Kmax-Kmin)/10^6
x0=Kmle
x1=Kmin
x2=(x1+x0)/2;
while abs(x1-x0) > ϵ
x2=(x1+x0)/2;
if g(x0)*g(x2) < 0 
 x1=x2
 else
 x0=x2
 end
 end
KKmin = x2
x0=Kmle
x1=Kmax
x2=(x1+x0)/2;
while abs(x1-x0) > ϵ
x2=(x1+x0)/2;
if g(x0)*g(x2) < 0 
 x1=x2
 else
 x0=x2
 end
 end
KKmax = x2

λsampled=zeros(N)
Ksampled=zeros(N)
C0sampled=zeros(N)

Ksampled=LinRange(KKmin,KKmax,N)
for i in 1:N
λsampled[i]=univariateK(Ksampled[i])[2][1]
C0sampled[i]=univariateK(Ksampled[i])[2][2]
end




CUnivariatetrace2 = zeros(length(tt),N)
CU2=zeros(length(tt))
CL2=zeros(length(tt))
for i in 1:N
CUnivariatetrace2[:,i]=model(tt,[λsampled[i],Ksampled[i],C0sampled[i]]);
end



for i in 1:length(tt)
CU2[i] = maximum(CUnivariatetrace2[i,:])
CL2[i] = minimum(CUnivariatetrace2[i,:])
end
    

pp1=plot(tt,CUnivariatetrace2[:,:],color=:grey,xlims=(0,1000),legend=false)
pp1=plot!(tt,CU2,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
pp1=plot!(tt,CL2,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
pp1=plot!(ymle,0,1000,lw=3,color=:turquoise1,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
display(pp1)
savefig(pp1, "UnivatiatePredictionPoissonK.pdf")





 function univariateC0(C0)
 a=zeros(2)    
 function funC0(a)
 return loglhood(data,[a[1],a[2],C0])
 end
 θG=[λ,K]
 lb=[λmin,Kmin]
 ub=[λmax,Kmax]
 (xopt,fopt)=optimise(funC0,θG,lb,ub)
 llb=fopt-fmle
 return llb,xopt
end 
 f(x) = univariateC0(x)[1]
 M=100;
 C0range=LinRange(0,C0max,M)
 ff=zeros(M)
 for i in 1:M
     ff[i]=univariateC0(C0range[i])[1]
 end
 
 q3=plot(C0range,ff,ylims=(-3,0.),xlims=(C0min,30),legend=false,lw=3)
 q3=hline!([llstar],legend=false,lw=3)
 q3=vline!([C0mle],legend=false,xlabel="C(0)",ylabel="ll",lw=3)


q4=plot(q1,q2,q3,layout=(1,3),legend=false)
display(q4)
savefig(q4, "univariatePoisson.pdf") 



  
g(x)=f(x)[1]-llstar
ϵ=(C0max-C0min)/10^6
x0=C0mle
x1=C0min
x2=(x1+x0)/2;
while abs(x1-x0) > ϵ
x2=(x1+x0)/2;
if g(x0)*g(x2) < 0 
 x1=x2
 else
 x0=x2
 end
 end
CC0min = x2
x0=C0mle
x1=C0max
x2=(x1+x0)/2;
while abs(x1-x0) > ϵ
x2=(x1+x0)/2;
if g(x0)*g(x2) < 0 
 x1=x2
 else
 x0=x2
 end
 end
CC0max = x2

λsampled=zeros(N)
Ksampled=zeros(N)
C0sampled=zeros(N)

C0sampled=LinRange(CC0min,CC0max,N)
for i in 1:N
λsampled[i]=univariateC0(C0sampled[i])[2][1]
Ksampled[i]=univariateC0(C0sampled[i])[2][2]
end




CUnivariatetrace3 = zeros(length(tt),N)
CU3=zeros(length(tt))
CL3=zeros(length(tt))
for i in 1:N
CUnivariatetrace3[:,i]=model(tt,[λsampled[i],Ksampled[i],C0sampled[i]]);
end



for i in 1:length(tt)
CU3[i] = maximum(CUnivariatetrace3[i,:])
CL3[i] = minimum(CUnivariatetrace3[i,:])
end
    

pp1=plot(tt,CUnivariatetrace3[:,:],color=:grey,xlims=(0,1000),legend=false)
pp1=plot!(tt,CU3,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
pp1=plot!(tt,CL3,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
pp1=plot!(ymle,0,1000,lw=3,color=:turquoise1,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
display(pp1)
savefig(pp1, "UnivatiatePredictionPoissonC0.pdf")



 CU=zeros(length(tt))
 CL=zeros(length(tt))

 for i in 1:length(tt)
 CU[i]=max(CU1[i],CU2[i],CU3[i])
 CL[i]=min(CL1[i],CL2[i],CL3[i])
 end



qq1=plot(tt,CtraceF[:,:],color=:grey,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
qq1=plot!(tt,CUF,lw=3,color=:gold,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
qq1=plot!(tt,CLF,lw=3,color=:gold,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],xlabel="t",ylabel="C(t)",legend=false)
qq1=plot!(tt,CU,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
qq1=plot!(tt,CL,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],xlabel="t",ylabel="C(t)",legend=false)
qq1=plot!(ymle,0,1000,lw=3,color=:turquoise1,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
display(qq1)
savefig(qq1, "PredictionComparisonPoissonUni.pdf")



    
df=2
llstar=-quantile(Chisq(df),0.95)/2
#Construct bivariates
function bivariateλK(λ,K)
 function funλK(a)
    return loglhood(data,[λ,K,a[1]])
    end
    θG = [C0]
    lb=[C0min]
    ub=[C0max]
    (xopt,fopt)  = optimise(funλK,θG,lb,ub)
llb=fopt-fmle
return llb,xopt[1]
end 

f(x,y) = bivariateλK(x,y)


g(x,y)=f(x,y)[1]-llstar
ϵ=(Kmax-Kmin)/10^8

N=100
λsamples_boundary=zeros(2*N)
Ksamples_boundary=zeros(2*N)
C0samples_boundary=zeros(2*N)
count=0

while count < N
x=rand(Uniform(λmin,λmax))
y0=rand(Uniform(Kmin,Kmax))
y1=rand(Uniform(Kmin,Kmax))

if g(x,y0)*g(x,y1) < 0 
count+=1
println(count)
while abs(y1-y0) > ϵ && y1 < Kmax && y1 > Kmin
y2=(y1+y0)/2;
    if g(x,y0)*g(x,y2) < 0 
    y1=y2
    else
    y0=y2
    end


end

λsamples_boundary[count]=x;
Ksamples_boundary[count]=y1;
C0samples_boundary[count]=f(x,y1)[2]
end
end 

ϵ=(λmax-λmin)/10^6
count=0
while count < N
y=rand(Uniform(Kmin,Kmax))
x0=rand(Uniform(λmin,λmax))
x1=rand(Uniform(λmin,λmax))
    
if g(x0,y)*g(x1,y) < 0 
count+=1
println(count)

while abs(x1-x0) > ϵ && x1 < λmax && x1 > λmin
    x2=(x1+x0)/2;
        if g(x0,y)*g(x2,y) < 0 
        x1=x2
        else
        x0=x2
        end
    
    
    end


    λsamples_boundary[N+count]=x1;
    Ksamples_boundary[N+count]=y;
    C0samples_boundary[N+count]=f(x1,y)[2]
    end
    end 
    
a1=scatter([λmle],[Kmle],xlims=(λmin,λmax),ylims=(Kmin,Kmax),markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,xlabel="λ",ylabel="K",label=false)
display(a1)
for i in 1:2*N
a1=scatter!([λsamples_boundary[i]],[Ksamples_boundary[i]],xlims=(0,0.03),ylims=(80,120),markersize=3,markershape=:circle,markercolor=:blue,msw=0, ms=5,label=false)
end
display(a1)


Ctrace1_boundary = zeros(length(tt),2*N)
CU1_boundary=zeros(length(tt))
CL1_boundary=zeros(length(tt))
for i in 1:2*N
Ctrace1_boundary[:,i]=model(tt,[λsamples_boundary[i],Ksamples_boundary[i],C0samples_boundary[i]]);
end
    
    
for i in 1:length(tt)
CU1_boundary[i] = maximum(Ctrace1_boundary[i,:])
CL1_boundary[i] = minimum(Ctrace1_boundary[i,:])
end
    



pp1=plot(tt,Ctrace1_boundary[:,:],color=:grey,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
pp1=plot!(ymle,0,1000,lw=3,color=:turquoise1,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
pp1=plot!(tt,CU1_boundary,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
pp1=plot!(tt,CL1_boundary,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],xlabel="t",ylabel="C(t)",legend=false)
pp3=plot(a1,pp1,layout=(1,2))
display(pp3)
savefig(pp3, "bivariateLK_boundary_Poisson.pdf")



function bivariateλC0(λ,C0)
    function funλC0(a)
    return loglhood(data,[λ,a[1],C0])
    end
    θG = [K]
    lb=[Kmin]
    ub=[Kmax]
    (xopt,fopt)  = optimise(funλC0,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt[1]
    end 
        
f(x,y) = bivariateλC0(x,y)
g(x,y)=f(x,y)[1]-llstar
ϵ=(C0max-C0min)/10^8
λsamples_boundary=zeros(2*N)
C0samples_boundary=zeros(2*N)
Ksamples_boundary=zeros(2*N)
count=0
        
while count < N 
x=rand(Uniform(λmin,λmax))
y0=rand(Uniform(C0min,C0max))
y1=rand(Uniform(C0min,C0max))

        
if g(x,y0)*g(x,y1) < 0 
count+=1
println(count)
while abs(y1-y0) > ϵ && y1 < C0max && y1 > C0min
y2=(y1+y0)/2;
    if g(x,y0)*g(x,y2) < 0 
    y1=y2
    else
    y0=y2
    end
        
        
end
        
λsamples_boundary[count]=x;
C0samples_boundary[count]=y1;
Ksamples_boundary[count]=f(x,y1)[2]
end
end 
        
ϵ=(λmax-λmin)/10^6
count=0
while count < N 
y=rand(Uniform(C0min,C0max))
x0=rand(Uniform(λmin,λmax))
x1=rand(Uniform(λmin,λmax))
            
if g(x0,y)*g(x1,y) < 0 
count+=1
println(count)
        
while abs(x1-x0) > ϵ && x1 < λmax && x1 > λmin
    x2=(x1+x0)/2;
    if g(x0,y)*g(x2,y) < 0 
    x1=x2
    else
    x0=x2
end
            
            
end
        
        
λsamples_boundary[N+count]=x1;
C0samples_boundary[N+count]=y;
Ksamples_boundary[N+count]=f(x1,y)[2]
end
end 


  
a2=scatter([λmle],[C0mle],xlims=(λmin,λmax),ylims=(C0min,C0max),markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,xlabel="λ",ylabel="C(0)",label=false)
display(a2)
for i in 1:2*N
a2=scatter!([λsamples_boundary[i]],[C0samples_boundary[i]],xlims=(0,0.03),ylims=(0,35),xticks=[0,0.015,0.03],yticks=[0,15,30],markersize=3,markershape=:circle,markercolor=:blue,msw=0, ms=5,label=false)
end
display(a2)



Ctrace2_boundary = zeros(length(tt),2*N)
CU2_boundary=zeros(length(tt))
CL2_boundary=zeros(length(tt))
for i in 1:2*N
Ctrace2_boundary[:,i]=model(tt,[λsamples_boundary[i],Ksamples_boundary[i],C0samples_boundary[i]]);
end
            
            
for i in 1:length(tt)
    CU2_boundary[i] = maximum(Ctrace2_boundary[i,:])
    CL2_boundary[i] = minimum(Ctrace2_boundary[i,:])
end
            
        
pp1=plot(tt,Ctrace2_boundary[:,:],color=:grey,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
pp1=plot!(ymle,0,1000,lw=3,color=:turquoise1,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
pp1=plot!(tt,CU2_boundary,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
pp1=plot!(tt,CL2_boundary,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],xlabel="t",ylabel="C(t)",legend=false)
pp3=plot(a2,pp1,layout=(1,2))
display(pp3)
savefig(pp3, "bivariateLC0_boundary_Poisson.pdf") 
            


# Q=20;
# λλ=LinRange(0.0,0.03,Q);
# CC0=LinRange(0,35,Q);


#aa2=contourf(λλ,CC0,(λλ,CC0)->f(λλ,CC0)[1],lw=0,xlabel="λ",ylabel="C(0)",c=:greens,colorbar=false)
# aa2=contour(λλ,CC0,(λλ,CC0)->f(λλ,CC0)[1],levels=[llstar],lw=4,xlabel="λ",ylabel="C(0)",c=:red,colorbar=false)
# aa2=scatter!([λmle],[C0mle],markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,label=false)
# for ii in 1:length(λλ)
#     for jj in 1:length(CC0)
#     aa1=scatter!([λλ[ii]],[CC0[jj]],markersize=2,markershape=:x,markercolor=:grey,msw=0,label=false)
#     end
# end
# display(aa2)


# λsamples_grid=zeros(Q^2)
# Ksamples_grid=zeros(Q^2)
# C0samples_grid=zeros(Q^2)
# llsamples_grid=zeros(Q^2)

# count=0
# for i in 1:Q
#     for j in 1:Q
#     count+=1
#     λsamples_grid[count]=λλ[i]
#     C0samples_grid[count]=CC0[j]
#     Ksamples_grid[count]=f(λλ[i],CC0[j])[2]
#     llsamples_grid[count]=loglhood(data,[λsamples_grid[count],Ksamples_grid[count],C0samples_grid[count]],σ)-fmle   
#     end
# end


# Ctrace2_grid = zeros(length(tt),Q^2)
# count=0
# for i in 1:Q^2
#         if llsamples_grid[i] > llstar  
#             count+=1  
#         Ctrace2_grid[:,count]=model(tt,[λsamples_grid[i],Ksamples_grid[i],C0samples_grid[i]],σ);
#         end
# end

# Ctrace_withingrid2=zeros(length(tt),count)
# CU2_grid=zeros(length(tt))
# CL2_grid=zeros(length(tt))

# for i in 1:count
# Ctrace_withingrid2[:,i]=Ctrace2_grid[:,i]
# end 

  
#     for i in 1:length(tt)
#     CU2_grid[i] = maximum(Ctrace_withingrid2[i,:])
#     CL2_grid[i] = minimum(Ctrace_withingrid2[i,:])
#     end
        
    
    
    
#     pp1=plot(tt,Ctrace_withingrid2[:,:],color=:grey,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
#     pp1=plot!(ymle,0,1000,lw=3,color=:turquoise1,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
#     pp1=plot!(tt,CU2_grid,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
#     pp1=plot!(tt,CL2_grid,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],xlabel="t",ylabel="C(t)",legend=false)
#     pp3=plot(aa2,pp1,layout=(1,2))
#     display(pp3)
#     savefig(pp3, "bivariateLC0_grid.pdf")




function bivariateKC0(K,C0)
function funKC0(a)
return loglhood(data,[a[1],K,C0])
end
θG = [λ]
lb=[λmin]
ub=[λmax]
(xopt,fopt)  = optimise(funKC0,θG,lb,ub)
llb=fopt-fmle
return llb,xopt[1]
end 
        
f(x,y) = bivariateKC0(x,y)
g(x,y)=f(x,y)[1]-llstar
ϵ=(C0max-C0min)/10^8
Ksamples_boundary=zeros(2*N)
C0samples_boundary=zeros(2*N)
λsamples_boundary=zeros(2*N)
count=0
        
while count < N
x=rand(Uniform(Kmin,Kmax))
y0=rand(Uniform(C0min,C0max))
y1=rand(Uniform(C0min,C0max))
        
if g(x,y0)*g(x,y1) < 0 
count+=1
        println(count)
        while abs(y1-y0) > ϵ && y1 < C0max && y1 > C0min
        y2=(y1+y0)/2;
            if g(x,y0)*g(x,y2) < 0 
            y1=y2
            else
            y0=y2
            end
        
        
        end
        
        Ksamples_boundary[count]=x;
        C0samples_boundary[count]=y1;
        λsamples_boundary[count]=f(x,y1)[2]
        end
        end 
        
        ϵ=1(Kmax-Kmin)/10^8
        count=0
        while count < N 
        y=rand(Uniform(C0min,C0max))
        x0=rand(Uniform(Kmin,Kmax))
        x1=rand(Uniform(Kmin,Kmax))
            
        if g(x0,y)*g(x1,y) < 0 
        count+=1
        println(count)
        
        while abs(x1-x0) > ϵ && x1 < Kmax && x1 > Kmin
            x2=(x1+x0)/2;
                if g(x0,y)*g(x2,y) < 0 
                x1=x2
                else
                x0=x2
                end
            
            
            end
        
        
            Ksamples_boundary[N+count]=x1;
            C0samples_boundary[N+count]=y;
            λsamples_boundary[N+count]=f(x1,y)[2]
            end
            end 



  
            a3=scatter([Kmle],[C0mle],xlims=(Kmin,Kmax),ylims=(C0min,C0max),markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,xlabel="K",ylabel="C(0)",label=false)
            display(a3)
            for i in 1:2*N
            a3=scatter!([Ksamples_boundary[i]],[C0samples_boundary[i]],xlims=(80,120),ylims=(C0min,35),xticks=[80,100,120],yticks=[0,15,30],markersize=3,markershape=:circle,markercolor=:blue,msw=0, ms=5,label=false)
            end
            display(a3)



            
            Ctrace3_boundary = zeros(length(tt),2*N)
            CU3_boundary=zeros(length(tt))
            CL3_boundary=zeros(length(tt))
            for i in 1:2*N
            Ctrace3_boundary[:,i]=model(tt,[λsamples_boundary[i],Ksamples_boundary[i],C0samples_boundary[i]]);
            end
            
            
            for i in 1:length(tt)
            CU3_boundary[i] = maximum(Ctrace3_boundary[i,:])
            CL3_boundary[i] = minimum(Ctrace3_boundary[i,:])
            end
            
        
        

            pp1=plot(tt,Ctrace3_boundary[:,:],color=:grey,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
            pp1=plot!(ymle,0,1000,lw=3,color=:turquoise1,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
            pp1=plot!(tt,CU3_boundary,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
            pp1=plot!(tt,CL3_boundary,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],xlabel="t",ylabel="C(t)",legend=false)
            pp3=plot(a3,pp1,layout=(1,2))
            display(pp3)
            savefig(pp3, "bivariateKC0_boundary_Poisson.pdf") 
     



            # Q=20;
            # KK=LinRange(80,120,Q);
            # CC0=LinRange(0,35,Q);
            
            
            #aa3=contourf(KK,CC0,(KK,CC0)->f(KK,CC0)[1],lw=0,xlabel="K",ylabel="C(0)",c=:greens,colorbar=false)
            # aa3=contour(KK,CC0,(KK,CC0)->f(KK,CC0)[1],levels=[llstar],lw=4,xlabel="K",ylabel="C(0)",c=:red,colorbar=false)
            # aa3=scatter!([Kmle],[C0mle],markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,label=false)
            # for ii in 1:length(KK)
            #     for jj in 1:length(CC0)
            #     aa3=scatter!([KK[ii]],[CC0[jj]],markersize=2,markershape=:x,markercolor=:grey,msw=0,label=false)
            #     end
            # end
            # display(aa3)
            
            
            # λsamples_grid=zeros(Q^2)
            # Ksamples_grid=zeros(Q^2)
            # C0samples_grid=zeros(Q^2)
            # llsamples_grid=zeros(Q^2)
            
            # count=0
            # for i in 1:Q
            #     for j in 1:Q
            #     count+=1
            #     Ksamples_grid[count]=KK[i]
            #     C0samples_grid[count]=CC0[j]
            #     λsamples_grid[count]=f(KK[i],CC0[j])[2]
            #     llsamples_grid[count]=loglhood(data,[λsamples_grid[count],Ksamples_grid[count],C0samples_grid[count]],σ)-fmle   
            #     end
            # end
            
            
            # Ctrace3_grid = zeros(length(tt),Q^2)
            # count=0
            # for i in 1:Q^2
            #         if llsamples_grid[i] > llstar  
            #             count+=1  
            #         Ctrace3_grid[:,count]=model(tt,[λsamples_grid[i],Ksamples_grid[i],C0samples_grid[i]],σ);
            #         end
            # end
            
            # Ctrace_withingrid3=zeros(length(tt),count)
            # CU3_grid=zeros(length(tt))
            # CL3_grid=zeros(length(tt))
            
            # for i in 1:count
            # Ctrace_withingrid3[:,i]=Ctrace3_grid[:,i]
            # end 
            
              
            #     for i in 1:length(tt)
            #     CU3_grid[i] = maximum(Ctrace_withingrid3[i,:])
            #     CL3_grid[i] = minimum(Ctrace_withingrid3[i,:])
            #     end
                    
                
                
                
            #     pp1=plot(tt,Ctrace_withingrid3[:,:],color=:grey,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
            #     pp1=plot!(ymle,0,1000,lw=3,color=:turquoise1,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
            #     pp1=plot!(tt,CU3_grid,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
            #     pp1=plot!(tt,CL3_grid,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],xlabel="t",ylabel="C(t)",legend=false)
            #     pp1=plot(aa3,pp1,layout=(1,2))
            #     display(pp1)
            #     savefig(pp1, "bivariateKC0_grid.pdf")
            
            





# CU_grid=zeros(length(tt))
# CL_grid=zeros(length(tt))
# for i in 1:length(tt)
# CU_grid[i]=max(CU1_grid[i],CU2_grid[i],CU3_grid[i])
# CL_grid[i]=min(CL1_grid[i],CL2_grid[i],CL3_grid[i])
# end


CU_boundary=zeros(length(tt))
CL_boundary=zeros(length(tt))
for i in 1:length(tt)
CU_boundary[i]=max(CU1_boundary[i],CU2_boundary[i],CU3_boundary[i])
CL_boundary[i]=min(CL1_boundary[i],CL2_boundary[i],CL3_boundary[i])
end


qq1=plot(tt,CtraceF[:,:],color=:grey,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
qq1=plot!(ymle,0,1000,lw=3,color=:turquoise1,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
qq1=plot!(tt,CUF,lw=3,color=:gold,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
qq1=plot!(tt,CLF,lw=3,color=:gold,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],xlabel="t",ylabel="C(t)",legend=false)
qq1=plot!(tt,CU_boundary,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
qq1=plot!(tt,CL_boundary,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
display(qq1)
savefig(qq1, "Bivariatecomparison_boundary_Poisson.pdf")




# qq1=plot(tt,CtraceF[:,:],color=:grey,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
# qq1=plot!(ymle,0,1000,lw=3,color=:turquoise1,xlabel="t",ylabel="C(t)",ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
# qq1=plot!(tt,CUF,lw=3,color=:gold,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
# qq1=plot!(tt,CLF,lw=3,color=:gold,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],xlabel="t",ylabel="C(t)",legend=false)
# qq1=plot!(tt,CU_grid,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
# qq1=plot!(tt,CL_grid,lw=3,color=:red,ylims=(0,120),xticks=[0,500,1000],yticks=[0,50,100],legend=false)
# display(qq1)
# savefig(qq1, "Bivariatecomparison_grid.pdf")