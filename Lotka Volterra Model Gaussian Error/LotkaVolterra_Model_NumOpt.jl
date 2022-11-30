using Plots, DifferentialEquations
using .Threads 
using Interpolations, Random, Distributions
using Roots, NLopt
gr()
a=zeros(4)
α = 0.9; β=1.1; x0=0.8; y0=0.3; 
t=LinRange(0,10,21);
tt=LinRange(0,10,2001)
σ=0.2;



function DE!(dC,C,p,t)
α,β=p
dC[1]=α*C[1]-C[1]*C[2];
dC[2]=β*C[1]*C[2]-C[2];
end


function odesolver(t,α,β,C01,C02)
p=(α,β)
C0=[C01,C02]
tspan=(0.0,maximum(t))
prob=ODEProblem(DE!,C0,tspan,p)
sol=solve(prob,saveat=t);
cc1=sol[1,:]
cc2=sol[2,:]
tt=sol.t[:]
return cc1,cc2
end

function model(t,a,σ)
x=zeros(length(t))
y=zeros(length(t))
(x,y)=odesolver(t,a[1],a[2],a[3],a[4])
return x,y
end

datax0=zeros(length(t));
datay0=zeros(length(t));
datax=zeros(length(t));
datay=zeros(length(t));
(datax0,datay0)=model(t,[α,β,x0,y0],σ);
datax=datax0+σ*randn(length(t));
datay=datay0+σ*randn(length(t));




function loglhood(datax,datay,a,σ)
x=zeros(length(t))
y=zeros(length(t))
(x,y)=model(t,a,σ);
e=0.0;
f=0.0;
dist=Normal(0,σ);
    for i in 1:15   
    e+=loglikelihood(dist,datax[i]-x[i])  
    f+=loglikelihood(dist,datay[i]-y[i])
    end
return e+f
end



αmin=0.7
αmax=1.2
βmin=0.7
βmax=1.4
x0min=0.5
x0max=1.2
y0min=0.1
y0max=0.5






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

a=zeros(4)
function funmle(a)
return loglhood(datax,datay,a,σ)
end

θG = [α,β,x0,y0]
lb=[αmin,βmin,x0min,y0min]
ub=[αmax,βmax,x0max,y0max]
(xopt,fopt)  = optimise(funmle,θG,lb,ub)
fmle=fopt
αmle=xopt[1];
βmle=xopt[2];
x0mle=xopt[3]; 
y0mle=xopt[4]; 



(xxmle,yymle)=odesolver(tt,xopt[1],xopt[2],xopt[3],xopt[4])
w1=scatter(t,datax,msw=0,ms=7,color=:coral,msa=:coral,legend=false)
w1=plot!(tt,xxmle,lw=2,xlabel="t",ylabel="x(t)",legend=false,xlims=(0,10),ylims=(0,2.5))
w2=scatter(t,datay,msw=0,ms=7,color=:lime,msa=:lime,legend=false)
w2=plot!(tt,yymle,lw=2,xlabel="t",ylabel="y(t)",legend=false,xlims=(0,10),ylims=(0,2.5))
w3=plot(w1,w2,layout=(1,2),legend=false)
display(w3)
savefig(w3, "mle.pdf")


df=2
llstar=-quantile(Chisq(df),0.95)/2



function bivariateαβ(α,β)
 function funαβ(a)
    return loglhood(datax,datay,[α,β,a[1],a[2]],σ)
    end
    θG = [x0mle,y0mle]
    lb=[x0min,y0min]
    ub=[x0max,y0max]
    (xopt,fopt)  = optimise(funαβ,θG,lb,ub)
llb=fopt-fmle
return llb,xopt
end 

f(x,y) = bivariateαβ(x,y)
g(x,y)=f(x,y)[1]-llstar

ϵ=(βmax-βmin)/10^3
N=50
αsamples=zeros(2*N)
βsamples=zeros(2*N)
x0samples=zeros(2*N)
y0samples=zeros(2*N)
count=0

while count < N
x=rand(Uniform(αmin,αmax))
y0=rand(Uniform(βmin,βmax))
y1=rand(Uniform(βmin,βmax))

if g(x,y0)*g(x,y1) < 0 
count+=1
println(count)
while abs(y1-y0) > ϵ && y1 < βmax && y1 > βmin
y2=(y1+y0)/2;
    if g(x,y0)*g(x,y2) < 0 
    y1=y2
    else
    y0=y2
    end


end

αsamples[count]=x;
βsamples[count]=y1;
x0samples[count]=f(x,y1)[2][1]
y0samples[count]=f(x,y1)[2][2]
end
end 

ϵ=(αmax-αmin)/10^3
count=0
while count < N
y=rand(Uniform(βmin,βmax))
x0=rand(Uniform(αmin,αmax))
x1=rand(Uniform(αmin,αmax))
    
if g(x0,y)*g(x1,y) < 0 
count+=1
println(count)

while abs(x1-x0) > ϵ && x1 < αmax && x1 > αmin
    x2=(x1+x0)/2;
        if g(x0,y)*g(x2,y) < 0 
        x1=x2
        else
        x0=x2
        end
    
    
    end


    αsamples[N+count]=x1;
    βsamples[N+count]=y;
    x0samples[N+count]=f(x1,y)[2][1]
    y0samples[N+count]=f(x1,y)[2][2]
  
    end
    end 
    


        
a1=scatter([αmle],[βmle],xlims=(αmin,αmax),ylims=(βmin,βmax),markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,xlabel="α",ylabel="β",label=false)
for i in 1:2*N
a1=scatter!([αsamples[i]],[βsamples[i]],xlims=(0.5,1.5),ylims=(0.5,1.5),markersize=3,markershape=:circle,markercolor=:blue,msw=0, ms=5,label=false)
end
display(a1)

    xtrace1 = zeros(length(tt),2*N);
    ytrace1 = zeros(length(tt),2*N);

    xU1=zeros(length(tt))
    xL1=zeros(length(tt))

    yU1=zeros(length(tt))
    yL1=zeros(length(tt))


    for i in 1:2*N
    (xtrace1[:,i],ytrace1[:,i])=odesolver(tt,αsamples[i],βsamples[i],x0samples[i],y0samples[i]);
    end
    
    
    for i in 1:length(tt)
    xU1[i] = maximum(xtrace1[i,:])
    xL1[i] = minimum(xtrace1[i,:])
    yU1[i] = maximum(ytrace1[i,:])
    yL1[i] = minimum(ytrace1[i,:])
    end
    


    pp1=plot(tt,xtrace1[:,:],color=:grey,xlabel="t",ylabel="x(t)",legend=false)
    pp1=plot!(tt,xU1,color=:red,lw=2,legend=false)
    pp1=plot!(tt,xL1,color=:red,lw=2,legend=false,xlims=(0,10),ylims=(0,2.5))
    pp2=plot(tt,ytrace1[:,:],color=:grey,xlabel="t",ylabel="y(t)",legend=false)
    pp2=plot!(tt,yU1,color=:red,lw=2,legend=false)
    pp2=plot!(tt,yL1,color=:red,lw=2,legend=false,xlims=(0,10),ylims=(0,2.5))
    pp3=plot(a1,pp1,pp2,layout=(1,3))
    display(pp3)
    savefig(pp3, "bivariateAB.pdf")



    

function bivariateαx0(α,x0)
     function funαx0(a)
        return loglhood(datax,datay,[α,a[1],x0,a[2]],σ)
        end
        θG = [βmle,y0mle]
        lb=[βmin,y0min]
        ub=[βmax,y0max]
        (xopt,fopt)  = optimise(funαx0,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt
    end 
    
    f(x,y) = bivariateαx0(x,y)
    g(x,y)=f(x,y)[1]-llstar
 
    
    
    ϵ=(x0max-x0min)/10^3
    αsamples=zeros(2*N)
    βsamples=zeros(2*N)
    x0samples=zeros(2*N)
    y0samples=zeros(2*N)
    count=0
    
    while count < N
    x=rand(Uniform(αmin,αmax))
    y0=rand(Uniform(x0min,x0max))
    y1=rand(Uniform(x0min,x0max))
    
    if g(x,y0)*g(x,y1) < 0 
    count+=1
    println(count)
    while abs(y1-y0) > ϵ && y1 < x0max && y1 > x0min
    y2=(y1+y0)/2;
        if g(x,y0)*g(x,y2) < 0 
        y1=y2
        else
        y0=y2
        end
    
    
    end
    
    αsamples[count]=x;
    x0samples[count]=y1;
    βsamples[count]=f(x,y1)[2][1]
    y0samples[count]=f(x,y1)[2][2]
    end
    end 
    
    ϵ=(x0max-x0min)/10^3
    count=0
    while count < N
    y=rand(Uniform(x0min,x0max))
    x0=rand(Uniform(αmin,αmax))
    x1=rand(Uniform(αmin,αmax))
        
    if g(x0,y)*g(x1,y) < 0 
    count+=1
    println(count)
    
    while abs(x1-x0) > ϵ  && x1 < αmax && x1 > αmin
        x2=(x1+x0)/2;
            if g(x0,y)*g(x2,y) < 0 
            x1=x2
            else
            x0=x2
            end
        
        
        end
    
    
        αsamples[N+count]=x1;
        x0samples[N+count]=y;
        βsamples[N+count]=f(x1,y)[2][1]
        y0samples[N+count]=f(x1,y)[2][2]
       
        end
        end 
        
    

        
        a2=scatter([αmle],[x0mle],xlims=(αmin,αmax),ylims=(x0min,x0max),markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,xlabel="α",ylabel="x(0)",label=false)
        for i in 1:2*N
        a2=scatter!([αsamples[i]],[x0samples[i]],xlims=(αmin,αmax),ylims=(x0min,x0max),markersize=3,markershape=:circle,markercolor=:blue,msw=0, ms=5,label=false)
        end
        display(a2)



        xtrace2 = zeros(length(tt),2*N);
        ytrace2 = zeros(length(tt),2*N);
    
        xU2=zeros(length(tt))
        xL2=zeros(length(tt))
    
        yU2=zeros(length(tt))
        yL2=zeros(length(tt))
    
    
        for i in 1:2*N
        (xtrace2[:,i],ytrace2[:,i])=odesolver(tt,αsamples[i],βsamples[i],x0samples[i],y0samples[i]);
        end
        
        
        for i in 1:length(tt)
        xU2[i] = maximum(xtrace2[i,:])
        xL2[i] = minimum(xtrace2[i,:])
        yU2[i] = maximum(ytrace2[i,:])
        yL2[i] = minimum(ytrace2[i,:])
    end
        
    
    
    
    pp4=plot(tt,xtrace2[:,:],color=:grey,xlabel="t",ylabel="x(t)",legend=false)
    pp4=plot!(tt,xU2,color=:red,lw=2,legend=false)
    pp4=plot!(tt,xL2,color=:red,lw=2,legend=false,xlims=(0,10),ylims=(0,2.5))
    pp5=plot(tt,ytrace2[:,:],color=:grey,xlabel="t",ylabel="y(t)",legend=false)
    pp5=plot!(tt,yU2,color=:red,lw=2,legend=false)
    pp5=plot!(tt,yL2,color=:red,lw=2,legend=false,xlims=(0,10),ylims=(0,2.5))
    pp6=plot(a2,pp4,pp5,layout=(1,3))
    display(pp6)
    savefig(pp6, "bivariateAx0.pdf")
    

        

function bivariateαy0(α,y0)
     function funαy0(a)
        return loglhood(datax,datay,[α,a[1],a[2],y0],σ)
        end
        θG = [βmle,x0mle]
        lb=[βmin,x0min]
        ub=[βmax,x0max]
        (xopt,fopt)  = optimise(funαy0,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt
    end 
    
    f(x,y) = bivariateαy0(x,y)
    g(x,y)=f(x,y)[1]-llstar
    
    
    
    ϵ=(y0max-y0min)/10^3
    αsamples=zeros(2*N)
    βsamples=zeros(2*N)
    x0samples=zeros(2*N)
    y0samples=zeros(2*N)
    count=0
    
    while count < N
    x=rand(Uniform(αmin,αmax))
    y0=rand(Uniform(y0min,y0max))
    y1=rand(Uniform(y0min,y0max))
    
    if g(x,y0)*g(x,y1) < 0 
    count+=1
    println(count)
    while abs(y1-y0) > ϵ && y1 < y0max && y1 > y0min
    y2=(y1+y0)/2; 
        if g(x,y0)*g(x,y2) < 0 
        y1=y2
        else
        y0=y2
        end
    
    
    end
    
    αsamples[count]=x;
    y0samples[count]=y1;
    βsamples[count]=f(x,y1)[2][1]
    x0samples[count]=f(x,y1)[2][2]
    end
    end 
    
    ϵ=(y0max-y0min)/10^3
    count=0
    while count < N
    y=rand(Uniform(y0min,y0max))
    x0=rand(Uniform(αmin,αmax))
    x1=rand(Uniform(αmin,αmax))
        
    if g(x0,y)*g(x1,y) < 0 
    count+=1
    println(count)
    
    while abs(x1-x0) > ϵ  && x1 < αmax && x1 > αmin
        x2=(x1+x0)/2;
            if g(x0,y)*g(x2,y) < 0 
            x1=x2
            else
            x0=x2
            end
        
        
        end
    
    
        αsamples[N+count]=x1;
        y0samples[N+count]=y;
        βsamples[N+count]=f(x1,y)[2][1]
        x0samples[N+count]=f(x1,y)[2][2]
        end
        end 
        

        a3=scatter([αmle],[y0mle],xlims=(αmin,αmax),ylims=(y0min,y0max),markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,xlabel="α",ylabel="y(0)",label=false)
        for i in 1:2*N
        a3=scatter!([αsamples[i]],[y0samples[i]],xlims=(αmin,αmax),ylims=(y0min,y0max),markersize=3,markershape=:circle,markercolor=:blue,msw=0, ms=5,label=false)
        end
        display(a3)
    
        xtrace3 = zeros(length(tt),2*N);
        ytrace3 = zeros(length(tt),2*N);
    
        xU3=zeros(length(tt))
        xL3=zeros(length(tt))
    
        yU3=zeros(length(tt))
        yL3=zeros(length(tt))
    
    
        for i in 1:2*N
        (xtrace3[:,i],ytrace3[:,i])=odesolver(tt,αsamples[i],βsamples[i],x0samples[i],y0samples[i]);
        end
        
        
        for i in 1:length(tt)
        xU3[i] = maximum(xtrace3[i,:])
        xL3[i] = minimum(xtrace3[i,:])
        yU3[i] = maximum(ytrace3[i,:])
        yL3[i] = minimum(ytrace3[i,:])
    end
        
    
    
    pp7=plot(tt,xtrace3[:,:],color=:grey,xlabel="t",ylabel="x(t)",legend=false)
    pp7=plot!(tt,xU3,color=:red,lw=2,legend=false)
    pp7=plot!(tt,xL3,color=:red,lw=2,legend=false,xlims=(0,10),ylims=(0,2.5))
    pp8=plot(tt,ytrace3[:,:],color=:grey,xlabel="t",ylabel="y(t)",legend=false)
    pp8=plot!(tt,yU3,color=:red,lw=2,legend=false)
    pp8=plot!(tt,yL3,color=:red,lw=2,legend=false,xlims=(0,10),ylims=(0,2.5))
    pp9=plot(a3,pp7,pp8,layout=(1,3))
    display(pp9)
    savefig(pp9, "bivariateAy0.pdf")
    

        

function bivariateβx0(β,x0)
     function funβx0(a)
        return loglhood(datax,datay,[a[1],β,x0,a[2]],σ)
        end
        θG = [αmle,y0mle]
        lb=[αmin,y0min]
        ub=[αmax,y0max]
        (xopt,fopt)  = optimise(funβx0,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt
    end 
    
    f(x,y) = bivariateβx0(x,y)
    g(x,y)=f(x,y)[1]-llstar
    ϵ=(x0max-x0min)/10^3
    αsamples=zeros(2*N)
    βsamples=zeros(2*N)
    x0samples=zeros(2*N)
    y0samples=zeros(2*N)
    count=0
    
    while count < N
    x=rand(Uniform(βmin,βmax))
    y0=rand(Uniform(x0min,x0max))
    y1=rand(Uniform(x0min,x0max))
    
    if g(x,y0)*g(x,y1) < 0 
    count+=1
    println(count)
    while abs(y1-y0) > ϵ && y1 < x0max && y1 > x0min
    y2=(y1+y0)/2;
        if g(x,y0)*g(x,y2) < 0 
        y1=y2
        else
        y0=y2
        end
    
    
    end
    
    βsamples[count]=x;
    x0samples[count]=y1;
    αsamples[count]=f(x,y1)[2][1]
    y0samples[count]=f(x,y1)[2][2]
    end
    end 
    
    ϵ=(βmax-βmin)/10^3
    count=0
    while count < N
    y=rand(Uniform(x0min,x0max))
    x0=rand(Uniform(βmin,βmax))
    x1=rand(Uniform(βmin,βmax))
        
    if g(x0,y)*g(x1,y) < 0 
    count+=1
    println(count)
    
    while abs(x1-x0) > ϵ && x1 < βmax && x1 > βmin
        x2=(x1+x0)/2;
            if g(x0,y)*g(x2,y) < 0 
            x1=x2
            else
            x0=x2
            end
        
        
        end
    
    
        βsamples[N+count]=x1;
        x0samples[N+count]=y;
        αsamples[N+count]=f(x1,y)[2][1]
        y0samples[N+count]=f(x1,y)[2][2]
        end
        end 
        
       


        
        a4=scatter([βmle],[x0mle],xlims=(βmin,βmax),ylims=(x0min,x0max),markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,xlabel="β",ylabel="x(0)",label=false)
        for i in 1:2*N
        a4=scatter!([βsamples[i]],[x0samples[i]],xlims=(βmin,βmax),ylims=(x0min,x0max),markersize=3,markershape=:circle,markercolor=:blue,msw=0, ms=5,label=false)
        end
        display(a4)
    
        xtrace4 = zeros(length(tt),2*N);
        ytrace4 = zeros(length(tt),2*N);
    
        xU4=zeros(length(tt))
        xL4=zeros(length(tt))
    
        yU4=zeros(length(tt))
        yL4=zeros(length(tt))
    
    
        for i in 1:2*N
        (xtrace4[:,i],ytrace4[:,i])=odesolver(tt,αsamples[i],βsamples[i],x0samples[i],y0samples[i]);
        end
        
        
        for i in 1:length(tt)
        xU4[i] = maximum(xtrace4[i,:])
        xL4[i] = minimum(xtrace4[i,:])
        yU4[i] = maximum(ytrace4[i,:])
        yL4[i] = minimum(ytrace4[i,:])
    end
        
    
    
    pp10=plot(tt,xtrace4[:,:],color=:grey,xlabel="t",ylabel="x(t)",legend=false)
    pp10=plot!(tt,xU4,color=:red,lw=2,legend=false)
    pp10=plot!(tt,xL4,color=:red,lw=2,legend=false,xlims=(0,10),ylims=(0,2.5))
    pp11=plot(tt,ytrace4[:,:],color=:grey,xlabel="t",ylabel="y(t)",legend=false)
    pp11=plot!(tt,yU4,color=:red,lw=2,legend=false)
    pp11=plot!(tt,yL4,color=:red,lw=2,legend=false,xlims=(0,10),ylims=(0,2.5))
    pp12=plot(a4,pp10,pp11,layout=(1,3))
    display(pp12)
    savefig(pp12, "bivariateBx0.pdf")
    





function bivariateβy0(β,y0)
     function funβy0(a)
        return loglhood(datax,datay,[a[1],β,a[2],y0],σ)
        end
        θG = [αmle,x0mle]
        lb=[αmin,x0min]
        ub=[αmax,x0max]
        (xopt,fopt)  = optimise(funβy0,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt
    end 
    
    f(x,y) = bivariateβy0(x,y)
    g(x,y)=f(x,y)[1]-llstar
    
    ϵ=(y0max-y0min)/10^3
    αsamples=zeros(2*N)
    βsamples=zeros(2*N)
    x0samples=zeros(2*N)
    y0samples=zeros(2*N)
    count=0
    
    while count < N
    x=rand(Uniform(βmin,βmax))
    y0=rand(Uniform(y0min,y0max))
    y1=rand(Uniform(y0min,y0max))
    
    if g(x,y0)*g(x,y1) < 0 
    count+=1
    println(count)
    while abs(y1-y0) > ϵ  && y1 < y0max && y1 > y0min
    y2=(y1+y0)/2;
        if g(x,y0)*g(x,y2) < 0 
        y1=y2
        else
        y0=y2
        end
    
    
    end
    
    βsamples[count]=x;
    y0samples[count]=y1;
    αsamples[count]=f(x,y1)[2][1]
    x0samples[count]=f(x,y1)[2][2]
    end
    end 
    
    ϵ=(βmax-βmin)/10^3
    count=0
    while count < N
    y=rand(Uniform(y0min,y0max))
    x0=rand(Uniform(βmin,βmax))
    x1=rand(Uniform(βmin,βmax))
        
    if g(x0,y)*g(x1,y) < 0 
    count+=1
    println(count)
    
    while abs(x1-x0) > ϵ && x1 < βmax && x1 > βmin
        x2=(x1+x0)/2;
            if g(x0,y)*g(x2,y) < 0 
            x1=x2
            else
            x0=x2
            end
        
        
        end
    
    
        βsamples[N+count]=x1;
        y0samples[N+count]=y;
        αsamples[N+count]=f(x1,y)[2][1]
        x0samples[N+count]=f(x1,y)[2][2]
        end
        end 
        
       
        
        a5=scatter([βmle],[y0mle],xlims=(βmin,βmax),ylims=(y0min,y0max),markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,xlabel="β",ylabel="y(0)",label=false)
        for i in 1:2*N
        a5=scatter!([βsamples[i]],[y0samples[i]],xlims=(βmin,βmax),ylims=(y0min,y0max),markersize=3,markershape=:circle,markercolor=:blue,msw=0, ms=5,label=false)
        end
        display(a5)
    
        xtrace5 = zeros(length(tt),2*N);
        ytrace5 = zeros(length(tt),2*N);
    
        xU5=zeros(length(tt))
        xL5=zeros(length(tt))
    
        yU5=zeros(length(tt))
        yL5=zeros(length(tt))
    
    
        for i in 1:2*N
        (xtrace5[:,i],ytrace5[:,i])=odesolver(tt,αsamples[i],βsamples[i],x0samples[i],y0samples[i]);
        end
        
        
        for i in 1:length(tt)
        xU5[i] = maximum(xtrace5[i,:])
        xL5[i] = minimum(xtrace5[i,:])
        yU5[i] = maximum(ytrace5[i,:])
        yL5[i] = minimum(ytrace5[i,:])
    end
        
    
    


    pp13=plot(tt,xtrace5[:,:],color=:grey,xlabel="t",ylabel="x(t)",legend=false)
    pp13=plot!(tt,xU5,color=:red,lw=2,legend=false)
    pp13=plot!(tt,xL5,color=:red,lw=2,legend=false,xlims=(0,10),ylims=(0,2.5))
    pp14=plot(tt,ytrace5[:,:],color=:grey,xlabel="t",ylabel="y(t)",legend=false)
    pp14=plot!(tt,yU5,color=:red,lw=2,legend=false)
    pp14=plot!(tt,yL5,color=:red,lw=2,legend=false,xlims=(0,10),ylims=(0,2.5))
    pp15=plot(a5,pp13,pp14,layout=(1,3))
    display(pp15)
    savefig(pp15, "bivariateBy0.pdf")


        

function bivariatex0y0(x0,y0)
     function funx0y0(a)
        return loglhood(datax,datay,[a[1],a[2],x0,y0],σ)
        end
        θG = [αmle,βmle]
        lb=[αmin,βmin]
        ub=[αmax,βmax]
        (xopt,fopt)  = optimise(funx0y0,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt
    end 
    
    f(x,y) = bivariatex0y0(x,y)
    g(x,y)=f(x,y)[1]-llstar
    ϵ=(y0max-y0min)/10^3
    αsamples=zeros(2*N)
    βsamples=zeros(2*N)
    x0samples=zeros(2*N)
    y0samples=zeros(2*N)
    count=0
    
    while count < N
    x=rand(Uniform(x0min,x0max))
    y0=rand(Uniform(y0min,y0max))
    y1=rand(Uniform(y0min,y0max))
    
    if g(x,y0)*g(x,y1) < 0 
    count+=1
    println(count)
    while abs(y1-y0) > ϵ  && y1 < y0max && y1 > y0min
    y2=(y1+y0)/2;
        if g(x,y0)*g(x,y2) < 0 
        y1=y2
        else
        y0=y2
        end
    
    
    end
    
    x0samples[count]=x;
    y0samples[count]=y1;
    αsamples[count]=f(x,y1)[2][1]
    βsamples[count]=f(x,y1)[2][2]
    end
    end 
    
    ϵ=(x0max-x0min)/10^3
    count=0
    while count < N
    y=rand(Uniform(y0min,y0max))
    x0=rand(Uniform(x0min,x0max))
    x1=rand(Uniform(x0min,x0max))
        
    if g(x0,y)*g(x1,y) < 0 
    count+=1
    println(count)
    
    while abs(x1-x0) > ϵ && x1 < x0max && x1 > x0min
        x2=(x1+x0)/2;
            if g(x0,y)*g(x2,y) < 0 
            x1=x2
            else
            x0=x2
            end
        
        
        end
    
    
        x0samples[N+count]=x1;
        y0samples[N+count]=y;
        αsamples[N+count]=f(x1,y)[2][1]
        βsamples[N+count]=f(x1,y)[2][2]
        end
        end 
        
        
        a6=scatter([x0mle],[y0mle],xlims=(x0min,x0max),ylims=(y0min,y0max),markersize=3,markershape=:circle,markercolor=:fuchsia,msw=0, ms=5,xlabel="x(0)",ylabel="y(0)",label=false)
        for i in 1:2*N
        a6=scatter!([x0samples[i]],[y0samples[i]],xlims=(x0min,x0max),ylims=(y0min,y0max),markersize=3,markershape=:circle,markercolor=:blue,msw=0, ms=5,label=false)
        end
        display(a6)
    
        xtrace6 = zeros(length(tt),2*N);
        ytrace6 = zeros(length(tt),2*N);
    
        xU6=zeros(length(tt))
        xL6=zeros(length(tt))
    
        yU6=zeros(length(tt))
        yL6=zeros(length(tt))
    
    
        for i in 1:2*N
        (xtrace6[:,i],ytrace6[:,i])=odesolver(tt,αsamples[i],βsamples[i],x0samples[i],y0samples[i]);
        end
        
        
        for i in 1:length(tt)
        xU6[i] = maximum(xtrace6[i,:])
        xL6[i] = minimum(xtrace6[i,:])
        yU6[i] = maximum(ytrace6[i,:])
        yL6[i] = minimum(ytrace6[i,:])
    end
        
    
    


    pp16=plot(tt,xtrace6[:,:],color=:grey,xlabel="t",ylabel="x(t)",legend=false)
    pp16=plot!(tt,xU6,color=:red,lw=2,legend=false)
    pp16=plot!(tt,xL6,color=:red,lw=2,legend=false,xlims=(0,10),ylims=(0,2.5))
    pp17=plot(tt,ytrace6[:,:],color=:grey,xlabel="t",ylabel="y(t)",legend=false)
    pp17=plot!(tt,yU6,color=:red,lw=2,legend=false)
    pp17=plot!(tt,yL6,color=:red,lw=2,legend=false,xlims=(0,10),ylims=(0,2.5))
    pp18=plot(a6,pp16,pp17,layout=(1,3))
    display(pp18)
    savefig(pp18, "bivariatex0y0.pdf")




XU=zeros(length(tt))
XL=zeros(length(tt))
YU=zeros(length(tt))
YL=zeros(length(tt))


for i in 1:length(tt)
XU[i]=max(xU1[i],xU2[i],xU3[i],xU4[i],xU5[i],xU6[i])
YU[i]=max(yU1[i],yU2[i],yU3[i],yU4[i],yU5[i],yU6[i])
XL[i]=max(xL1[i],xL2[i],xL3[i],xL4[i],xL5[i],xL6[i])
YL[i]=min(yL1[i],yL2[i],yL3[i],yL4[i],yL5[i],yL6[i])
end



pp19=plot(tt,XL,color=:red,lw=2,legend=false)
pp19=plot!(tt,XU,color=:red,lw=2,xlabel="t",ylabel="x(t)",legend=false,xlims=(0,10),ylims=(0,2.5))
pp20=plot(tt,YL,color=:red,lw=2,legend=false)
pp20=plot!(tt,YU,color=:red,lw=2,xlabel="t",ylabel="y(t)",legend=false,xlims=(0,10),ylims=(0,2.5))
pp21=plot(pp19,pp20,layout=(1,2))
display(pp21)
savefig(pp21, "Unionprofiles.pdf")




#Let's compute and push forward from the full 4D likelihood function

        
df=4
llstar=-quantile(Chisq(df),0.95)/2

N=10000
αs=rand(Uniform(αmin,αmax),N);
βs=rand(Uniform(βmin,βmax),N);
x0s=rand(Uniform(x0min,x0max),N);
y0s=rand(Uniform(y0min,y0max),N);

lls=zeros(N)
for i in 1:N
lls[i]=loglhood(datax,datay,[αs[i],βs[i],x0s[i],y0s[i]],σ)-fmle
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

αsampled=zeros(M)
βsampled=zeros(M)
x0sampled=zeros(M)
y0sampled=zeros(M)
xtraceF = zeros(length(tt),M)
ytraceF = zeros(length(tt),M)
xUF=zeros(length(tt))
xLF=zeros(length(tt))
yUF=zeros(length(tt))
yLF=zeros(length(tt))

j=0
for i in 1:N
    if lls[i] > llstar
    global j = j + 1
    αsampled[j]=αs[i]
    βsampled[j]=βs[i]
    x0sampled[j]=x0s[i]
    y0sampled[j]=y0s[i]
    (xtraceF[:,j],ytraceF[:,j])=odesolver(tt,αs[i],βs[i],x0s[i],y0s[i]);
    end
end


for i in 1:length(tt)
xUF[i] = maximum(xtraceF[i,:])
xLF[i] = minimum(xtraceF[i,:])
yUF[i] = maximum(ytraceF[i,:])
yLF[i] = minimum(ytraceF[i,:])
end





qq1=plot(tt,xtraceF[:,:],color=:grey,xlabel="t",ylabel="x(t)",legend=false)
qq1=plot!(tt,xxmle,lw=3,color=:turquoise1,legend=false)
qq1=plot!(tt,xUF,lw=3,color=:gold,legend=false)
qq1=plot!(tt,xLF,lw=3,color=:gold,xlabel="t",ylabel="x(t)",legend=false)
qq1=plot!(tt,XL,color=:red,lw=2,legend=false)
qq1=plot!(tt,XU,color=:red,lw=2,xlabel="t",ylabel="x(t)",legend=false,xlims=(0,10),ylims=(0,2.5))



qq2=plot(tt,ytraceF[:,:],color=:grey,xlabel="t",ylabel="y(t)",legend=false)
qq2=plot!(tt,yymle,lw=3,color=:turquoise1,legend=false)
qq2=plot!(tt,yUF,lw=3,color=:gold,legend=false)
qq2=plot!(tt,yLF,lw=3,color=:gold,xlabel="t",ylabel="y(t)",legend=false)
qq2=plot!(tt,YL,color=:red,lw=2,legend=false)
qq2=plot!(tt,YU,color=:red,lw=2,xlabel="t",ylabel="y(t)",legend=false,xlims=(0,10),ylims=(0,2.5))



ww1=plot(qq1,qq2,layout=(1,2),legend=false)
display(ww1)
savefig(ww1, "Comparison.pdf")