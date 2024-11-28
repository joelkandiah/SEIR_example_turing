# Imports 
using OrdinaryDiffEq
using Distributions
using DataInterpolations
using Turing
using LinearAlgebra
using StatsPlots
using Random
using Bijectors
using TOML
using SparseArrays
using Plots.PlotMeasures

# Read parameters from command line for the varied submission scripts
seed_idx = 1
Δ_βt = 90
Window_size = 180
n_chains = 24
n_samples_per_chain = 500
tmax = 400


# List the seeds to generate a set of data scenarios (for reproducibility)
seeds_list = [1234, 1357, 2358, 3581]

# Set seed
Random.seed!(seeds_list[seed_idx])

n_threads = Threads.nthreads()

# Set and Create locations to save the plots and Chains
outdir = string("Results/seed $seed_idx/")
tmpstore = string("Chains/seed $seed_idx/")

if !isdir(outdir)
    mkpath(outdir)
end
if !isdir(tmpstore)
    mkpath(tmpstore)
end

# Initialise the model parameters (fixed)
tspan = (0.0, tmax)
obstimes = 1.0:1.0:tmax
NA = 1
N = 1_000_000
NA_N = [N]
I0 = [100]
u0 = zeros(NA,5)
u0[:,1] = NA_N - I0
u0[:,3] = I0
d_I= 10  
d_L = 3
#initialise infectious and latent periods
γ = 1/ d_I
σ = 1/ d_L
p = [γ, σ, N];

I0_μ_prior_orig = log.(I0 ./ N)
I0_μ_prior = -9.0

# Set parameters for inference and draw betas from prior
β₀σ = 0.15
β₀μ = 0.14
βσ = 0.15
true_beta = repeat([NaN], Integer(ceil(tmax/ Δ_βt)) + 1)
true_beta[1] = exp(rand(Normal(log(β₀μ), β₀σ)))
for i in 2:(length(true_beta) - 1)
    true_beta[i] = exp(log(true_beta[i-1]) + rand(Normal(0.0,βσ)))
end 
true_beta[length(true_beta)] = true_beta[length(true_beta)-1]
knots = collect(0.0:Δ_βt:tmax)
knots = knots[end] != tmax ? vcat(knots, tmax) : knots
K = length(knots)


# Construct an ODE for the SEIR model
function sir_tvp_ode!(du::Array{T1}, u::Array{T2}, p_, t) where {T1 <: Real, T2 <: Real}
    @inbounds begin
        S = @view u[1,:]
        E = @view u[2,:]
        I = @view u[3,:]
        # R = @view u[:,4]
        # I_tot = @view u[:,5]
    end
    (γ, σ, N) = p_.params_floats
    βt = p_.β_function(t)
    It = sum(I)
    for a in axes(du,2)
        local infection = (1.0 - exp(It * log(1.0 - βt / N))) * S[a]
        local infectious = σ * E[a]
        local recovery = γ * I[a]
        @inbounds begin
            du[1,a] = - infection
            du[2,a] = infection - infectious
            du[3,a] = infectious - recovery
            du[4,a] = infection
            du[5,a] = infectious
        end
    end
end;# Construct an ODE for the SEIR model


struct idd_params{T <: Real, T2 <: DataInterpolations.AbstractInterpolation}
    params_floats::Vector{T}
    β_function::T2
    N_regions::Int
end

NR = 1

params_test = idd_params(p, ConstantInterpolation(true_beta, knots), NR)

# Initialise the specific values for the ODE system and solve
prob_ode = ODEProblem(sir_tvp_ode!, u0', tspan, params_test)
#? Note the choice of tstops and d_discontinuities to note the changepoints in β
#? Also note the choice of solver to resolve issues with the "stiffness" of the ODE system
sol_ode = solve(prob_ode,
            Tsit5(; thread = OrdinaryDiffEq.False()),
            maxiters = 1e6,
            abstol = 1e-8,
            reltol = 1e-5,
            # callback = cb,
            saveat = 1.0,
            tstops = knots[2:end-1],
            d_discontinuities = knots);

# Optionally plot the SEIR system
StatsPlots.plot(stack(map(x -> x[3,:], sol_ode.u))',
    xlabel="Time",
    ylabel="Number",
    linewidth = 1)


# Find the cumulative number of cases
I_tot_2 = Array(sol_ode(obstimes))[5,:,:]

# Define utility function for the difference between consecutive arguments in a list f: Array{N} x Array{N} -> Array{N-1}
function rowadjdiff(ary)
    ary1 = copy(ary)
    ary1[:, begin + 1:end] =  (@view ary[:, begin+1:end]) - (@view ary[:,begin:end-1])
    return ary1
end

function adjdiff(ary)
    ary1 = copy(ary)
    ary1[ begin + 1:end] =  (@view ary[begin+1:end]) - (@view ary[begin:end-1])
    return ary1
end


# Number of new infections
X = rowadjdiff(I_tot_2)

# Define Gamma distribution by mean and standard deviation
function Gamma_mean_sd_dist(μ, σ)
    α = @. (μ * μ) / (σ * σ)
    θ = @. (σ * σ) / μ
    return Gamma.(α, θ)
end

# Define helpful distributions (arbitrary choice from sample in RTM)
incubation_dist = Gamma_mean_sd_dist(4.0, 1.41)
symp_to_hosp = Gamma_mean_sd_dist(9.0, 8.0666667)

# Define approximate convolution of Gamma distributions
# f: Distributions.Gamma x Distributions.Gamma -> Distributions.Gamma
function approx_convolve_gamma(d1::Gamma, d2::Gamma)
    μ_new = (d1.α * d1.θ) + (d2.α * d2.θ)

    var1 = d1.α * d1.θ * d1.θ
    var2 = d2.α * d2.θ * d2.θ

    σ_new = sqrt(var1 + var2)

    return Gamma_mean_sd_dist(μ_new, σ_new)
end

# Define observation distributions (new infections to reported hospitalisations)
inf_to_hosp = approx_convolve_gamma(incubation_dist,symp_to_hosp)
inf_to_hosp_array_cdf = cdf(inf_to_hosp,1:80)
inf_to_hosp_array_cdf = adjdiff(inf_to_hosp_array_cdf)

# Create function to create a matrix to calculate the discrete convolution (multiply convolution matrix by new infections vector to get mean of number of (eligible) hospitalisations per day)
function construct_pmatrix(
    v = inf_to_hosp_array_cdf,
    l = Integer(tmax))
    rev_v = @view v[end:-1:begin]
    len_v = length(rev_v)
    ret_mat = zeros(l, l)
    for i in axes(ret_mat, 1)
        ret_mat[i, max(1, i + 1 - len_v):min(i, l)] .= @view rev_v[max(1, len_v-i+1):end]        
    end
    return sparse(ret_mat)
end

# Evaluate mean number of hospitalisations (using proportion of 0.3)
conv_mat = construct_pmatrix(;)  
Y_mu = mapreduce(x -> 0.3 * conv_mat * x, hcat, eachrow(X))'

# Create function to construct Negative binomial with properties matching those in Birell et. al (2021)
function NegativeBinomial3(μ, ϕ)
    p = 1 / (1 + ϕ)
    r = μ / ϕ
    return NegativeBinomial(r, p)
end

# Draw sample of hospitalisations
Y = @. rand(NegativeBinomial3(Y_mu + 1e-3, 10));

# Plot mean hospitalisations over hospitalisations
StatsPlots.bar(obstimes, Y', legend=true, alpha = 0.3)
StatsPlots.plot!(obstimes, eachrow(Y_mu))


# Define the model taking in the data and the times the beta values changepoints
# Add named args for fixed ODE parameters and for the convolution matrix, the times the beta values change and the specific times for evaluating the ODE
@model function bayes_sir_tvp(
    # y,
    K,
    γ = γ,
    σ = σ,
    N = N,
    NA = NA,
    NA_N = NA_N,
    N_regions = NR,
    conv_mat = conv_mat,
    knots = knots,
    obstimes = obstimes,
    I0_μ_prior = I0_μ_prior,
    β₀μ = β₀μ,
    β₀σ = β₀σ,
    βσ = βσ,
    ::Type{T} = Float64,
    ::Type{T2} = Float64,
    ::Type{T3} = Float64;
) where {T <: Real, T2 <: Real, T3 <: Real}

    # Set prior for initial infected
    log_I₀  ~ truncated(Normal(I0_μ_prior, 0.2); lower = log(1.0 / N), upper = 0.0)
    I = exp(log_I₀) * N
    
    I_list = zero(Vector{T2}(undef, NA))
    I_list[1] = I
    u0 = zero(Matrix{T3}(undef, 5, NA))
    u0[1,:] = NA_N - I_list
    u0[3,:] = I_list
    

    # Set priors for betas
    ## Note how we clone the endpoint of βt
    β = Vector{T}(undef, K)
    log_β = Vector{T}(undef, K-2)
    p = [γ, σ, N]
    log_β₀ ~ Normal(β₀μ, β₀σ)
    β[1] = exp(log_β₀)
    for i in 2:K-1
        log_β[i-1] ~ Normal(0.0, βσ)
        β[i] = exp(log(β[i-1]) + log_β[i-1])
    end
    β[K] = β[K-1]

    if(I < 1)
        @DynamicPPL.addlogprob! -Inf
        return
    end

    if(any(β .> N) | any(isnan.(β)))
        @DynamicPPL.addlogprob! -Inf
        return
    end

    params_test = idd_params(p, ConstantInterpolation(β, knots), 1) 
    # Run model
    ## Remake with new initial conditions and parameter values
    tspan = (0, maximum(obstimes))
    prob = ODEProblem{true}(sir_tvp_ode!, u0, tspan, params_test)
    
    ## Solve
    sol = 
    # try 
        solve(prob,
            Tsit5(),
            saveat = obstimes,
            d_discontinuities = knots[2:end-1],
            tstops = knots[2:end-1],
            )
    # catch e
    #     if e isa InexactError
    #         # if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
    #             @DynamicPPL.addlogprob! -Inf
    #             return
    #         # end
    #     else
    #         rethrow(e)
    #     end
    # end

    if any(sol.retcode != :Success)
            @DynamicPPL.addlogprob! -Inf
            return
    end
    
    ## Calculate new infections per day, X
    sol_I_tot = Array(sol(obstimes))[5,:,:]
    sol_X = rowadjdiff(sol_I_tot)
    if (any(sol_X .< -(1e-3)) | any(Array(sol(obstimes))[3,:,:] .< -1e-3))
        @DynamicPPL.addlogprob! -Inf
        return
    end
    check = minimum(sol_X)
    y_μ = (conv_mat * (0.3 .* sol_X)') |> transpose

    # Assume Poisson distributed counts
    ## Calculate number of timepoints
    if (any(isnan.(y_μ)))
        @DynamicPPL.addlogprob! -Inf
        return
    end
    y ~ product_distribution(NegativeBinomial3.(y_μ .+ 1e-3, 10))

    return (; sol, p, check)
end;

# Define the parameters for the model given the known window size (i.e. vectors that fit within the window)
knots_window = collect(0:Δ_βt:Window_size)
knots_window = knots_window[end] != Window_size ? vcat(knots_window, Window_size) : knots_window
K_window = length(knots_window)
conv_mat_window = construct_pmatrix(inf_to_hosp_array_cdf, Window_size)
obstimes_window = 1.0:1.0:Window_size

# Define the model for the windowed data
model_window_unconditioned = bayes_sir_tvp(
    K_window,
    γ,
    σ,
    N,
    NA,
    NA_N,
    NR,
    conv_mat_window,
    knots_window,
    obstimes_window,
    I0_μ_prior,
    β₀μ,
    β₀σ,
    βσ;
    )

# Perform the chosen inference algorithm
t1_init = time_ns()
ode_nuts = sample(model_window_unconditioned| (y = Y[:,1:Window_size],), Turing.NUTS(1500, 0.65;), MCMCThreads(), n_samples_per_chain, n_chains, discard_initial = 0, thinning = 1);
t2_init = time_ns()
runtime_init = convert(Int64, t2_init-t1_init)

logjoint(model_window_unconditioned | (y = Y[:,1:Window_size],) ,ode_nuts)


# Create a function to take in the chains and evaluate the number of infections and summarise them (at a specific confidence level)
function generate_confint_infec_init(chn, y_data, K, conv_mat, knots, obstimes; cri = 0.95)
    chnm_res = generated_quantities(
        bayes_sir_tvp(K;
            conv_mat = conv_mat,
            knots = knots,
            obstimes = obstimes
            )| (y = y_data,),
        chn) 


    infecs = stack(map(x -> Array(x.sol)[3,:,:], chnm_res[1,:]))
    lowci_inf = mapslices(x -> quantile(x,(1-cri) / 2), infecs, dims = 3)[:,:,1]
    medci_inf = mapslices(x -> quantile(x, 0.5), infecs, dims = 3)[:, :, 1]
    uppci_inf = mapslices(x -> quantile(x, cri + (1-cri) / 2), infecs, dims = 3)[:, :, 1]
    return (; lowci_inf, medci_inf, uppci_inf)
end

# Create a function to take in the chains and evaluate the number of recovereds and summarise them (at a specific confidence level)
function generate_confint_recov_init(chn, y_data, K, conv_mat, knots, obstimes; cri = 0.95)
    chnm_res = generated_quantities(
        bayes_sir_tvp(K;
            conv_mat = conv_mat,
            knots = knots,
            obstimes = obstimes
            )| (y = y_data, ),
        chn) 

    infecs = stack(map(x -> Array(x.sol)[4,:,:], chnm_res[1,:]))
    lowci_inf = mapslices(x -> quantile(x,(1-cri) / 2), infecs, dims = 3)[:,:,1]
    medci_inf = mapslices(x -> quantile(x, 0.5), infecs, dims = 3)[:, :, 1]
    uppci_inf = mapslices(x -> quantile(x, cri + (1-cri) / 2), infecs, dims = 3)[:, :, 1]
    return (; lowci_inf, medci_inf, uppci_inf)
end

I_dat = Array(sol_ode(obstimes))[3,:,:] # Population of infecteds at times
R_dat = Array(sol_ode(obstimes))[4,:,:] # Population of recovereds at times

get_beta_quantiles = function(chn, K; ci = 0.95)
    # Get the beta values and calculate the estimated confidence interval and median
        betas = Array(chn)
        beta_idx = [collect(2:K); K]
    
        betas[:,2:end] =exp.(cumsum(betas[:,2:end], dims = 2))
        beta_μ = [quantile(betas[:,i], 0.5) for i in beta_idx]
        betas_lci = [quantile(betas[:,i], (1 - ci) / 2) for i in beta_idx]
        betas_uci = [quantile(betas[:,i], 1 - ((1-ci) / 2)) for i in beta_idx]
        return (beta_μ, betas_lci, betas_uci)
end

beta_μ, betas_lci, betas_uci = get_beta_quantiles(ode_nuts, K_window)

betat_no_win = ConstantInterpolation(true_beta, knots)

StatsPlots.plot(obstimes[1:Window_size],
    ConstantInterpolation(beta_μ, knots_window)(obstimes[1:Window_size]),
    ribbon = (ConstantInterpolation(beta_μ, knots_window)(obstimes[1:Window_size]) - ConstantInterpolation(betas_lci, knots_window)(obstimes[1:Window_size]), ConstantInterpolation(betas_uci, knots_window)(obstimes[1:Window_size]) - ConstantInterpolation(beta_μ, knots_window)(obstimes[1:Window_size])),
    xlabel = "Time",
    ylabel = "β",
    label="Using the NUTS algorithm",
    title="\nEstimates of β",
    color=:blue,
    lw = 2,
    titlefontsize=18,
    guidefontsize=18,
    tickfontsize=16,
    legendfontsize=12,
    fillalpha = 0.4,
    legendposition = :outerbottom,
    margin = 10mm,
    bottom_margin = 0mm)
StatsPlots.plot!(obstimes[1:Window_size],
    betat_no_win(obstimes[1:Window_size]),
    color=:red,
    label="True β",
    lw = 2)
StatsPlots.plot!(size = (1200,800))

savefig(string(outdir,"nuts_betas_window_1_$seed_idx.png"))


# Plot the infecteds
confint = generate_confint_infec_init(ode_nuts, Y[:,1:Window_size], K_window, conv_mat_window, knots_window, obstimes_window; cri = 0.9)
StatsPlots.plot(confint.medci_inf', ribbon = (confint.medci_inf' - confint.lowci_inf', confint.uppci_inf' - confint.medci_inf') , legend = false)
StatsPlots.plot!(I_dat[:,1:Window_size]', linewidth = 2, color = :red)
StatsPlots.plot!(size = (1200,800))

savefig(string(outdir,"infections_nuts_window_1_$seed_idx.png"))

# Plot the recovereds
confint = generate_confint_recov_init(ode_nuts, Y[:,1:Window_size], K_window, conv_mat_window, knots_window, obstimes_window; cri = 0.9)
StatsPlots.plot(confint.medci_inf', ribbon = (confint.medci_inf' - confint.lowci_inf', confint.uppci_inf' - confint.medci_inf')  , legend = false)
StatsPlots.plot!(R_dat[:,1:Window_size]', linewidth = 2, color = :red)
StatsPlots.plot!(size = (1200,800))

savefig(string(outdir,"recoveries_nuts_window_1_$seed_idx.png"))