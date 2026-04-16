using ITensors
using ITensorMPS
using HDF5
using Printf
#using Plots

function ene_measurement(G, L, Hterm)
    AA = L[2] * G[1]
    BB = G[2] * L[2]
    ipsi = AA * L[1] * BB
    ene  = dag(ipsi)*noprime!(Hterm[1]*ipsi)

    AA = L[1] * G[2]
    BB = G[1] * L[1]
    ipsi = AA * L[2] * BB
    ene += dag(ipsi)*noprime!(Hterm[2]*ipsi)

    return real(ene[1])
end

function MakeVidalMPS(psi, maxdim)
    Gm=[]; Ld=[]

    U, Sv, V = svd(psi[1] * psi[2], inds(psi[1]); maxdim=maxdim, cutoff=eps(Float64))
    Sv /= norm(Sv)
    push!(Gm, U)
    push!(Ld, Sv)
    push!(Gm, noprime(V))

    Gm[2], Sv, Gm[1] = svd(Gm[2] * Gm[1], inds(Gm[2]); maxdim=maxdim, cutoff=eps(Float64))
    Sv /= norm(Sv)
    push!(Ld, Sv)

    #@show Gm
    #@show Ld

    return Gm, Ld
end

function Gate_2sites(G1, G2, L1, L2, Gate, truncErr, maxdim, normalize::Bool)
    a = uniqueinds(G1, L1; tags="Link,v")
    b = uniqueinds(G2, L1; tags="Link,u")
    
    inv = ITensor(a,b)
    for dd=1:dim(a)
        inv[dd,dd] = 1.0 / L2[dd,dd]
    end
    #@show inv

    AA = L2 * G1
    BB = G2 * L2
    #@show AA
    #@show BB
    Gpsi = Gate * AA * L1 * BB
    Gpsi= noprime!(Gpsi)
    AA, L1, BB = svd(Gpsi, inds(AA);maxdim=maxdim,cutoff=eps(Float64))
    #@show AA
    #@show L1
    #@show BB

    if normalize==true
        truncErr_tmp = 1.0-sqrt(norm(L1)/norm(Gpsi))
        if truncErr_tmp > truncErr 
            truncErr = truncErr_tmp
        end
        L1 /= norm(L1)
    end
    G1 = inv * AA
    G2 = BB * inv

    return G1, G2, L1, truncErr
end

function ST_decomp_4th_im(Gm, Ld, H, step, truncErr, maxdim)   
    s = 1.3512071919596576

    Gate = exp(-0.5 * s * step * H[1])
    Gm[1], Gm[2], Ld[1], truncErr = Gate_2sites(Gm[1], Gm[2], Ld[1], Ld[2], Gate, truncErr, maxdim, true)

    Gate = exp(-s * step * H[2])
    Gm[2], Gm[1], Ld[2], truncErr = Gate_2sites(Gm[2], Gm[1], Ld[2], Ld[1], Gate, truncErr, maxdim, true)

    Gate = exp(0.5 * (s - 1.0) * step * H[1])
    Gm[1], Gm[2], Ld[1], truncErr = Gate_2sites(Gm[1], Gm[2], Ld[1], Ld[2], Gate, truncErr, maxdim, true)

    Gate = exp( (2.0 * s - 1.0) * step * H[2])
    Gm[2], Gm[1], Ld[2], truncErr = Gate_2sites(Gm[2], Gm[1], Ld[2], Ld[1], Gate, truncErr, maxdim, true)

    Gate = exp(0.5 * (s-1.0) * step * H[1])
    Gm[1], Gm[2], Ld[1], truncErr = Gate_2sites(Gm[1], Gm[2], Ld[1], Ld[2], Gate, truncErr, maxdim, true)

    Gate = exp(-s * step * H[2])
    Gm[2], Gm[1], Ld[2], truncErr = Gate_2sites(Gm[2], Gm[1], Ld[2], Ld[1], Gate, truncErr, maxdim, true)

    Gate = exp(-0.5 * s * step * H[1])
    Gm[1], Gm[2], Ld[1], truncErr = Gate_2sites(Gm[1], Gm[2], Ld[1], Ld[2], Gate, truncErr, maxdim, true)
    
    return Gm, Ld, truncErr
end

function ST_decomp_4th_real(Gm, Ld, H, V, time, step, truncErr, maxdim)   
    s = 1.3512071919596576

    E = ext_field(time + step * (s - 4) / 4)
    Gate = exp(-0.5 * s * im * step * (H[1] + E * V[1]))
    Gm[1], Gm[2], Ld[1], truncErr = Gate_2sites(Gm[1], Gm[2], Ld[1], Ld[2], Gate, truncErr, maxdim, true)

    E = ext_field(time + step * (s - 2) / 2)
    Gate = exp(-s * im * step * (H[2] + E * V[2]))
    Gm[2], Gm[1], Ld[2], truncErr = Gate_2sites(Gm[2], Gm[1], Ld[2], Ld[1], Gate, truncErr, maxdim, true)

    E = ext_field(time + step * (s - 3) / 4)
    Gate = exp(0.5 * (s - 1.0) * im * step * (H[1] + E * V[1]))
    Gm[1], Gm[2], Ld[1], truncErr = Gate_2sites(Gm[1], Gm[2], Ld[1], Ld[2], Gate, truncErr, maxdim, true)

    E = ext_field(time - step / 2 )
    Gate = exp( (2.0 * s - 1.0) * im * step * (H[2] + E * V[2]))
    Gm[2], Gm[1], Ld[2], truncErr = Gate_2sites(Gm[2], Gm[1], Ld[2], Ld[1], Gate, truncErr, maxdim, true)

    E = ext_field(time + step * (s - 1) * 4)
    Gate = exp(0.5 * (s-1.0) * im * step * (H[1] + E * V[1]))
    Gm[1], Gm[2], Ld[1], truncErr = Gate_2sites(Gm[1], Gm[2], Ld[1], Ld[2], Gate, truncErr, maxdim, true)

    E = ext_field(time - step * s / 2)
    Gate = exp(-s * im * step * (H[2] + E * V[2]))
    Gm[2], Gm[1], Ld[2], truncErr = Gate_2sites(Gm[2], Gm[1], Ld[2], Ld[1], Gate, truncErr, maxdim, true)

    E = ext_field(time - step * s / 4)
    Gate = exp(-0.5 * s * im * step * (H[1] + E * V[1]))
    Gm[1], Gm[2], Ld[1], truncErr = Gate_2sites(Gm[1], Gm[2], Ld[1], Ld[2], Gate, truncErr, maxdim, true)
    
    return Gm, Ld, truncErr
end

function itebd_gs_2site(sites, beta_step_init, maxdim)
    ### setting imarginary time step ###

    beta_step_min = 1e-14
    #@show beta_step_init
    #@show beta_step_min

    ####################################

    states = [isodd(n) ? "Up" : "Dn" for n in 1:2]
    psi = MPS(sites, states)
    #@show psi

    ###########################

    Gamma, Lambda = MakeVidalMPS(psi, maxdim)
    #@show Gamma[1]
    #@show Gamma[2]
    #@show Lambda[1]
    #@show Lambda[2]
        
    Hterm = sethamiltonian(sites)
    #@show Hterm
    ene = ene_measurement(Gamma, Lambda, Hterm)
    println("initial energy per site =", ene / 2.0)

    ene_conv_old = ene
    ene_init = ene

    ene_diff_old = 1e-16

    println("\n--- Imaginary-time evolution by iTEBD ---")

    beta_step = beta_step_init

    for step=1:100000
        truncErr = 0.0

        s = 1.3512071919596576

        Gate = exp(-0.5 * s * beta_step * Hterm[1])
        Gamma[1], Gamma[2], Lambda[1], truncErr = Gate_2sites(Gamma[1], Gamma[2], Lambda[1], Lambda[2], Gate, truncErr, maxdim, true)

        Gate = exp(-s * beta_step * Hterm[2])
        Gamma[2], Gamma[1], Lambda[2], truncErr = Gate_2sites(Gamma[2], Gamma[1], Lambda[2], Lambda[1], Gate, truncErr, maxdim, true)

        Gate = exp(0.5 * (s - 1.0) * beta_step * Hterm[1])
        Gamma[1], Gamma[2], Lambda[1], truncErr = Gate_2sites(Gamma[1], Gamma[2], Lambda[1], Lambda[2], Gate, truncErr, maxdim, true)

        Gate = exp( (2.0 * s - 1.0) * beta_step * Hterm[2])
        Gamma[2], Gamma[1], Lambda[2], truncErr = Gate_2sites(Gamma[2], Gamma[1], Lambda[2], Lambda[1], Gate, truncErr, maxdim, true)

        Gate = exp(0.5 * (s-1.0) * beta_step * Hterm[1])
        Gamma[1], Gamma[2], Lambda[1], truncErr = Gate_2sites(Gamma[1], Gamma[2], Lambda[1], Lambda[2], Gate, truncErr, maxdim, true)

        Gate = exp(-s * beta_step * Hterm[2])
        Gamma[2], Gamma[1], Lambda[2], truncErr = Gate_2sites(Gamma[2], Gamma[1], Lambda[2], Lambda[1], Gate, truncErr, maxdim, true)

        Gate = exp(-0.5 * s * beta_step * Hterm[1])
        Gamma[1], Gamma[2], Lambda[1], truncErr = Gate_2sites(Gamma[1], Gamma[2], Lambda[1], Lambda[2], Gate, truncErr, maxdim, true)
                
        if step % 100 == 0
            println("\n", step, "-th iteration, beta_step = ", beta_step)

            d1 = commonind(Gamma[1], Lambda[1])
            d2 = commonind(Gamma[2], Lambda[2])
            println("m[1] = ", dim(d1))
            println("m[2] = ", dim(d2))
            println("truncation error = ", truncErr)

            ipsi = Lambda[2] * Gamma[1] * Lambda[1]
            Sz = dag(ipsi) * noprime!(op("Sz", sites[1]) * ipsi)
            @printf("<Sz(1)> = %.12f\n", real(Sz[1]))
            #Sx = dag(ipsi) * noprime!(op("Sx", sites[1]) * ipsi)
            #@printf("<Sx(1)> = %.12f\n", real(Sx[1]))

            ipsi = Lambda[1] * Gamma[2] * Lambda[2]
            Sz = dag(ipsi) * noprime!(op("Sz", sites[2]) * ipsi)
            @printf("<Sz(2)> = %.12f\n", real(Sz[1]))
            #Sx = dag(ipsi) * noprime!(op("Sx", sites[2]) * ipsi)
            #@printf("<Sx(2)> = %.12f\n", real(Sx[1]))

            ene = ene_measurement(Gamma, Lambda, Hterm)
            @printf("energy per site = %.12f\n", ene / 2.0)

            ene_diff = abs(ene - ene_conv_old)
            println("ene_diff = ", ene_diff)

            if (ene_diff > 0.9 * ene_diff_old && ene_diff < ene_diff_old) || ene_diff < 1e-13
                beta_step = beta_step * 0.5
            end

            ene_conv_old = ene
            ene_diff_old = ene_diff

        end

        if abs(beta_step) < beta_step_min 
            break
        end

    end

	println("\n Imaginary-TE computation has been finished.")

    #addtags!( Gamma[1], "l=1";tags = "Link,v")
    #addtags!(Lambda[1], "l=2";tags = "Link,v")
    #addtags!( Gamma[2], "l=2";tags = "Link,v")
    #addtags!(Lambda[2], "l=1";tags = "Link,v")
    
    psi[1] = Gamma[1] * Lambda[1]
    psi[2] = Gamma[2] * Lambda[2]
    
    f_mps = h5open("MPS-GS.h5", "w")
    write(f_mps, "MPS-GS", psi)
    close(f_mps)

    #@show Lambda[2]
    f_lambda = h5open("Lambda-GS.h5", "w")
    write(f_lambda, "Lambda-GS", denseblocks(Lambda[2]))
    close(f_lambda)

end

function itebd_realtime_2site(maxdim, t_step, step_period, N_t, input_name, output_name, show_output::Bool)
    println("\n Loading the prepared MPS data...")
        
    f_mps = h5open("MPS-$(input_name).h5", "r")
    psi = read(f_mps, "MPS-$(input_name)", MPS)
    close(f_mps)
    f_lambda = h5open("Lambda-$(input_name).h5", "r")
    Ld0 = read(f_lambda, "Lambda-$(input_name)", ITensor)
    close(f_lambda)
    #@show psi
    #@show Lambda0

    sites_temp = siteinds(psi)
    Gamma = []; Lambda = []
    
    aa = dag(uniqueinds(Ld0, psi[1]; tags="Link,u"))
    bb = commoninds(psi[1], Ld0; tags="Link,v")
    inv = ITensor(aa,bb)
    for dd=1:dim(aa)
        inv[dd, dd] = 1.0 / Ld0[dd, dd]
    end
    AA = Ld0 * psi[1]
    ipsi = AA * psi[2]
    AA, Sv, BB = svd(ipsi, inds(AA); maxdim=maxdim, cutoff=eps(Float64))
    push!(Gamma, inv * AA)
    push!(Gamma, BB * inv)
    push!(Lambda, Sv)

    Ld1=Lambda[1]
    aa = dag(uniqueinds(Ld1, Gamma[2]; tags="Link,u"))
    bb = commoninds(Gamma[2], Ld1; tags="Link,v")
    inv = ITensor(aa,bb)
    for dd=1:dim(aa)
        inv[dd, dd] = 1.0 / Ld1[dd, dd]
    end
    AA = Lambda[1] * Gamma[2]
    BB = Gamma[1] * Lambda[1]
    ipsi = AA * Ld0 * BB
    AA, Sv, BB = svd(ipsi, inds(AA); maxdim=maxdim, cutoff=eps(Float64))
    Gamma[2] = inv * AA
    Gamma[1] = BB * inv
    push!(Lambda, Sv)

    Lambda[1] /= norm(Lambda[1])
    Lambda[2] /= norm(Lambda[2])
    truncErr = 1.0-sqrt(norm(Lambda[1])*norm(Lambda[2])/(norm(psi[1])*norm(psi[2])))

    #@show sites_temp
    
    time = 0.0

    println("\n--- Real-time evolution by iTEBD ---")

    Hterm = sethamiltonian(sites_temp)
    Vterm = setperturbation(sites_temp)
    #@show Hterm, Vterm
    
    ipsi = Lambda[2] * Gamma[1] * Lambda[1]
    Sz = dag(ipsi) * noprime!(op("Sz", sites_temp[1]) * ipsi)
    mz1 = real(Sz[1])
    ipsi = Lambda[1] * Gamma[2] * Lambda[2]
    Sz = dag(ipsi) * noprime!(op("Sz", sites_temp[2]) * ipsi)
    mz2 = real(Sz[1])

    open("Sz_$output_name.csv", "w") do f_mag
        @printf(f_mag, "time, Sz(1), Sz(2)\n")
        @printf(f_mag, "%2.8e, %2.25e, %2.25e\n", time, mz1, mz2)
    end

    pol = measurement_2site(sites_temp, Gamma, Lambda)
    @printf("polarization per site = %.12f\n", pol / 2.0)

    open("polarization_$output_name.csv", "w") do f_pol
        @printf(f_pol, "time, polarization\n")
        @printf(f_pol, "%2.8e, %2.25e\n", time, pol / 2.0)
    end

    open("current_$output_name.csv", "w") do f_current
        @printf(f_current, "time, current\n")
    end
    
    d1 = commonind(Gamma[1], Lambda[1])
    d2 = commonind(Gamma[2], Lambda[2])
    #@show d1, d2

    EE1 = 0.0; L=Lambda[1]
    for i=1:dim(d1)
        EE1 -= L[i,i]*log(L[i,i])
    end
    EE2 = 0.0; L=Lambda[2]
    for i=1:dim(d2)
        EE2 -= L[i,i]*log(L[i,i])
    end

    open("TE-Err_$output_name.csv", "w") do f_dim
        @printf(f_dim, "time, dim[1], dim[2], EE[1], EE[2], truncErr\n")
        @printf(f_dim, "%2.8e, %4i, %4i, %.16e, %.16e, %2.8e\n", time, dim(d1), dim(d2), EE1, EE2, truncErr)
    end

    truncErr = 0.0

    for itime = 1:N_t
        s = 1.3512071919596576

        time = itime * t_step

        E = ext_field(time + t_step * (s - 4) / 4)
        Gate = exp(-0.5 * s * im * t_step * (Hterm[1] - E * Vterm[1]))
        Gamma[1], Gamma[2], Lambda[1], truncErr = Gate_2sites(Gamma[1], Gamma[2], Lambda[1], Lambda[2], Gate, truncErr, maxdim, true)

        E = ext_field(time + t_step * (s - 2) / 2)
        Gate = exp(-s * im * t_step * (Hterm[2] - E * Vterm[2]))
        Gamma[2], Gamma[1], Lambda[2], truncErr = Gate_2sites(Gamma[2], Gamma[1], Lambda[2], Lambda[1], Gate, truncErr, maxdim, true)

        E = ext_field(time + t_step * (s - 3) / 4)
        Gate = exp(0.5 * (s - 1.0) * im * t_step * (Hterm[1] - E * Vterm[1]))
        Gamma[1], Gamma[2], Lambda[1], truncErr = Gate_2sites(Gamma[1], Gamma[2], Lambda[1], Lambda[2], Gate, truncErr, maxdim, true)

        E = ext_field(time - t_step / 2 )
        Gate = exp( (2.0 * s - 1.0) * im * t_step * (Hterm[2] - E * Vterm[2]))
        Gamma[2], Gamma[1], Lambda[2], truncErr = Gate_2sites(Gamma[2], Gamma[1], Lambda[2], Lambda[1], Gate, truncErr, maxdim, true)

        E = ext_field(time - t_step * (s + 1) * 4)
        Gate = exp(0.5 * (s-1.0) * im * t_step * (Hterm[1] - E * Vterm[1]))
        Gamma[1], Gamma[2], Lambda[1], truncErr = Gate_2sites(Gamma[1], Gamma[2], Lambda[1], Lambda[2], Gate, truncErr, maxdim, true)

        E = ext_field(time - t_step * s / 2)
        Gate = exp(-s * im * t_step * (Hterm[2] - E * Vterm[2]))
        Gamma[2], Gamma[1], Lambda[2], truncErr = Gate_2sites(Gamma[2], Gamma[1], Lambda[2], Lambda[1], Gate, truncErr, maxdim, true)

        E = ext_field(time - t_step * s / 4)
        Gate = exp(-0.5 * s * im * t_step * (Hterm[1] - E * Vterm[1]))
        Gamma[1], Gamma[2], Lambda[1], truncErr = Gate_2sites(Gamma[1], Gamma[2], Lambda[1], Lambda[2], Gate, truncErr, maxdim, true)

        if itime%step_period == 0
            d1 = commonind(Gamma[1], Lambda[1])
            d2 = commonind(Gamma[2], Lambda[2])
            
            EE1 = 0.0; L=Lambda[1]
            for i=1:dim(d1)
                EE1 -= L[i,i]*log(L[i,i])
            end
            EE2 = 0.0; L=Lambda[2]
            for i=1:dim(d2)
                EE2 -= L[i,i]*log(L[i,i])
            end
            
            open("TE-Err_$output_name.csv", "a") do f_dim
                @printf(f_dim, "%2.8e, %4i, %4i, %.16e, %.16e, %2.8e\n", time, dim(d1), dim(d2), EE1, EE2, truncErr)
            end

            ipsi = Lambda[2] * Gamma[1] * Lambda[1]
            Sz = dag(ipsi) * noprime!(op("Sz", sites_temp[1]) * ipsi)
            mz1 = real(Sz[1])
            ipsi = Lambda[1] * Gamma[2] * Lambda[2]
            Sz = dag(ipsi) * noprime!(op("Sz", sites_temp[2]) * ipsi)
            mz2 = real(Sz[1])

            open("Sz_$output_name.csv", "a") do f_mag
                @printf(f_mag, "%2.8e, %2.25e, %2.25e\n", time, mz1, mz2)
            end

            pol = measurement_2site(sites_temp, Gamma, Lambda)
            current1, current2 = current_measurement1(sites_temp, Gamma, Lambda)
            #current2 = current_measurement2(sites_temp, Gamma, Lambda)
            
            open("polarization_$output_name.csv", "a") do f_pol
                @printf(f_pol, "%2.8e, %2.25e\n", time, pol / 2.0)
            end
    
            open("current_$output_name.csv", "a") do f_current
                @printf(f_current, "%2.8e, %2.25e, %2.25e, %2.25e\n", time, current1 / 2.0, current2 / 2.0, (current1 + current2) / 2.0)
            end

            if show_output==true
                println("\ntime = ", time, ", t_step = ", t_step * step_period)
                println("m[1] = ", dim(d1))
                println("m[2] = ", dim(d2))
                @printf("EE[1] = %.12f\n", EE1)
                @printf("EE[2] = %.12f\n", EE2)
                println("truncation error = ", truncErr)
            
                @printf("<Sz(1)> = %.12f\n", mz1)
                @printf("<Sz(2)> = %.12f\n", mz2)
                @printf("polarization per site = %.12f\n", pol / 2.0)
            end

            truncErr = 0.0

        end

    end

    psi[1] = Gamma[1] * Lambda[1] 
    psi[2] = Gamma[2] * Lambda[2]

    println("Saving wave function ...")

    f_mps = h5open("MPS-temp.h5", "w")
    write(f_mps, "MPS-temp", psi)
    close(f_mps)

    f_lambda = h5open("Lambda-temp.h5", "w")
    write(f_lambda, "Lambda-temp", denseblocks(Lambda[2]))
    close(f_lambda)

    println("\n Real-time evolution computation has been finished.\n")

end
