using ITensors, ITensorMPS

function ext_field(time)
    ### function of E-field (should modify in each simulation !!!) ###

    global E_amp, Omega, Ncyc

    if time > 2.0 * pi * Ncyc / Omega
        A = 0
    else
        A = E_amp * (sin(Omega * time * 0.5 / Ncyc))^2 * cos(Omega * time)
    end

    return A
end

function sethamiltonian(sites)
    global Jxy, Jz, b1, h

    H1  =        Jz * op("Sz",sites[1]) * op("Sz",sites[2])
    H1 += 0.5 * Jxy * op("S+",sites[1]) * op("S-",sites[2])
    H1 += 0.5 * Jxy * op("S-",sites[1]) * op("S+",sites[2])
    H1 -=         h * op("Sz",sites[1]) * op("Id",sites[2])

    H2  =       b1 *  Jz * op("Sz",sites[2]) * op("Sz",sites[1])
    H2 += 0.5 * b1 * Jxy * op("S+",sites[2]) * op("S-",sites[1])
    H2 += 0.5 * b1 * Jxy * op("S-",sites[2]) * op("S+",sites[1])
    H2 +=              h * op("Sz",sites[2]) * op("Id",sites[1])

    return [H1, H2]
end

function setperturbation(sites)
    global Pxy, Pz, b2

    P1  =        -Pz * op("Sz",sites[1]) * op("Sz",sites[2])
    P1 -= 0.5 *  Pxy * op("S+",sites[1]) * op("S-",sites[2])
    P1 -= 0.5 *  Pxy * op("S-",sites[1]) * op("S+",sites[2])

    P2  =       -b2 * Pz * op("Sz",sites[2]) * op("Sz",sites[1])
    P2 -= 0.5 *  b2 * Pxy * op("S+",sites[2]) * op("S-",sites[1])
    P2 -= 0.5 *  b2 * Pxy * op("S-",sites[2]) * op("S+",sites[1])

    return [P1, P2]
end

function measurement_2site(sites, G, L)
    op2body = setperturbation(sites)

    AA = L[2] * G[1]
    BB = G[2] * L[2]
    ipsi = AA * L[1] * BB
    observables  = dag(ipsi)*noprime!(op2body[1]*ipsi)

    AA = L[1] * G[2]
    BB = G[1] * L[1]
    ipsi = AA * L[2] * BB
    observables += dag(ipsi)*noprime!(op2body[2]*ipsi)

    return real(observables[1])
end

function current_measurement1(sites, G, L)
    global Jxy, Jz, Pxy, Pz, b1, b2
    
    AA = L[2] * G[1]
    BB = G[2] * L[2]
    CC = G[1] * L[1]
    ipsi = AA * L[1] * BB

    cur1 = dag(AA)*noprime!(op("S+", sites[1])*ipsi)
    cur1 = dag(L[1]*BB)*noprime!(op("Sz", sites[2])*cur1*CC)
    cur1 = dag(CC)*noprime!(op("S-",sites[1])*cur1)
    
    cur2 = dag(AA)*noprime!(op("S-", sites[1])*ipsi)
    cur2 = dag(L[1]*BB)*noprime!(op("Sz", sites[2])*cur2*CC)
    cur2 = dag(CC)*noprime!(op("S+",sites[1])*cur2)
    
    cur3 = dag(AA)*noprime!(op("Sz", sites[1])*ipsi)
    cur3 = dag(L[1]*BB)*noprime!(op("S+", sites[2])*cur3*CC)
    cur3 = dag(CC)*noprime!(op("S-",sites[1])*cur3)
    
    cur4 = dag(AA)*noprime!(op("Sz", sites[1])*ipsi)
    cur4 = dag(L[1]*BB)*noprime!(op("S-", sites[2])*cur4*CC)
    cur4 = dag(CC)*noprime!(op("S+",sites[1])*cur4)
    
    cur5 = dag(AA)*noprime!(op("S+", sites[1])*ipsi)
    cur5 = dag(L[1]*BB)*noprime!(op("S-", sites[2])*cur5*CC)
    cur5 = dag(CC)*noprime!(op("Sz",sites[1])*cur5)
    
    cur6 = dag(AA)*noprime!(op("S-", sites[1])*ipsi)
    cur6 = dag(L[1]*BB)*noprime!(op("S+", sites[2])*cur6*CC)
    cur6 = dag(CC)*noprime!(op("Sz",sites[1])*cur6)

    current1 = 0.5*im*(b2-b1)*Jxy*Pxy*(cur1-cur2)
    current2 = 0.5*im*(
        (b1*Jxy*Pz-b2*Jz*Pxy)*(cur3-cur4)
        -(b2*Jxy*Pz-b1*Jz*Pxy)*(cur5-cur6)
    )
    
    AA = L[1] * G[2]
    BB = G[1] * L[1]
    CC = G[2] * L[2]
    ipsi = AA * L[2] * BB

    cur1 = dag(AA)*noprime!(op("S+", sites[2])*ipsi)
    cur1 = dag(L[2]*BB)*noprime!(op("Sz", sites[1])*cur1*CC)
    cur1 = dag(CC)*noprime!(op("S-",sites[2])*cur1)
    
    cur2 = dag(AA)*noprime!(op("S-", sites[2])*ipsi)
    cur2 = dag(L[2]*BB)*noprime!(op("Sz", sites[1])*cur2*CC)
    cur2 = dag(CC)*noprime!(op("S+",sites[2])*cur2)
    
    cur3 = dag(AA)*noprime!(op("Sz", sites[2])*ipsi)
    cur3 = dag(L[2]*BB)*noprime!(op("S+", sites[1])*cur3*CC)
    cur3 = dag(CC)*noprime!(op("S-",sites[2])*cur3)
    
    cur4 = dag(AA)*noprime!(op("Sz", sites[2])*ipsi)
    cur4 = dag(L[2]*BB)*noprime!(op("S-", sites[1])*cur4*CC)
    cur4 = dag(CC)*noprime!(op("S+",sites[2])*cur4)
    
    cur5 = dag(AA)*noprime!(op("S+", sites[2])*ipsi)
    cur5 = dag(L[2]*BB)*noprime!(op("S-", sites[1])*cur5*CC)
    cur5 = dag(CC)*noprime!(op("Sz",sites[2])*cur5)
    
    cur6 = dag(AA)*noprime!(op("S-", sites[2])*ipsi)
    cur6 = dag(L[2]*BB)*noprime!(op("S+", sites[1])*cur6*CC)
    cur6 = dag(CC)*noprime!(op("Sz",sites[2])*cur6)

    current1 -= 0.5*im*(b2-b1)*Jxy*Pxy*(cur1-cur2)
    current2 += 0.5*im*(
        (b2*Jxy*Pz-b1*Jz*Pxy)*(cur3-cur4)
        -(b1*Jxy*Pz-b2*Jz*Pxy)*(cur5-cur6)
    )
    
    return real(current1[1]), real(current2[1])
end

function current_measurement2(sites, G, L)
    global Jxy, Jz, Pxy, Pz, b1, b2

    J1  = 0.5*im * (Jz*Pxy - Jxy*Pz) * op("S+ * Sz",sites[1]) * op("S+",sites[2])
    J1 -= 0.5*im * (Jz*Pxy - Jxy*Pz) * op("S- * Sz",sites[1]) * op("S-",sites[2])
    J1 += 0.5*im * (Jz*Pxy - Jxy*Pz) * op("S+",sites[1]) * op("Sz * S+",sites[2])
    J1 -= 0.5*im * (Jz*Pxy - Jxy*Pz) * op("S-",sites[1]) * op("Sz * S-",sites[2])

    J2  = 0.5*im*b1*b2 * (Jz*Pxy - Jxy*Pz) * op("S+ * Sz",sites[2]) * op("S+",sites[1])
    J2 -= 0.5*im*b1*b2 * (Jz*Pxy - Jxy*Pz) * op("S- * Sz",sites[2]) * op("S-",sites[1])
    J2 += 0.5*im*b1*b2 * (Jz*Pxy - Jxy*Pz) * op("S+",sites[2]) * op("Sz * S+",sites[1])
    J2 -= 0.5*im*b1*b2 * (Jz*Pxy - Jxy*Pz) * op("S-",sites[2]) * op("Sz * S-",sites[1])

    AA = L[2] * G[1]
    BB = G[2] * L[2]
    ipsi = AA * L[1] * BB
    current = dag(ipsi)*noprime!(J1*ipsi)

    AA = L[1] * G[2]
    BB = G[1] * L[1]
    ipsi = AA * L[2] * BB
    current += dag(ipsi)*noprime!(J2*ipsi)

    return real(current[1])
end


### under construction ###############################################
