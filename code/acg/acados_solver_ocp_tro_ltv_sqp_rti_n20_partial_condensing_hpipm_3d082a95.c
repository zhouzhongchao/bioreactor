/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h> // memcpy
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific

#include "tro_ltv_sqp_rti_n20_partial_condensing_hpipm_model/tro_ltv_sqp_rti_n20_partial_condensing_hpipm_model.h"


#include "tro_ltv_sqp_rti_n20_partial_condensing_hpipm_cost/tro_ltv_sqp_rti_n20_partial_condensing_hpipm_cost.h"



#include "acados_solver_ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95.h"

#define NX     OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NX
#define NZ     OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NZ
#define NU     OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NU
#define NP     OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NP
#define NP_GLOBAL     OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NP_GLOBAL
#define NY0    OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NY0
#define NY     OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NY
#define NYN    OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NYN

#define NBX    OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NBX
#define NBX0   OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NBX0
#define NBU    OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NBU
#define NG     OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NG
#define NBXN   OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NBXN
#define NGN    OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NGN

#define NH     OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NH
#define NHN    OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NHN
#define NH0    OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NH0
#define NPHI   OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NPHI
#define NPHIN  OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NPHIN
#define NPHI0  OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NPHI0
#define NR     OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NR

#define NS     OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NS
#define NS0    OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NS0
#define NSN    OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NSN

#define NSBX   OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NSBX
#define NSBU   OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NSBU
#define NSH0   OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NSH0
#define NSH    OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NSH
#define NSHN   OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NSHN
#define NSG    OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NSG
#define NSPHI0 OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NSPHI0
#define NSPHI  OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NSPHI
#define NSPHIN OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NSPHIN
#define NSGN   OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NSGN
#define NSBXN  OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_NSBXN





// ** solver data **

ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule * ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule));
    ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule *capsule = (ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule *) capsule_mem;

    return capsule;
}


int ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_free_capsule(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule)
{
    int N_shooting_intervals = OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_update_time_steps(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule, int N, double* new_time_steps)
{

    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;

}

/**
 * Internal function for ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create: step 1
 */
void ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP_RTI;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->relaxed_ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->nlp_cost[0] = EXTERNAL;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = EXTERNAL;

    nlp_solver_plan->nlp_cost[N] = EXTERNAL;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = DISCRETE_MODEL;
        // discrete dynamics does not need sim solver option, this field is ignored
        nlp_solver_plan->sim_solver_plan[i].sim_solver = INVALID_SIM_SOLVER;
    }

    nlp_solver_plan->nlp_constraints[0] = BGH;

    for (int i = 1; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;

    nlp_solver_plan->regularization = CONVEXIFY;

    nlp_solver_plan->globalization = FIXED_STEP;
}


static ocp_nlp_dims* ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_setup_dimensions(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 18
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;
    int* np  = intNp1mem + (N+1)*17;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
        np[i]     = NP;
    }

    // for initial state
    nbx[0] = NBX0;
    nsbx[0] = 0;
    ns[0] = NS0;
    
    nbxe[0] = 18;
    
    ny[0] = NY0;
    nh[0] = NH0;
    nsh[0] = NSH0;
    nsphi[0] = NSPHI0;
    nphi[0] = NPHI0;


    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "np", np);

    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "np_global", 0);
    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "n_global_data", 0);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    free(intNp1mem);

    return nlp_dims;
}


/**
 * Internal function for ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create: step 3
 */
void ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_setup_functions(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_external_param_casadi_create(&capsule->__CAPSULE_FNC__, &ext_fun_opts); \
    } while(false)

    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);


    ext_fun_opts.external_workspace = true;
    if (N > 0)
    {
        // external cost
        MAP_CASADI_FNC(ext_cost_0_fun, tro_ltv_sqp_rti_n20_partial_condensing_hpipm_cost_ext_cost_0_fun);
        MAP_CASADI_FNC(ext_cost_0_fun_jac, tro_ltv_sqp_rti_n20_partial_condensing_hpipm_cost_ext_cost_0_fun_jac);
        MAP_CASADI_FNC(ext_cost_0_fun_jac_hess, tro_ltv_sqp_rti_n20_partial_condensing_hpipm_cost_ext_cost_0_fun_jac_hess);



    
        // discrete dynamics
        capsule->discr_dyn_phi_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++)
        {
            MAP_CASADI_FNC(discr_dyn_phi_fun[i], tro_ltv_sqp_rti_n20_partial_condensing_hpipm_dyn_disc_phi_fun);
        }

        capsule->discr_dyn_phi_fun_jac_ut_xt = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++)
        {
            MAP_CASADI_FNC(discr_dyn_phi_fun_jac_ut_xt[i], tro_ltv_sqp_rti_n20_partial_condensing_hpipm_dyn_disc_phi_fun_jac);
        }

    

    

    
        capsule->discr_dyn_phi_fun_jac_ut_xt_hess = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++)
        {
            MAP_CASADI_FNC(discr_dyn_phi_fun_jac_ut_xt_hess[i], tro_ltv_sqp_rti_n20_partial_condensing_hpipm_dyn_disc_phi_fun_jac_hess);
        }
        // external cost
        capsule->ext_cost_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++)
        {
            MAP_CASADI_FNC(ext_cost_fun[i], tro_ltv_sqp_rti_n20_partial_condensing_hpipm_cost_ext_cost_fun);
        }

        capsule->ext_cost_fun_jac = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++)
        {
            MAP_CASADI_FNC(ext_cost_fun_jac[i], tro_ltv_sqp_rti_n20_partial_condensing_hpipm_cost_ext_cost_fun_jac);
        }

        capsule->ext_cost_fun_jac_hess = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++)
        {
            MAP_CASADI_FNC(ext_cost_fun_jac_hess[i], tro_ltv_sqp_rti_n20_partial_condensing_hpipm_cost_ext_cost_fun_jac_hess);
        }

        


        


        
    } // N > 0
    // external cost - function
    MAP_CASADI_FNC(ext_cost_e_fun, tro_ltv_sqp_rti_n20_partial_condensing_hpipm_cost_ext_cost_e_fun);

    // external cost - jacobian
    MAP_CASADI_FNC(ext_cost_e_fun_jac, tro_ltv_sqp_rti_n20_partial_condensing_hpipm_cost_ext_cost_e_fun_jac);

    // external cost - hessian
    MAP_CASADI_FNC(ext_cost_e_fun_jac_hess, tro_ltv_sqp_rti_n20_partial_condensing_hpipm_cost_ext_cost_e_fun_jac_hess);

    // external cost - jacobian wrt params
    

    

    

#undef MAP_CASADI_FNC
}


/**
 * Internal function for ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create: step 5
 */
void ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_set_default_parameters(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule)
{

    const int N = capsule->nlp_solver_plan->N;

    // initialize parameters to initial value
    
    double* p = calloc(NP, sizeof(double));

    for (int i = 0; i <= N; i++) {
        ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_update_params(capsule, i, p, NP);
    }
    free(p);


    // no global parameters defined
}


/**
 * Internal function for ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create: step 5
 */
void ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_setup_nlp_in_numerical_values(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = capsule->nlp_in;
    /************************************************
    *  nlp_out
    ************************************************/
    ocp_nlp_out * nlp_out = capsule->nlp_out;

    // set up time_steps and cost_scaling

    if (new_time_steps)
    {
        // NOTE: this sets scaling and time_steps
        ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_update_time_steps(capsule, N, new_time_steps);
    }
    else
    {
        // set time_steps
    
        double time_step = 0.05;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
        }
        // set cost scaling
        double* cost_scaling = malloc((N+1)*sizeof(double));
        cost_scaling[0] = 0.05;
        cost_scaling[1] = 0.05;
        cost_scaling[2] = 0.05;
        cost_scaling[3] = 0.05;
        cost_scaling[4] = 0.05;
        cost_scaling[5] = 0.05;
        cost_scaling[6] = 0.05;
        cost_scaling[7] = 0.05;
        cost_scaling[8] = 0.05;
        cost_scaling[9] = 0.05;
        cost_scaling[10] = 0.05;
        cost_scaling[11] = 0.05;
        cost_scaling[12] = 0.05;
        cost_scaling[13] = 0.05;
        cost_scaling[14] = 0.05;
        cost_scaling[15] = 0.05;
        cost_scaling[16] = 0.05;
        cost_scaling[17] = 0.05;
        cost_scaling[18] = 0.05;
        cost_scaling[19] = 0.05;
        cost_scaling[20] = 1;
        for (int i = 0; i <= N; i++)
        {
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &cost_scaling[i]);
        }
        free(cost_scaling);
    }



    /**** Cost ****/






    /**** Constraints ****/

    // bounds for initial stage
    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;
    idxbx0[5] = 5;
    idxbx0[6] = 6;
    idxbx0[7] = 7;
    idxbx0[8] = 8;
    idxbx0[9] = 9;
    idxbx0[10] = 10;
    idxbx0[11] = 11;
    idxbx0[12] = 12;
    idxbx0[13] = 13;
    idxbx0[14] = 14;
    idxbx0[15] = 15;
    idxbx0[16] = 16;
    idxbx0[17] = 17;

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);
    // idxbxe_0
    int* idxbxe_0 = malloc(18 * sizeof(int));
    idxbxe_0[0] = 0;
    idxbxe_0[1] = 1;
    idxbxe_0[2] = 2;
    idxbxe_0[3] = 3;
    idxbxe_0[4] = 4;
    idxbxe_0[5] = 5;
    idxbxe_0[6] = 6;
    idxbxe_0[7] = 7;
    idxbxe_0[8] = 8;
    idxbxe_0[9] = 9;
    idxbxe_0[10] = 10;
    idxbxe_0[11] = 11;
    idxbxe_0[12] = 12;
    idxbxe_0[13] = 13;
    idxbxe_0[14] = 14;
    idxbxe_0[15] = 15;
    idxbxe_0[16] = 16;
    idxbxe_0[17] = 17;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbxe", idxbxe_0);
    free(idxbxe_0);












    /* constraints that are the same for initial and intermediate */
    // u
    int* idxbu = malloc(NBU * sizeof(int));
    idxbu[0] = 0;
    idxbu[1] = 1;
    idxbu[2] = 2;
    idxbu[3] = 3;
    idxbu[4] = 4;
    idxbu[5] = 5;
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    ubu[0] = 2500;
    ubu[1] = 2500;
    ubu[2] = 2500;
    ubu[3] = 2500;
    ubu[4] = 2500;
    ubu[5] = 2500;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);






    /* Path constraints */














    /* terminal constraints */




















}

// this function only sets external functions, numerical values are set in ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_setup_nlp_in_numerical_values
void ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_setup_nlp_in(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule, const int N)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = capsule->nlp_in;
    /************************************************
    *  nlp_out
    ************************************************/
    ocp_nlp_out * nlp_out = capsule->nlp_out;


    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun", &capsule->discr_dyn_phi_fun[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun_jac",
                                   &capsule->discr_dyn_phi_fun_jac_ut_xt[i]);
        
        
        
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun_jac_hess",
                                   &capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i]);
    }

    /**** Cost ****/
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_fun", &capsule->ext_cost_0_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_fun_jac", &capsule->ext_cost_0_fun_jac);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_fun_jac_hess", &capsule->ext_cost_0_fun_jac_hess);
    
    
    
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun", &capsule->ext_cost_fun[i-1]);
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun_jac", &capsule->ext_cost_fun_jac[i-1]);
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun_jac_hess", &capsule->ext_cost_fun_jac_hess[i-1]);
        
        
        
    }
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun", &capsule->ext_cost_e_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun_jac", &capsule->ext_cost_e_fun_jac);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun_jac_hess", &capsule->ext_cost_e_fun_jac_hess);
    
    
    

    /**** Constraints ****/

    // bounds for initial stage






    /* terminal constraints */
}


static void ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_set_opts(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/


    int nlp_solver_exact_hessian = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess", &nlp_solver_exact_hessian);

    int exact_hess_dyn = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess_dyn", &exact_hess_dyn);

    int exact_hess_cost = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess_cost", &exact_hess_cost);

    int exact_hess_constr = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess_constr", &exact_hess_constr);

    int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);

    double globalization_fixed_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_fixed_step_length", &globalization_fixed_step_length);




    int with_solution_sens_wrt_params_forw = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params_forw", &with_solution_sens_wrt_params_forw);

    int with_solution_sens_wrt_params_adj = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params_adj", &with_solution_sens_wrt_params_adj);

    int with_value_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_value_sens_wrt_params", &with_value_sens_wrt_params);

    double solution_sens_qp_t_lam_min = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "solution_sens_qp_t_lam_min", &solution_sens_qp_t_lam_min);

    int globalization_full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_full_step_dual", &globalization_full_step_dual);

    double levenberg_marquardt = 0.00000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;const int qp_solver_cond_N_ori = 5;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);
    double reg_epsilon = 0.0001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "reg_epsilon", &reg_epsilon);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

    bool store_iterates = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "store_iterates", &store_iterates);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");



    int qp_solver_t0_init = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_t0_init", &qp_solver_t0_init);




    int as_rti_iter = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_iter", &as_rti_iter);

    int as_rti_level = 4;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_level", &as_rti_level);

    int rti_log_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_residuals", &rti_log_residuals);

    int rti_log_only_available_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_only_available_residuals", &rti_log_only_available_residuals);

    bool with_anderson_acceleration = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_anderson_acceleration", &with_anderson_acceleration);

    double anderson_activation_threshold = 10;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "anderson_activation_threshold", &anderson_activation_threshold);

    int qp_solver_iter_max = 100;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);


    int qp_solver_warm_start = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_warm_start", &qp_solver_warm_start);

    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "cost_numerical_hessian", &ext_cost_num_hess);
    }
    ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, N, "cost_numerical_hessian", &ext_cost_num_hess);
}


/**
 * Internal function for ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create: step 7
 */
void ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_set_nlp_out(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with x0


    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create: step 9
 */
int ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_precompute(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_with_discretization(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_N && !new_time_steps) {
        fprintf(stderr, "ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, OCP_TRO_LTV_SQP_RTI_N20_PARTIAL_CONDENSING_HPIPM_3D082A95_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 2) create and set dimensions
    capsule->nlp_dims = ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_setup_dimensions(capsule);

    // 3) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_set_opts(capsule);

    // 4) create and set nlp_out
    // 4.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 4.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_set_nlp_out(capsule);

    // 5) create nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);

    // 6) setup functions, nlp_in and default parameters
    ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_setup_functions(capsule);
    ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_setup_nlp_in(capsule, N);
    ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_setup_nlp_in_numerical_values(capsule, N, new_time_steps);
    ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_set_default_parameters(capsule);

    // 7) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);


    // 8) do precomputations
    int status = ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_update_qp_solver_cond_N(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // -> 9) do precomputations
    int status = ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_precompute(capsule);
    return status;
}


int ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_reset(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule, int reset_qp_solver_mem, int reset_numerical_values, int reset_solver_options, int reset_x_to_x0_bar)
{
    // set initialization to all zeros
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    // sets primal and dual iterates to zero
    ocp_nlp_out_set_values_to_zero(nlp_config, nlp_dims, nlp_out);

    // reset integrator memory
    ocp_nlp_solver_reset_integrator_memory(nlp_solver, nlp_in, nlp_out);
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    if (reset_numerical_values)
    {
        // reset parameters to initial values
        ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_set_default_parameters(capsule);

        // reset numerical values in nlp_in
        ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_setup_nlp_in_numerical_values(capsule, N, NULL);
    }

    if (reset_solver_options)
    {
        // reset solver options to initial values
        ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_create_set_opts(capsule);
    }

    if (reset_x_to_x0_bar)
    {double* buffer = calloc(NX, sizeof(double));
        ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, 0, "lbx", buffer);
        for (int i=0; i<N+1; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", buffer);
        }
        free(buffer);
    }
    return 0;
}




int ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_update_params(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 254;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, "parameter_values", p);

    return solver_status;
}


int ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_update_params_sparse(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    ocp_nlp_in_set_params_sparse(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, idx, p, n_update);

    return 0;
}


int ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_set_p_global_and_precompute_dependencies(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule, double* data, int data_len)
{

    // printf("No global_data, ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_set_p_global_and_precompute_dependencies does nothing.\n");
    return 0;
}




int ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_solve(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



int ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_setup_qp_matrices_and_factorize(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule)
{
    int solver_status = ocp_nlp_setup_qp_matrices_and_factorize(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}






int ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_free(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun[i]);
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun_jac_ut_xt[i]);
        
        

        
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i]);
    }
    free(capsule->discr_dyn_phi_fun);
    free(capsule->discr_dyn_phi_fun_jac_ut_xt);
  
  
  
    free(capsule->discr_dyn_phi_fun_jac_ut_xt_hess);

    // cost
    external_function_external_param_casadi_free(&capsule->ext_cost_0_fun);
    external_function_external_param_casadi_free(&capsule->ext_cost_0_fun_jac);
    external_function_external_param_casadi_free(&capsule->ext_cost_0_fun_jac_hess);
    
    
    
    for (int i = 0; i < N - 1; i++)
    {
        external_function_external_param_casadi_free(&capsule->ext_cost_fun[i]);
        external_function_external_param_casadi_free(&capsule->ext_cost_fun_jac[i]);
        external_function_external_param_casadi_free(&capsule->ext_cost_fun_jac_hess[i]);
        
        
        
    }
    free(capsule->ext_cost_fun);
    free(capsule->ext_cost_fun_jac);
    free(capsule->ext_cost_fun_jac_hess);
    external_function_external_param_casadi_free(&capsule->ext_cost_e_fun);
    external_function_external_param_casadi_free(&capsule->ext_cost_e_fun_jac);
    external_function_external_param_casadi_free(&capsule->ext_cost_e_fun_jac_hess);
    
    
    

    // constraints



    return 0;
}


void ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_print_stats(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule)
{
    int nlp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_solver, "nlp_iter", &nlp_iter);
    ocp_nlp_get(capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_solver, "stat_m", &stat_m);


    int stat_n_max = 16;
    if (stat_n > stat_n_max)
    {
        printf("stat_n_max = %d is too small, increase it in the template!\n", stat_n_max);
        exit(1);
    }
    double stat[1616];
    ocp_nlp_get(capsule->nlp_solver, "statistics", stat);

    int nrow = nlp_iter+1 < stat_m ? nlp_iter+1 : stat_m;


    printf("iter\tqp_stat\tqp_iter\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            tmp_int = (int) stat[i + j * nrow];
            printf("%d\t", tmp_int);
        }
        printf("\n");
    }
}

int ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_custom_update(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_get_nlp_in(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_get_nlp_out(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_get_sens_out(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_get_nlp_solver(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_get_nlp_config(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule) { return capsule->nlp_config; }
void *ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_get_nlp_opts(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_get_nlp_dims(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_acados_get_nlp_plan(ocp_tro_ltv_sqp_rti_n20_partial_condensing_hpipm_3d082a95_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
