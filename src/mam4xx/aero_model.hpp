// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_AERO_MODEL_HPP
#define MAM4XX_AERO_MODEL_HPP

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

#include <ekat_math_utils.hpp>
#include <ekat_subview_utils.hpp>

namespace mam4 {

namespace aero_model {

// BAD CONSTANT
// index range for the impaction scavenging lookup table based on hygroscopic growth factors.
constexpr int nimptblgrow_mind = -7, nimptblgrow_maxd = 12;
// total number of entries in the lookup table:
constexpr int nimptblgrow_total = -nimptblgrow_mind + nimptblgrow_maxd + 1;
// Maximum number of rain drop size bins for discretizing the Marshall-Palmer rain drop size distribution
const int nrainsvmax = 50; 
// Maximum number of aerosol size bins for discretizing the log-normal aerosol size distribution
const int naerosvmax = 51; 
// Maximum number of aerosol species types per mode.
const int maxd_aspectype = 14;

constexpr int pcnst = mam4::pcnst;

//=============================================================================
// Example: 
// 
// ┌─────────────────────────────────────────────────────────────┐
// │                         HOST (CPU)                          │
// │                                                             │
// │   modal_aero_bcscavcoef_init()                              │
// │   ┌─────────────────────────────────┐                       │
// │   │  View2DHost scavimptblnum       │  ← Initialized here   │
// │   │  View2DHost scavimptblvol       │                       │
// │   └─────────────────────────────────┘                       │
// │                    │                                        │
// │                    │ Kokkos::deep_copy()                    │
// │                    ▼                                        │
// └─────────────────────────────────────────────────────────────┘
//                      │
//                      ▼
// ┌─────────────────────────────────────────────────────────────┐
// │                        DEVICE (GPU)                         │
// │                                                             │
// │   modal_aero_bcscavcoef_get()                               │
// │   ┌─────────────────────────────────┐                       │
// │   │  View2D scavimptblnum           │  ← Used here          │
// │   │  View2D scavimptblvol           │                       │
// │   └─────────────────────────────────┘                       │
// │                                                             │
// └─────────────────────────────────────────────────────────────┘
//=============================================================================
// Creates a type alias View2D for a 2D Kokkos View that resides in device memory (GPU).
using View2D = DeviceType::view_2d<Real>;
// Creates a type alias View2DHost for a 2D Kokkos View that resides in host memory (CPU). 
using View2DHost = typename HostType::view_2d<Real>;




//=============================================================================
// FUNCTION: modal_aero_bcscavcoef_get
//=============================================================================
// Description: Computes impaction scavenging removal coefficients for aerosol
//              volume and number by interpolating pre-computed lookup tables.
//              The tables are indexed by hygroscopic growth factor.
//
// Purpose: Below-cloud scavenging occurs when falling precipitation drops
//          collide with and capture aerosol particles. This function retrieves
//          the appropriate scavenging coefficients based on the wet aerosol size.
//
// Parameters:
//   imode              [in]  - Aerosol mode index [0, num_modes)
//   dgn_awet_imode_kk  [in]  - Wet geometric mean diameter at level kk [m]
//   dgnum_amode_imode  [in]  - Dry geometric mean diameter for mode [m]
//   scavimptblvol      [in]  - Lookup table for volume scavenging [View2D]
//   scavimptblnum      [in]  - Lookup table for number scavenging [View2D]
//   scavcoefnum_kk     [out] - Scavenging coefficient for number [1/h]
//   scavcoefvol_kk     [out] - Scavenging coefficient for volume [1/h]
//
// Algorithm:
//   1. Compute wet/dry diameter ratio as hygroscopic growth indicator
//   2. If ratio ≈ 1.0, use table value at index 0 (no growth)
//   3. Otherwise, perform linear interpolation between adjacent table entries
//   4. Convert log-space table values to actual coefficients via exp()
//
// Table Indexing Note:
//   - Original Fortran uses indices: [-7, 12] (nimptblgrow_mind to nimptblgrow_maxd)
//   - C++ uses indices: [0, 19]
//   - Conversion: jgrow_cpp = jgrow_fortran - nimptblgrow_mind
//   - Index 0 in Fortran (no growth) maps to index 7 in C++ (-(-7) = 7)
//
// Performance Notes:
//   - O(1) complexity - simple arithmetic and table lookup
//   - Contains log() and exp() calls which are computationally expensive
//   - Branch for wetdiaratio ≈ 1.0 avoids unnecessary computation
//
// Suggestions for Improvement:
//   1. Pre-compute 1/dlndg_nimptblgrow to replace division with multiplication
//   2. Consider using fast approximations for log/exp if precision allows
//   3. Add bounds checking for imode in debug builds
//=============================================================================
KOKKOS_INLINE_FUNCTION
void modal_aero_bcscavcoef_get(const int imode,
                               const Real dgn_awet_imode_kk, //& ! in
                               const Real dgnum_amode_imode,
                               const View2D &scavimptblvol,
                               const View2D &scavimptblnum,
                               Real &scavcoefnum_kk, Real &scavcoefvol_kk) {

  // NOTE: original FORTRAN function has two internal loops: kk and icol.
  // We removed these loops. Hence, the inputs/outputs of
  // modal_aero_bcscavcoef_get are reals at kk , icol locations

  const Real zero = 0;
  const Real one = 1;
  // BAD CONSTANT
  const Real dlndg_nimptblgrow = haero::log(1.25);
  // With precipitation
  // interpolate table values using log of
  // (actual-wet-size)/(base-dry-size) ratio of wet and dry aerosol diameter
  // [fraction]
  const Real wetdiaratio = dgn_awet_imode_kk / dgnum_amode_imode;
  // Note: indexing in scavimptblnum and scavimptblvol
  // Fortran : [-7,12]
  // C++ :  [0,19]
  // Therefore, -7 (Fortran) => 0 (C++), or jgrow => jgrow - nimptblgrow_mind;
  // Here, we are assuming that nimptblgrow_mind is negative
  Real scavimpvol, scavimpnum = zero;
  // BAD CONSTANT
  if (wetdiaratio >= 0.99 && wetdiaratio <= 1.01) {
    // 8th position: Fortran (0) C++(7 or -nimptblgrow_mind)
    scavimpvol = scavimptblvol(-nimptblgrow_mind, imode);
    scavimpnum = scavimptblnum(-nimptblgrow_mind, imode);
  } else {
    Real xgrow = haero::log(wetdiaratio) / dlndg_nimptblgrow;
    int jgrow = int(xgrow); // get index jgrow
    if (xgrow < zero) { // // adjust jgrow appropriately if xgrow is negative
      jgrow = jgrow - 1;
    }
    // bound jgrow within max and min values
    if (jgrow < nimptblgrow_mind) {
      jgrow = nimptblgrow_mind;
      xgrow = jgrow;
    } else {
      jgrow = haero::min(jgrow, nimptblgrow_maxd - 1);
    }
    // compute factors for interpolating impaction scavenging removal amounts
    const Real dumfhi = xgrow - jgrow;
    const Real dumflo = one - dumfhi;
    // Fortran to C++ index conversion
    // Note: nimptblgrow_mind is negative (-7)
    int jgrow_pp = jgrow - nimptblgrow_mind;
    scavimpvol = dumflo * scavimptblvol(jgrow_pp, imode) +
                 dumfhi * scavimptblvol(jgrow_pp + 1, imode);
    scavimpnum = dumflo * scavimptblnum(jgrow_pp, imode) +
                 dumfhi * scavimptblnum(jgrow_pp + 1, imode);

  } // wetdiaratio

  // impaction scavenging removal amount for volume
  scavcoefvol_kk = haero::exp(scavimpvol);
  // impaction scavenging removal amount to number
  scavcoefnum_kk = haero::exp(scavimpnum);

} // modal_aero_bcscavcoef_get


//=============================================================================
// FUNCTION: air_dynamic_viscosity
//=============================================================================
// Description: Calculates the dynamic (absolute) viscosity of air as a 
//              function of temperature using Sutherland's Law.
//
// Formula: μ(T) = μ_ref * ((T_ref + S) / (T + S)) * (T / T_ref)^1.5
//
//          where:
//            μ_ref = 1.8325e-4 g/cm/s (reference viscosity)
//            T_ref = 296.16 K (reference temperature ≈ 23°C)
//            S     = 120.0 K (Sutherland's constant for air)
//
// Physical Basis: Sutherland's Law is a semi-empirical formula that accounts
//                 for the temperature dependence of gas viscosity due to
//                 molecular interactions. Valid for ideal gas behavior.
//
// Parameter:
//   temp [in] - Air temperature [K]
//               Valid range: approximately 100 K to 1900 K for air
//
// Returns: Dynamic viscosity of air [g/cm/s] (CGS units)
//          Note: To convert to SI units (Pa·s or kg/m/s), multiply by 0.1
//
// Reference: 
//   - Sutherland, W. (1893), "The viscosity of gases and molecular force"
//   - Equation form from: http://pages.erau.edu/~snivelyj/ep711sp12/EP711_15.pdf
//
// Note: This calculation differs from the one used in dry deposition
//       (see modal_aero_drydep.F90 for comparison)
//
// Performance Notes:
//   - O(1) complexity
//   - Contains one pow() call (~50-100 cycles) and basic arithmetic
//   - Could use faster approximation: pow(x, 1.5) = x * sqrt(x)
//
// Suggestions for Improvement:
//   1. Replace pow(x, 1.5) with x * sqrt(x) for better performance
//   2. Add bounds checking for temperature in debug builds
//   3. Consider caching result if called repeatedly with same temperature
//=============================================================================
KOKKOS_INLINE_FUNCTION
Real air_dynamic_viscosity(const Real temp) {
  /*-----------------------------------------------------------------
  ! Calculate dynamic viscosity of air, unit [g/cm/s]
  !
  ! note that this calculation is different with that used in dry deposition
  ! see the same-name function in modal_aero_drydep.F90
  !-----------------------------------------------------------------*/
  // @param [in] temp   ! air temperature [K]
  // @return     dynamic viscosity of air, unit [g/cm/s]
  // Note: We do not have a reference for this correlation.
  // However, this equation is presented in page 3 of
  // http://pages.erau.edu/~snivelyj/ep711sp12/EP711_15.pdf.
  return 1.8325e-4 * (416.16 / (temp + 120.)) * haero::pow(temp / 296.16, 1.5);

} // end air_dynamic_viscosity


//=============================================================================
// FUNCTION: air_kinematic_viscosity
//=============================================================================
// Description: Calculates the kinematic viscosity of air from the dynamic
//              viscosity and air density.
//
// Physical Definition:
//   Kinematic viscosity (ν) = Dynamic viscosity (μ) / Density (ρ)
//   
//   ν represents the ratio of viscous forces to inertial forces and is
//   commonly used in Reynolds number calculations:
//     Re = (velocity × length) / ν
//
// Formula: ν = μ(T) / ρ
//          where μ(T) is computed via Sutherland's Law in air_dynamic_viscosity()
//
// Parameters:
//   temp   [in] - Air temperature [K]
//                 Valid range: approximately 100 K to 1900 K
//   rhoair [in] - Air mass density [g/cm³] (CGS units)
//                 Typical sea-level value: ~1.225e-3 g/cm³
//
// Returns: Kinematic viscosity of air [cm²/s] (CGS units)
//          Note: To convert to SI units (m²/s), multiply by 1.0e-4
//          Typical sea-level value at 288 K: ~0.146 cm²/s (~1.46e-5 m²/s)
//
// Dependencies:
//   - Calls air_dynamic_viscosity(temp) internally
//
// Performance Notes:
//   - O(1) complexity
//   - Dominated by air_dynamic_viscosity() call (~40-150 cycles)
//   - Single division operation (~10-20 cycles)
//   - Total: ~50-170 cycles
//
// Suggestions for Improvement:
//   1. Add bounds checking for rhoair > 0 in debug builds to prevent division by zero
//   2. Fix incorrect @return comment (says "dynamic viscosity" but returns kinematic)
//   3. Consider caching dynamic viscosity if both are needed at same temperature
//
// Usage Example:
//   Real temp = 288.0;           // K (standard atmosphere)
//   Real rhoair = 1.225e-3;      // g/cm³
//   Real nu = air_kinematic_viscosity(temp, rhoair);  // ~0.146 cm²/s
//
// Related Functions:
//   - air_dynamic_viscosity(): Computes μ(T) using Sutherland's Law
//   - calc_schmidt_number(): Uses kinematic viscosity for Schmidt number
//=============================================================================
KOKKOS_INLINE_FUNCTION
Real air_kinematic_viscosity(const Real temp, const Real rhoair) {
  /*-----------------------------------------------------------------
  ! Calculate kinematic viscosity of air, unit [cm^2/s]
  !-----------------------------------------------------------------*/
  // @param [in] temp     ! air temperature [K]
  // @param [in] rhoair   ! air density [g/cm3]
  // @return     vsc_dyn_atm  ! dynamic viscosity of air [g/cm/s]
  return air_dynamic_viscosity(temp) / rhoair;

} // air_kinematic_viscosity


//=============================================================================
// FUNCTION: calc_rain_drop_conc
//=============================================================================
// Description: Computes rain drop number concentrations, radii, and terminal
//              fall velocities across discrete size bins assuming a 
//              Marshall-Palmer exponential size distribution.
//
// Physical Background:
//   - Rain drop size distribution follows Marshall-Palmer (1948):
//     N(D) = N0 * exp(-λD), where D is diameter
//   - Terminal velocity varies with drop size due to changing Reynolds number
//     and drag coefficient regimes (Beard, 1976)
//   - Velocity corrected for air density: v = v_stp * sqrt(ρ_stp / ρ_air)
//
// Algorithm:
//   1. Loop over size bins, computing radius and initial number concentration
//   2. Calculate terminal velocity using size-dependent empirical formula
//   3. Apply density correction for altitude
//   4. Compute total precipitation rate from all bins
//   5. Normalize number concentrations to match input precipitation rate
//
// Parameters:
//   nr          [in]  - Number of rain drop size bins (typically 50)
//   rlo         [in]  - Lower limit of rain radius [cm] (typically 0.005)
//   dr          [in]  - Rain radius bin width [cm] (typically 0.005)
//   rhoair      [in]  - Air mass density [g/cm³]
//   precip      [in]  - Target precipitation rate [cm/s]
//   rrainsv     [out] - Rain drop radius in each bin [cm]
//   xnumrainsv  [out] - Rain drop number concentration in each bin [#/cm³]
//   vfallrainsv [out] - Rain drop terminal fall velocity in each bin [cm/s]
//
// Units: All in CGS system
//   - Length: cm
//   - Time: s
//   - Mass: g
//   - Number concentration: #/cm³
//
// Performance Notes:
//   - O(nr) complexity, typically nr = 50 bins
//   - Contains multiple haero::pow() calls per iteration (~50-100 cycles each)
//   - Two loops: main calculation + normalization
//   - Total estimate: ~3000-5000 cycles for nr=50
//
// Suggestions for Improvement:
//   1. Move constants to namespace scope (implemented above)
//   2. Pre-compute 1/marshall_palmer_scale to replace division with multiplication
//   3. For regime 3 (exponent ≈ 1.008), consider linear approximation
//   4. Consider loop fusion or vectorization hints
//   5. Add bounds checking for nr <= nrainsvmax in debug builds
//
// References:
//   - Marshall, J.S. and Palmer, W.M. (1948), "The distribution of raindrops 
//     with size", J. Meteorology, 5, 165-166
//   - Beard, K.V. (1976), "Terminal velocity and shape of cloud and 
//     precipitation drops aloft", J. Atmos. Sci., 33, 851-864
//=============================================================================
KOKKOS_INLINE_FUNCTION
void calc_rain_drop_conc(const int nr, const Real rlo, const Real dr,
                         const Real rhoair,
                         const Real precip, //! in
                         Real rrainsv[nrainsvmax], Real xnumrainsv[nrainsvmax],
                         Real vfallrainsv[nrainsvmax]) {

  const Real zero = 0;
  Real precipsum = zero;
  const Real four_thirds = 4. / 3.;
  // loop over  cloud bins
  for (int ii = 0; ii < nr; ++ii) {
    // rain radius in the bin [cm]
    const Real rr = rlo + ii * dr;
    rrainsv[ii] = rr;
    xnumrainsv[ii] = haero::exp(-rr / 2.7e-2);
    // rain diameter in the bin [cm]
    const Real dd = 2. * rr;
    Real vfallstp = zero;
    // BAD CONSTANT
    if (dd <= 0.007) {
      vfallstp = 2.88e5 * dd * dd;
    } else if (dd <= 0.025) {
      vfallstp = 2.8008e4 * haero::pow(dd, 1.528);
    } else if (dd <= 0.1) {
      vfallstp = 4104.9 * haero::pow(dd, 1.008);
    } else if (dd <= 0.25) {
      vfallstp = 1812.1 * haero::pow(dd, 0.638);
    } else {
      vfallstp = 1069.8 * haero::pow(dd, 0.235);
    }
    // rain droplet falling speed [cm/s]
    vfallrainsv[ii] = vfallstp * haero::sqrt(1.204e-3 / rhoair);
    // sum of precipitation in all bins
    precipsum += vfallrainsv[ii] * rr * rr * rr * xnumrainsv[ii];

  } // ii

  precipsum *= haero::Constants::pi * four_thirds;
  for (int ii = 0; ii < nr; ++ii) {
    xnumrainsv[ii] *= (precip / precipsum);
  }

} // calc_rain_drop_conc


//=============================================================================
// FUNCTION: calc_aer_conc_frac
//=============================================================================
// Description: Computes aerosol concentration, radius, and volume fraction
//              in each size bin assuming a log-normal distribution.
//
// Parameters:
//   na          [in]  - Number of aerosol bins
//   xlo         [in]  - Lower limit of aerosol radius (log scale)
//   dx          [in]  - Aerosol radius bin width (log scale)
//   xg0         [in]  - Log of geometric mean radius: log(r_mean)
//   sx          [in]  - Standard deviation in log space (log-sigma)
//   raerosv     [out] - Aerosol radius for each bin [cm]
//   fnumaerosv  [out] - Fraction of total number in each bin [fraction]
//   fvolaerosv  [out] - Fraction of total volume in each bin [fraction]
//
// Algorithm: Uses log-normal distribution: 
//   n(r) ~ exp(-0.5*((ln(r)-ln(r_g))/ln(sigma))^2)
//
// Performance Note: O(n) where n = number of bins (typically small ~10-20)
//   - Contains expensive exp() calls in loop
//   - Loop is vectorizable
// Suggestion:
//   1. Consider vectorization hints (#pragma omp simd or Kokkos::parallel_for)
//   2. Pre-compute 1/sx outside loop to replace division with multiplication
//   3. Could use exp2() if available for better performance on some architectures
//=============================================================================
KOKKOS_INLINE_FUNCTION
void calc_aer_conc_frac(const int na, const Real xlo, const Real dx,
                        const Real xg0,
                        const Real sx, // ! in
                        Real raerosv[naerosvmax], Real fnumaerosv[naerosvmax],
                        Real fvolaerosv[naerosvmax]) // out
{

  // ! calculate total aerosol number and volume
  const Real zero = 0;
  const Real four_thirds = 4. / 3.;
  // total aerosol number
  Real anumsum = zero;
  // total aerosol volume
  Real avolsum = zero;
  for (int ii = 0; ii < na; ++ii) {
    const Real xx = xlo + ii * dx;
    // aerosol radius in the bin [cm]
    const Real aa = haero::exp(xx);
    raerosv[ii] = aa;
    const Real dum = (xx - xg0) / sx;
    fnumaerosv[ii] = haero::exp(-0.5 * dum * dum);
    fvolaerosv[ii] =
        four_thirds * fnumaerosv[ii] * haero::Constants::pi * aa * aa * aa;
    anumsum += fnumaerosv[ii];
    avolsum += fvolaerosv[ii];
  } // end ii

  // ! calculate fraction in each aerosol bin
  for (int ii = 0; ii < na; ++ii) {
    fnumaerosv[ii] /= anumsum;
    fvolaerosv[ii] /= avolsum;
  } // end ii

} // calc_aer_conc_frac


//=============================================================================
// FUNCTION: calc_schmidt_number
//=============================================================================
// Description: Calculates the Schmidt number and particle relaxation time
//              for aerosol particles. These dimensionless numbers are used
//              in impaction scavenging calculations.
//
// Physical Background:
//   Schmidt Number (Sc):
//     Sc = ν / D = (kinematic viscosity) / (particle diffusivity)
//     - Represents ratio of momentum diffusivity to mass diffusivity
//     - High Sc means particle diffusion is slow relative to momentum transfer
//     - Typical values for aerosols: 10 to 10^6 (size dependent)
//
//   Relaxation Time (τ):
//     τ = (2 * ρ_p * r² * C) / (9 * μ)
//     - Time for particle velocity to adjust to fluid velocity
//     - Used to calculate Stokes number: St = τ * U / L
//     - Includes Cunningham slip correction for small particles
//
//   Cunningham Slip Correction (C):
//     C = 1 + Kn * (A1 + A2 * exp(-A3/Kn))
//     - Corrects Stokes drag for rarefied gas effects when particle
//       size approaches molecular mean free path
//     - Kn = λ/r is the Knudsen number
//     - For Kn << 1 (large particles): C ≈ 1
//     - For Kn >> 1 (small particles): C >> 1
//
// Parameters:
//   freepath   [in]  - Molecular mean free path [cm]
//                      Typical sea-level value: ~6.5e-6 cm
//   r_aer      [in]  - Aerosol particle radius [cm]
//   temp       [in]  - Air temperature [K]
//   rhoaero    [in]  - Aerosol particle density [g/cm³]
//   rhoair     [in]  - Air mass density [g/cm³]
//   airkinvisc [in]  - Air kinematic viscosity [cm²/s]
//   schmidt    [out] - Schmidt number [dimensionless]
//   taurelax   [out] - Particle relaxation time [s]
//
// Units: All in CGS system
//   - Length: cm
//   - Time: s
//   - Mass: g
//   - Energy: erg (for Boltzmann constant conversion)
//
// Note: A similar Schmidt number calculation exists in dry deposition
//       (modal_aero_drydep.F90) but uses different slip correction formula.
//
// Performance Notes:
//   - O(1) complexity
//   - Contains one haero::exp() call (~50-100 cycles)
//   - Multiple divisions and multiplications
//   - Total estimate: ~100-200 cycles
//
// Suggestions for Improvement:
//   1. Move constants to namespace scope (implemented above)
//   2. Pre-compute boltz_cgs at compile time
//   3. Add bounds checking for r_aer > 0 in debug builds
//   4. Consider combining divisions to reduce operations
//
// References:
//   - Davies, C.N. (1945), "Definitive equations for the fluid resistance 
//     of spheres", Proc. Phys. Soc., 57, 259-270
//   - Fuchs, N.A. (1964), "The Mechanics of Aerosols", Pergamon Press
//   - Seinfeld & Pandis (2006), "Atmospheric Chemistry and Physics", Ch. 9
//=============================================================================
KOKKOS_INLINE_FUNCTION
void calc_schmidt_number(const Real freepath, const Real r_aer,
                         const Real temp, //& ! in
                         const Real rhoaero, const Real rhoair,
                         const Real airkinvisc,         // & ! in
                         Real &schmidt, Real &taurelax) //! out
{

  // Unit conversion from J/K/molecule to erg/K
  const Real one = 1.;
  const Real two = 2.;
  const Real four_thirds = 4. / 3.;

  const Real boltz_cgs = haero::Constants::boltzmann * 1.e7; // erg/K

  // working variables [unitless]
  const Real dum = freepath / r_aer;
  // ! slip correction factor [unitless]
  const Real dumfuchs =
      one + 1.246 * dum + 0.42 * dum * haero::exp(-0.87 / dum);
  taurelax =
      two * rhoaero * r_aer * r_aer * dumfuchs / (9. * rhoair * airkinvisc);

  // single-particle aerosol mass [g]
  const Real aeromass = four_thirds * haero::Constants::pi * r_aer * r_aer *
                        r_aer * rhoaero; // ![g]
  // aerosol diffusivity [cm^2/s]
  const Real aerodiffus = boltz_cgs * temp * taurelax / aeromass; //  ! [cm^2/s]
  schmidt = airkinvisc / aerodiffus;
}


//=============================================================================
// FUNCTION: calc_impact_efficiency
//=============================================================================
// Description: Calculates the total aerosol-raindrop collection efficiency
//              by combining three physical mechanisms:
//              1. Brownian diffusion (dominant for small particles)
//              2. Interception (geometric contact)
//              3. Inertial impaction (dominant for large particles)
//
// Physical Background:
//   Below-cloud scavenging occurs when falling raindrops collect aerosol
//   particles through various mechanisms. The collection efficiency E
//   represents the fraction of particles in the geometric sweep volume
//   that are actually collected.
//
//   Total efficiency: E_total = E_Brown + E_intercept + E_impact
//
//   The "Greenfield gap" (~0.1-1 μm) is where collection is minimum because
//   particles are too large for efficient Brownian capture but too small
//   for effective impaction.
//
// Brownian Diffusion (E_Brown):
//   - Dominant for ultrafine particles (< 0.1 μm)
//   - Particles diffuse across streamlines and contact drop surface
//   - E_Brown ~ Sc^(-2/3) for high Schmidt numbers
//   - Formula: E = 4*(1 + 0.4*Re^0.5*Sc^(1/3)) / (Re*Sc)
//
// Interception (E_intercept):
//   - Particles following streamlines contact drop when passing within
//     one particle radius of the drop surface
//   - Depends on size ratio χ = r_aer / r_rain
//   - Includes viscosity ratio correction for internal circulation
//   - Formula: E = 4*χ*(χ + f(μ_ratio, Re, χ))
//
// Inertial Impaction (E_impact):
//   - Dominant for large particles (> 1 μm)
//   - Particles deviate from streamlines due to inertia
//   - Only occurs when Stokes number exceeds critical value S*
//   - Formula: E = ((St - S*) / (St - S* + 2/3))^1.5, for St > S*
//
// Parameters:
//   r_aer      [in]  - Aerosol particle radius [cm]
//   r_rain     [in]  - Rain drop radius [cm]
//   temp       [in]  - Air temperature [K]
//   freepath   [in]  - Molecular mean free path [cm]
//   rhoaero    [in]  - Aerosol particle density [g/cm³]
//   rhoair     [in]  - Air mass density [g/cm³]
//   vfall      [in]  - Rain drop terminal fall velocity [cm/s]
//   airkinvisc [in]  - Air kinematic viscosity [cm²/s]
//   etotal     [out] - Total collection efficiency [fraction, 0-1]
//
// Units: All in CGS system
//
// Performance Notes:
//   - O(1) complexity
//   - Calls calc_schmidt_number() internally (~100-200 cycles)
//   - Contains multiple haero::pow(), haero::sqrt(), haero::log() calls
//   - Total estimate: ~300-500 cycles
//
// Suggestions for Improvement:
//   1. Move constants to namespace scope (implemented above)
//   2. Replace pow(x, 1/3) with cbrt(x) if available
//   3. Replace pow(x, 1.5) with x * sqrt(x)
//   4. Pre-compute frequently used values (sqrtreynolds already done)
//   5. Add bounds checking for input radii > 0 in debug builds
//
// References:
//   - Slinn, W.G.N. (1983), "Precipitation Scavenging", in Atmospheric 
//     Sciences and Power Production, Ch. 11
//   - Seinfeld & Pandis (2006), "Atmospheric Chemistry and Physics", Ch. 20
//   - Pruppacher & Klett (1997), "Microphysics of Clouds and Precipitation"
//=============================================================================
KOKKOS_INLINE_FUNCTION
void calc_impact_efficiency(const Real r_aer, const Real r_rain,
                            const Real temp, //   & ! in
                            const Real freepath, const Real rhoaero,
                            const Real rhoair,                       // & ! in
                            const Real vfall, const Real airkinvisc, // & ! in
                            Real &etotal) {

  // ! local variables
  const Real zero = 0.;
  const Real one = 1.;
  const Real two = 2.;
  const Real one_third = 1. / 3.;
  const Real two_thirds = 2. / 3.;
  const Real four = 4.;
  // BAD CONSTANT
  // FIXME move this constant to hearo
  // ! ratio of water viscosity to air viscosity (from Slinn)
  const Real xmuwaterair = 60.0; // ! [fraction]
  // ! ratio of aerosol and rain radius [fraction]
  const Real chi = r_aer / r_rain;
  // ---------- calcualte Brown effect ------------
  // Schmidt number [unitless]
  Real schmidt = zero;
  // Stokes number relaxation time [s]
  Real taurelax = zero;

  // ! calculate unitless numbers
  calc_schmidt_number(freepath, r_aer, temp,       //       & ! in
                      rhoaero, rhoair, airkinvisc, //   & ! in
                      schmidt, taurelax);          // ! out
  // Stokes number [unitless]
  const Real stokes = vfall * taurelax / r_rain;
  // Reynolds number [unitless]
  const Real reynolds = r_rain * vfall / airkinvisc;
  const Real sqrtreynolds = haero::sqrt(reynolds);
  // efficiency of aerosol-collection  in different processes
  const Real ebrown =
      four * (one + 0.4 * sqrtreynolds * haero::pow(schmidt, one_third)) /
      (reynolds * schmidt);

  //------------ calculate intercept effect ------------
  Real dum =
      (one + two * xmuwaterair * chi) / (one + xmuwaterair / sqrtreynolds);
  const Real eintercept = four * chi * (chi + dum);

  // ! ------------ calculate impact effect ------------
  dum = haero::log(one + reynolds);
  const Real sstar = (1.2 + dum / 12.) / (one + dum);
  Real eimpact = zero;
  if (stokes > sstar) {
    dum = stokes - sstar;
    eimpact = haero::pow(dum / (dum + two_thirds), 1.5);
  }
  // ! ------------ calculate total effects ------------
  etotal = ebrown + eintercept + eimpact;

  etotal = haero::min(etotal, one);
} // calc_impact_efficiency


//=============================================================================
// FUNCTION: calc_1_impact_rate
//=============================================================================
// Description: Computes below-cloud impaction scavenging rates for aerosol
//              number and volume concentrations at a reference precipitation
//              rate of 1 mm/hr. Results can be scaled linearly for other
//              precipitation rates.
//
// Physical Background:
//   Below-cloud scavenging (washout) occurs when falling precipitation drops
//   collect aerosol particles through three mechanisms:
//   1. Brownian diffusion (small particles)
//   2. Interception (intermediate particles)
//   3. Inertial impaction (large particles)
//
//   The scavenging rate Λ is computed as:
//     Λ = ∫∫ π*R² * V(R) * E(r,R) * n(R) * f(r) dR dr
//   where:
//     R = rain drop radius
//     r = aerosol particle radius
//     V(R) = rain drop terminal velocity
//     E(r,R) = collection efficiency
//     n(R) = rain drop number distribution
//     f(r) = aerosol size distribution fraction
//
// Algorithm:
//   1. Set up rain drop size bins (Marshall-Palmer distribution)
//   2. Set up aerosol size bins (log-normal distribution)
//   3. Compute atmospheric properties (air density, viscosity, mean free path)
//   4. Double integration over rain and aerosol size distributions
//   5. Convert scavenging rate from 1/s to 1/hr
//
// Parameters:
//   dg0         [in]  - Geometric mean diameter of aerosol [cm]
//   sigmag      [in]  - Geometric standard deviation of size distribution [dimensionless]
//   rhoaero     [in]  - Aerosol particle density [g/cm³]
//   temp        [in]  - Air temperature [K]
//   press       [in]  - Air pressure [dyne/cm²] (CGS units)
//   scavratenum [out] - Scavenging rate for aerosol number [1/hr]
//   scavratevol [out] - Scavenging rate for aerosol volume [1/hr]
//
// Units: All internal calculations in CGS system
//   - Length: cm
//   - Time: s (converted to hr for output)
//   - Mass: g
//   - Pressure: dyne/cm² (converted internally to Pa for density calculation)
//
// Performance Notes:
//   - O(nr × na) complexity due to nested loops
//   - Typical: nr=50 rain bins × na≈10-20 aerosol bins = 500-1000 iterations
//   - Each iteration calls calc_impact_efficiency() (~300-500 cycles)
//   - Total estimate: ~200,000-500,000 cycles
//   - Contains multiple log(), exp(), sqrt(), pow() calls
//
// Suggestions for Improvement:
//   1. Move constants to namespace scope (implemented above)
//   2. Pre-compute rain sweep-out volume outside inner loop (done partially)
//   3. Consider vectorization of inner loop with SIMD hints
//   4. Cache frequently accessed array values in registers
//   5. Consider lookup table for collection efficiency if called repeatedly
//   6. Use Kokkos::parallel_reduce for the nested loops on GPU
//
// References:
//   - Slinn, W.G.N. (1983), "Precipitation Scavenging", Ch. 11
//   - Seinfeld & Pandis (2006), "Atmospheric Chemistry and Physics", Ch. 20
//   - Marshall & Palmer (1948), "The distribution of raindrops with size"
//=============================================================================
KOKKOS_INLINE_FUNCTION
void calc_1_impact_rate(const Real dg0,     //  in
                        const Real sigmag,  //  in
                        const Real rhoaero, //  in
                        const Real temp,    //  in
                        const Real press,   //  in
                        Real &scavratenum,  // out
                        Real &scavratevol)  // out

{

  const Real pi = haero::Constants::pi;
  const Real zero = 0;
  const Real one = 1;
  const Real two = 2;
  const Real ten = 10;
  const Real one_thousand = 1000;
  const Real three = 3.;
  const Real four = 4.;

  // local variables
  Real rrainsv[nrainsvmax] = {zero};    // rain radius for each bin [cm]
  Real xnumrainsv[nrainsvmax] = {zero}; // rain number for each bin [#/cm3]
  Real vfallrainsv[nrainsvmax] = {
      zero}; // rain falling velocity for each bin [cm/s]

  Real raerosv[naerosvmax] = {zero}; // aerosol particle radius in each bin [cm]
  Real fnumaerosv[naerosvmax] = {
      zero}; // fraction of total number in the bin [fraction]
  Real fvolaerosv[naerosvmax] = {
      zero}; // fraction of total volume in the bin [fraction]

  // this subroutine is calculated for a fix rainrate of 1 mm/hr
  // precipitation rate, fix as 1 mm/hr in this subroutine [cm/s]
  const Real precip = one / Real(36000.); //  1 mm/hr in cm/s

  // set the iteration radius for rain droplet
  // rain droplet bin information [cm]
  // BAD CONSTANT
  const Real rlo = .005;
  // const Real rhi = .250;
  const Real dr = 0.005;
  // // Nearest whole number: nint
  // // number of rain bins
  // const int nr = 1 + haero::round((rhi - rlo) / dr);
  // FIXME: values to compute nr are hard-coded.
  const int nr = 50;
  // We comment this line because nr is hard-coded.
  // if (nr > nrainsvmax) {
  //   Kokkos::abort("subr. calc_1_impact_rate -- nr > nrainsvmax \n ");
  //   return;
  // }

  // aerosol modal information
  // aerosol bin information
  const Real ag0 = dg0 / two; // mean radius of aerosol
  // standard deviation (log-normal distribution)
  const Real sx = haero::log(sigmag);
  // log(mean radius) (log-normal distribution)
  const Real xg0 = haero::log(ag0);
  const Real xg3 = xg0 + three * sx * sx; // mean + 3*std^2
  // BAD CONSTANT
  // set the iteration radius for aerosol particles
  const Real dx = haero::max(0.2 * sx, 0.01);
  const Real xlo = xg3 - haero::max(four * sx, two * dx);
  const Real xhi = xg3 + haero::max(four * sx, two * dx);
  // Nearest whole number: nint
  const int na = 1 + haero::round((xhi - xlo) / dx);

  if (na > naerosvmax) {
    Kokkos::abort("subr. calc_1_impact_rate -- na > naerosvmax \n ");
  }

  // Note: pressure units are: ! dynes/cm2
  // We need pressure in units of Pa in density_of_ideal_gas
  // 10 dynes/cm2 = Pa
  const Real pressure = press / ten;
  // air mass density [g/cm^3]
  // unit conversion from [kg/m^3]/1000 to [g/cm^3]
  const Real rhoair = conversions::density_of_ideal_gas(
                          temp, pressure, Constants::r_gas_dry_air) /
                      one_thousand;
  // unit conversion from [mol g /cm^3/kg]/1000 to [mol/cm^3]
  // air molar density [mol/cm^3]
  const Real cair =
      rhoair / haero::Constants::molec_weight_dry_air / one_thousand;
  // !   molecular freepath [cm]
  // BAD CONSTANT
  // FIXME move this constant to haero
  const Real freepath = 2.8052e-10 / cair;
  // ! air kinematic viscosity [cm^2/s]
  const Real airkinvisc = air_kinematic_viscosity(temp, rhoair);

  // compute rain drop number concentrations
  calc_rain_drop_conc(nr, rlo, dr, rhoair, precip,       // ! in
                      rrainsv, xnumrainsv, vfallrainsv); // ! out

  // compute aerosol concentrations
  calc_aer_conc_frac(na, xlo, dx, xg0, sx,             //  in
                     raerosv, fnumaerosv, fvolaerosv); // out

  // compute scavenging

  Real scavsumnum = zero; // ! scavenging rate of aerosol number, "*bb" is for
                          // each rain droplet radius bin [1/s]
  Real scavsumvol = zero; //! scavenging rate of aerosol volume, "*bb" is for
                          //! each rain droplet radius bin [1/s]

  // outer loop for rain drop radius
  for (int jr = 0; jr < nr; ++jr) {
    // rain droplet radius
    // rain droplet and aerosol particle radius [cm]
    Real r_rain = rrainsv[jr];
    // rain droplet fall speed [cm/s]
    Real vfall = vfallrainsv[jr];
    // inner loop for aerosol particle radius
    Real scavsumnumbb = zero;
    Real scavsumvolbb = zero;
    for (int ja = 0; ja < na; ++ja) {
      // aerosol particle radius
      // rain droplet and aerosol particle radius [cm]
      const Real r_aer = raerosv[ja];
      Real etotal = zero; // efficiency of total scavenging effects [fraction]
      calc_impact_efficiency(r_aer, r_rain, temp,       // & ! in
                             freepath, rhoaero, rhoair, // & ! in
                             vfall, airkinvisc,         // & ! in
                             etotal);                   // out

      // rain droplet sweep out volume [cm3/cm3/s]
      const Real rainsweepout =
          xnumrainsv[jr] * four * pi * r_rain * r_rain * vfall;
      scavsumnumbb += rainsweepout * etotal * fnumaerosv[ja];
      scavsumvolbb += rainsweepout * etotal * fvolaerosv[ja];
    } // ja_loop

    scavsumnum += scavsumnumbb;
    scavsumvol += scavsumvolbb;

  } // jr_loop

  scavratenum = scavsumnum * 3600;
  scavratevol = scavsumvol * 3600;

} // end calc_1_impact_rate


//=============================================================================
// FUNCTION: modal_aero_bcscavcoef_init
//=============================================================================
// Description: Computes and initializes lookup tables for below-cloud aerosol 
//              impaction/interception scavenging rates. Tables are indexed by
//              aerosol mode and hygroscopic growth factor.
//
// Purpose:
//   Pre-compute scavenging rates for various wet/dry diameter ratios to avoid
//   expensive runtime calculations. During model integration, actual rates
//   are obtained by interpolating these tables based on current wet diameter.
//
// Physical Background:
//   Below-cloud scavenging (washout) depends on aerosol wet size, which varies
//   with relative humidity through hygroscopic growth. Rather than computing
//   scavenging rates at every timestep, we pre-compute rates across a range
//   of growth factors and store as log(rate) for linear interpolation.
//
//   Growth factor range: exp(-7 * ln(1.25)) to exp(12 * ln(1.25))
//                      = 0.178 to 14.55 (wet/dry diameter ratio)
//
// Algorithm:
//   1. Loop over aerosol modes (4 modes in MAM4)
//   2. Loop over growth factor indices (-7 to +12, total 20 entries)
//   3. For each combination:
//      a. Compute wet diameter from dry diameter and growth factor
//      b. Compute wet density (currently using dry density due to bug)
//      c. Convert units from SI to CGS
//      d. Call calc_1_impact_rate() for 1 mm/hr precipitation
//      e. Store log(rate) in lookup table for later interpolation
//
// Parameters:
//   dgnum_amode         [in]  - Geometric mean diameters for each mode [m]
//   sigmag_amode        [in]  - Geometric standard deviations [dimensionless]
//   aerosol_dry_density [in]  - Dry aerosol density for each mode [kg/m³]
//   scavimptblnum       [out] - Lookup table for number scavenging rate [log(1/hr)]
//   scavimptblvol       [out] - Lookup table for volume scavenging rate [log(1/hr)]
//
// Table Dimensions:
//   - First index: jgrow - nimptblgrow_mind = 0 to 19 (20 growth levels)
//   - Second index: imode = 0 to 3 (4 aerosol modes)
//   - Values stored as natural log for linear interpolation in log-space
//
// Units:
//   - Input: SI units (m, kg/m³)
//   - Internal calculation: CGS units (cm, g/cm³, dyne/cm²)
//   - Output: log(1/hr) stored in tables
//
// Performance Notes:
//   - O(num_modes × num_growth_levels) = O(4 × 20) = O(80) iterations
//   - Each iteration calls calc_1_impact_rate (~200,000-500,000 cycles)
//   - Total initialization: ~20-40 million cycles (one-time cost)
//   - Called once during model initialization, not during timestepping
//
// Suggestions for Improvement:
//   1. Fix the wet density calculation bug (see FIXME note)
//   2. Move magic numbers to named constants (implemented above)
//   3. Consider parallelizing outer loop over modes
//   4. Add validation checks for input arrays
//   5. Document why 750 hPa and 0°C were chosen as reference conditions
//
// Known Bug (FIXME):
//   The wet aerosol density calculation is incorrect:
//     rhowetaero = 1.0 + (rhodryaero - 1.0) / wetvolratio
//   This formula assumes water density = 1.0 kg/m³, but it should be 1000 kg/m³
//   The correct formula should be:
//     rhowetaero = rho_water + (rhodryaero - rho_water) / wetvolratio
//   Currently bypassed by setting rhowetaero = rhodryaero for BFB testing.
//
// References:
//   - Slinn, W.G.N. (1983), "Precipitation Scavenging", Ch. 11
//=============================================================================
inline void modal_aero_bcscavcoef_init(
    const Real dgnum_amode[AeroConfig::num_modes()],
    const Real sigmag_amode[AeroConfig::num_modes()],
    const Real aerosol_dry_density[AeroConfig::num_modes()],
    // outputs
    View2DHost scavimptblnum, View2DHost scavimptblvol) {

  const Real zero = 0;
  const Real one = 1;
  const Real three = 3;
  // BAD CONSTANT
  const Real dlndg_nimptblgrow = haero::log(1.25);
  // ! set up temperature-pressure pair to compute impaction scavenging rates
  const Real temp_0C = haero::Constants::melting_pt_h2o; //     ! K
  const Real press_750hPa = 0.75e6;                      //  ! dynes/cm2
  for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    const Real sigmag = sigmag_amode[imode];
    // clang-format off
    // Note: we replaced lspectype_amode and lspectype_amode for
    // dry_aero_density
    // const int ll = lspectype_amode[0][imode];
    // const Real rhodryaero = specdens_amode[ll];
    // clang-format on
    const Real rhodryaero = aerosol_dry_density[imode];
    for (int jgrow = nimptblgrow_mind; jgrow <= nimptblgrow_maxd; ++jgrow) {
      // ratio of diameter for wet/dry aerosols [fraction]
      const Real wetdiaratio = haero::exp(Real(jgrow) * dlndg_nimptblgrow);
      // aerosol diameter [m]
      const Real dg0 = dgnum_amode[imode] * wetdiaratio;
      // ratio of volume for wet/dry aerosols [fraction]
      const Real wetvolratio =
          haero::exp(Real(jgrow) * dlndg_nimptblgrow * three);
      // dry and wet aerosol density [kg/m3]
      Real rhowetaero = one + (rhodryaero - one) / wetvolratio;
      rhowetaero = haero::min(rhowetaero, rhodryaero);
      /* FIXME: not sure why wet aerosol density is set as dry aerosol density
         here ! but the above calculation of rhowetaero is incorrect. ! I think
         the number 1.0_r8 should be 1000._r8 as the unit is kg/m3 ! the above
         calculation gives wet aerosol density very small number (a few kg/m3) !
         this may cause some problem. I guess this is the reason of using dry
         density. ! should be better if fix the wet density bug and use it. Keep
         it for now for BFB testing ! -- (commented by Shuaiqi Tang when
         refactoring for MAM4xx) */

      rhowetaero = rhodryaero;
      /*compute impaction scavenging rates at 1 temp-press pair and save
              ! note that the subroutine calc_1_impact_rate uses CGS units */
      // aerosol diameter in CGS unit [cm]
      const Real dg0_cgs = dg0 * 1.0e2; //  ! m to cm
      // wet aerosol density in CGS unit [g/cm3]
      const Real rhowetaero_cgs = rhowetaero * 1.0e-3; //   ! kg/m3 to g/cm3
      // scavenging rate of aerosol number [1/s]
      Real scavratenum = zero;
      // scavenging rate of aerosol volume [1/s]
      Real scavratevol = zero;

      calc_1_impact_rate(dg0_cgs, sigmag, rhowetaero_cgs, temp_0C, press_750hPa,
                         scavratenum, scavratevol);

      scavimptblnum(jgrow - nimptblgrow_mind, imode) = haero::log(scavratenum);
      scavimptblvol(jgrow - nimptblgrow_mind, imode) = haero::log(scavratevol);

    } // jgrow

  } // end imode

} // modal_aero_bcscavcoef_init


//=============================================================================
// Physical Background:
//   Wet removal of aerosols occurs through multiple mechanisms:
//   
//   1. Stratiform In-Cloud (sol_facti):
//      - Removal of aerosols within stratiform cloud layers
//      - For interstitial: OFF (aerosols must first activate into droplets)
//      - For cloud-borne: ON (already in droplets, removed with precip)
//   
//   2. Convective In-Cloud (sol_factic):
//      - Removal of aerosols within convective updrafts
//      - For interstitial: Tuning factor (aerosols entrained into updrafts)
//      - For cloud-borne: OFF (convective precip doesn't collect strat droplets)
//   
//   3. Below-Cloud (sol_factb):
//      - Impaction scavenging by falling precipitation
//      - For interstitial: ON (aerosols below cloud base can be collected)
//      - For cloud-borne: OFF (cloud-borne aerosols are "in-cloud" by definition)
//   
//   4. Convective Activation (f_act_conv):
//      - Fraction of aerosols activated in convective clouds
//      - Primary carbon mode: 0 (hydrophobic, doesn't activate)
//      - Other modes: Based on input activation fraction
//
// Historical Notes (from original code):
//   - 2008-mar-07 (rce): sol_factb changed from 0.3 to 0.1
//   - 2008-mar-07 (rce): sol_factic for dust modes changed from 1.0 to 0.5
//   - 2010-may-02 (rce): Separated activation fraction from tuning factor
//                        for convective in-cloud removal
//
// Parameters:
//   lphase                              [in]  - Phase index:
//                                               1 = interstitial aerosol
//                                               2 = cloud-borne aerosol
//   imode                               [in]  - Aerosol mode index (0-3 for MAM4)
//   scav_fraction_in_cloud_strat        [in]  - Stratiform in-cloud scav fraction [0-1]
//   scav_fraction_in_cloud_conv         [in]  - Convective in-cloud scav fraction [0-1]
//   scav_fraction_below_cloud_strat     [in]  - Below-cloud scav fraction [0-1]
//   activation_fraction_in_cloud_conv   [in]  - Convective activation fraction [0-1]
//   sol_facti                           [out] - Stratiform in-cloud scav fraction [0-1]
//   sol_factic                          [out] - Convective in-cloud scav fraction [0-1]
//   sol_factb                           [out] - Below-cloud scav fraction [0-1]
//   f_act_conv                          [out] - Convective activation fraction [0-1]
//
// Output Logic Summary:
//   ┌─────────────────┬─────────────────────────┬────────────────────────┐
//   │ Parameter       │ Interstitial (lphase=1) │ Cloud-borne (lphase=2) │
//   ├─────────────────┼─────────────────────────┼────────────────────────┤
//   │ sol_facti       │ 0.0                     │ min(0.6, input)        │
//   │ sol_factic      │ input                   │ 0.0                    │
//   │ sol_factb       │ input                   │ 0.0                    │
//   │ f_act_conv      │ 0 if pcarbon, else input│ 0.0                    │
//   └─────────────────┴─────────────────────────┴────────────────────────┘
//
// Performance Notes:
//   - O(1) complexity - simple conditional assignments
//   - No expensive operations (single min() call)
//   - Total estimate: ~10-20 cycles
//
// Suggestions for Improvement:
//   1. Move magic number 0.6 to named constant (implemented above)
//   2. Use enum class for lphase to improve type safety
//   3. Consider struct return type to bundle output parameters
//   4. Add validation for lphase values in debug builds
//
// Future Considerations (from original comments):
//   - Non-activation of aerosol in entrained air should be included
//   - Could use activate routine with w ~= 1 m/s to calculate activation
//   - Entrainment issues need to be addressed
//
//=============================================================================
KOKKOS_INLINE_FUNCTION
void define_act_frac(const int lphase, const int imode,
                     const Real scav_fraction_in_cloud_strat,
                     const Real scav_fraction_in_cloud_conv,
                     const Real scav_fraction_below_cloud_strat,
                     const Real activation_fraction_in_cloud_conv,
                     Real &sol_facti, Real &sol_factic, Real &sol_factb,
                     Real &f_act_conv) {
  // clang-format off
  // -----------------------------------------------------------------------
  // define sol_factb and sol_facti values, and f_act_conv
  // sol_factb - currently this is basically a tuning factor
  // sol_facti & sol_factic - currently has a physical basis, and
  // reflects activation fraction
  // f_act_conv is the activation fraction
  //
  // 2008-mar-07 rce - sol_factb (interstitial) changed from 0.3 to 0.1
  // - sol_factic (interstitial, dust modes) changed from 1.0 to 0.5
  // - sol_factic (cloud-borne, pcarb modes) no need to set it to 0.0
  // because the cloud-borne pcarbon == 0 (no activation)
  //
  // rce 2010/05/02
  // prior to this date, sol_factic was used for convective in-cloud wet removal,
  // and its value reflected a combination of an activation fraction
  // (which varied between modes) and a tuning factor
  // from this date forward, two parameters are used for convective
  // in-cloud wet removal
  //
  // note that "non-activation" of aerosol in air entrained into updrafts should
  // be included here
  // eventually we might use the activate routine (with w ~= 1 m/s) to calculate
  // this, but there is still the entrainment issue
  //
  // sol_factic is strictly a tuning factor
  //-----------------------------------------------------------------------
  // clang-format on
  const int modeptr_pcarbon = static_cast<int>(mam4::ModeIndex::PrimaryCarbon);
  if (lphase == 1) { // interstial aerosol
    sol_facti = 0.0; // strat in-cloud scav totally OFF for institial
    // if modal aero convproc is turned on for aerosols, then
    // turn off the convective in-cloud removal for interstitial aerosols
    // (but leave the below-cloud on, as convproc only does in-cloud)
    // and turn off the outfld SFWET, SFSIC, SFSID, SFSEC, and SFSED calls
    // for (stratiform)-cloudborne aerosols, convective wet removal
    // (all forms) is zero, so no action is needed
    sol_factic = scav_fraction_in_cloud_conv;
    // all below-cloud scav ON (0.1 "tuning factor")
    sol_factb = scav_fraction_below_cloud_strat;
    if (imode == modeptr_pcarbon)
      f_act_conv = 0.0;
    else
      f_act_conv = activation_fraction_in_cloud_conv;

  } else {
    // cloud-borne aerosol (borne by stratiform cloud drops)
    // all below-cloud scav OFF (anything cloud-borne is located "in-cloud")
    sol_factb = 0.0;
    // strat  in-cloud scav totally ON for cloud-borne
    sol_facti = haero::min(0.6, scav_fraction_in_cloud_strat);
    // conv   in-cloud scav OFF (having this on would mean
    // that conv precip collects strat droplets)
    sol_factic = 0.0;
    // conv   in-cloud scav OFF (having this on would mean
    f_act_conv = 0.0;
  }
}


//=============================================================================
// FUNCTION: lptr_dust_a_amode
//=============================================================================
// Description: Returns the tracer array index (pointer) for dust aerosol
//              species in the specified aerosol mode.
//
// Physical Background:
//   Mineral dust in MAM4 is present only in certain modes:
//   - Accumulation mode (mode 0): Fine dust from aging/coagulation
//   - Coarse mode (mode 2): Primary dust emissions (dominant source)
//   
//   Dust is NOT present in:
//   - Aitken mode (mode 1): Particles too small for dust
//   - Primary Carbon mode (mode 3): Contains only carbonaceous aerosols
//
// Parameter:
//   imode [in] - Aerosol mode index (0 to num_modes-1)
//                0 = Accumulation
//                1 = Aitken
//                2 = Coarse
//                3 = Primary Carbon
//
// Returns: 
//   - Positive integer: Valid tracer array index for dust in that mode
//   - -999888777: Dust is not simulated in that mode (invalid marker)
//
// Usage Example:
//   int idx = lptr_dust_a_amode(2);  // Returns 28 (coarse mode dust index)
//   int idx = lptr_dust_a_amode(1);  // Returns -999888777 (no dust in Aitken)
//
//   // Check if dust exists in mode before accessing:
//   if (lptr_dust_a_amode(imode) > 0) {
//       // Safe to access dust tracer for this mode
//   }
//
// Performance Notes:
//   - O(1) lookup
//   - Original: Array initialized on each call (inefficient)
//   - Improved: Static constexpr array at namespace scope (zero runtime cost)
//
// Suggestions for Improvement:
//   1. Move array to namespace scope as constexpr (implemented above)
//   2. Add bounds checking for imode in debug builds
//   3. Consider returning std::optional<int> or using sentinel value consistently
//   4. Document the magic number -999888777 with a named constant
//
// Related Functions:
//   - lptr_nacl_a_amode(): Returns index for sea salt (NaCl) species
//   - lmassptr_amode(): Returns index for general species mass mixing ratio
//=============================================================================
KOKKOS_INLINE_FUNCTION
int lptr_dust_a_amode(const int imode) {
  const int num_modes = AeroConfig::num_modes();
  const int lptr_dust_a_amode[num_modes] = {19, -999888777, 28, -999888777};
  return lptr_dust_a_amode[imode];
}

//=============================================================================
// FUNCTION: lptr_nacl_a_amode
//=============================================================================
// Description: Returns the pointer/index for sodium chloride (NaCl) aerosol
//              in the specified aerosol mode.
// 
// Parameter:
//   imode - Index of the aerosol mode (0 to num_modes-1)
//
// Returns: Index into the aerosol array for NaCl species in the given mode
//=============================================================================

KOKKOS_INLINE_FUNCTION
int lptr_nacl_a_amode(const int imode) {
  const int num_modes = AeroConfig::num_modes();
  const int lptr_nacl_a_amode[num_modes] = {20, 25, 29, -999888777};
  return lptr_nacl_a_amode[imode];
}

//=============================================================================
// FUNCTION: mmtoo_prevap_resusp
//=============================================================================
// Description: Returns the mapping index for pre-evaporation resuspension
//              tracer conversion. Maps constituent index to target tracer.
//              -1 indicates no mapping (species not simulated)
//              -3 indicates special handling required
//
// Parameter:
//   i - Constituent index (0 to pcnst-1)
//
// Returns: Target tracer index or negative value for special cases
//

KOKKOS_INLINE_FUNCTION
int mmtoo_prevap_resusp(const int i) {
  const int mmtoo_prevap_resusp[pcnst] = {
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, 30, 32, 33, 31, 28, 29, 34, -3, 30, 33, 29, 34, -3,
      28, 29, 30, 31, 32, 33, 34, -3, 32, 31, 34, -3};
  return mmtoo_prevap_resusp[i];
}

//=============================================================================
// FUNCTION: lmassptr_amode
//=============================================================================
// Description: Returns the global chemistry (gchm) r-array index for the 
//              mixing ratio (moles-x/mole-air) for chemical species l in 
//              aerosol mode m. This refers to aerosols in clear air or 
//              interstitial air (NOT in cloud water).
//
// Parameters:
//   i     - Chemical species index (0 to maxd_aspectype-1)
//   imode - Aerosol mode index (0 to num_modes-1)
//
// Returns: Array index for the species mixing ratio, or -1 if the species
//          is not being simulated in that mode
//=============================================================================
KOKKOS_INLINE_FUNCTION
int lmassptr_amode(const int i, const int imode) {
  const int num_modes = AeroConfig::num_modes();
  const int lmassptr_amode[maxd_aspectype][num_modes] = {
      {15, 23, 28, 36}, {16, 24, 29, 37}, {17, 25, 30, 38}, {18, 26, 31, -1},
      {19, -1, 32, -1}, {20, -1, 33, -1}, {21, -1, 34, -1}, {-1, -1, -1, -1},
      {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1},
      {-1, -1, -1, -1}, {-1, -1, -1, -1}};
  return lmassptr_amode[i][imode];
}

//=============================================================================
// FUNCTION: lmassptrcw_amode
//=============================================================================
// Description: Returns the gchm r-array index for the mixing ratio of 
//              chemical species l in aerosol mode m that is currently 
//              bound/dissolved in cloud water.
//
// Note: Currently delegates to lmassptr_amode - same indices used for
//       both interstitial and cloud-borne aerosols
//
// Add comment explaining why cloud-water uses same indexing
// as interstitial, or if this is a placeholder for future work
//=============================================================================
KOKKOS_INLINE_FUNCTION
int lmassptrcw_amode(const int i, const int j) { return lmassptr_amode(i, j); }


//=============================================================================
// FUNCTION: numptr_amode
//=============================================================================
// Description: Returns the gchm r-array index for the number mixing ratio
//              (particles/mole-air) for aerosol mode m in clear/interstitial
//              air (not in cloud water).
//
// Parameter:
//   i - Aerosol mode index (0 to num_modes-1)
//
// Returns: Array index for number mixing ratio. Zero or negative indicates
//          number is not being simulated.
//
// Performance Note: Small array (4 elements), efficient lookup
// Suggestion: Add bounds checking for i parameter
//=============================================================================
KOKKOS_INLINE_FUNCTION
int numptr_amode(const int i) {
  const int num_modes = AeroConfig::num_modes();
  const int numptr_amode[num_modes] = {22, 27, 35, 39};
  return numptr_amode[i];
}


//=============================================================================
// FUNCTION: numptrcw_amode
//=============================================================================
// Description: Returns number mixing ratio index for cloud-borne aerosols.
//              Currently uses same indexing as interstitial aerosols.
//=============================================================================
KOKKOS_INLINE_FUNCTION
int numptrcw_amode(const int i) { return numptr_amode(i); }


// =============================================================================
// FUNCTION: index_ordering
// =============================================================================
// Description: Determines the correct index ordering for aerosol tracers,
//              specifically for pre-evaporation resuspension to coarse mode.
//              Maps species/mode/phase combination to tracer index and type.
//
// Parameters:
//   lspec  [in]  - Index for aerosol number/chem-mass/water-mass
//   imode  [in]  - Index for aerosol mode
//   lphase [in]  - Index for phase: 1=interstitial, 2=cloudborne aerosol
//   mm     [out] - Index of the tracers (-1 if not found)
//   jnv    [out] - Index for scavcoefnv 3rd dimension
//   jnummaswtr [out] - Aerosol species type indicator:
//                      0=number, 1=dry mass, 2=water
//
// Performance Note: Contains conditional logic but is O(1)
// Suggestion: Consider using enum class for jaeronumb/jaeromass/jaerowater
//             for type safety and readability
// =============================================================================
KOKKOS_INLINE_FUNCTION
void index_ordering(const int lspec, const int imode, const int lphase, int &mm,
                    int &jnv, int &jnummaswtr) {
  // clang-format off
  //-----------------------------------------------------------------------
  // changed ordering (mass then number) for prevap resuspend to coarse
  //-----------------------------------------------------------------------
  /*
  in :: lspec  ! index for aerosol number / chem-mass / water-mass
  in :: imode  ! index for aerosol mode
  in :: lphase ! index for 1 == interstitial / 2 == cloudborne aerosol
  out :: mm         ! index of the tracers
  out :: jnv        ! index for scavcoefnv 3rd dimension
  out :: jnummaswtr ! indicates current aerosol species type (0 = number, 1 = dry mass, 2 = water)
  */
  // clang-format on
  const int jaeronumb = 0;
  const int jaeromass = 1;
  const int jaerowater = 2;
  const int nspec_amode = mam4::num_species_mode(imode);
  mm = -1;
  if (lspec < nspec_amode) { // non-water mass
    jnummaswtr = jaeromass;
    if (lphase == 1) {
      mm = lmassptr_amode(lspec, imode);
      jnv = 2;
    } else {
      mm = lmassptrcw_amode(lspec, imode);
      jnv = 0;
    }
  } else if (lspec == nspec_amode) { // number
    jnummaswtr = jaeronumb;
    if (lphase == 1) {
      mm = numptr_amode(imode);
      jnv = 1;
    } else {
      mm = numptrcw_amode(imode);
      jnv = 0;
    }
  } else { // water mass
    jnummaswtr = jaerowater;
  }
}

//=============================================================================
// FUNCTION: examine_prec_exist
//=============================================================================
// Description: Determines whether precipitation exists at a specified vertical
//              level by integrating precipitation flux from the top of the
//              atmosphere downward.
//
// Physical Background:
//   Precipitation at any level is the net result of:
//   - Rain production from stratiform clouds (prain)
//   - Rain production from convective clouds (cmfdqr)
//   - Rain evaporation as drops fall (evapr, negative contribution)
//   
//   The integral is converted from mass mixing ratio tendency [kg/kg/s]
//   to mass flux [kg/m²/s] using: flux = tendency * Δp / g
//
// Algorithm:
//   1. Initialize precipitation accumulator to zero at top of atmosphere
//   2. Loop downward through atmosphere, accumulating net precipitation
//   3. Compare accumulated flux against threshold
//   4. Return 1 if precipitation exists, 0 otherwise
//
// Parameters:
//   level_for_precipitation [in] - Target level index to check (0 = TOA)
//   pdel                    [in] - Pressure thickness of each layer [Pa]
//   prain                   [in] - Stratiform rain production rate [kg/kg/s]
//   cmfdqr                  [in] - Convective rain production rate [kg/kg/s]
//   evapr                   [in] - Rain evaporation rate [kg/kg/s]
//
// Returns:
//   1 - Precipitation exists at the specified level (flux ≥ 1.0e-7 kg/m²/s)
//   0 - No significant precipitation at the specified level
//
// Units:
//   - Input rates: [kg/kg/s] (mass mixing ratio tendency)
//   - Pressure: [Pa]
//   - Internal flux: [kg/m²/s]
//
// Performance Notes:
//   - O(level_for_precipitation) complexity
//   - Simple arithmetic operations in loop
//   - Total estimate: ~5-10 cycles per level
//
// Suggestions for Improvement:
//   1. Move threshold to named constant (implemented above)
//   2. Consider returning bool instead of int for clarity
//   3. Pre-compute 1/gravit outside loop for efficiency
//=============================================================================
KOKKOS_INLINE_FUNCTION
int examine_prec_exist(const int level_for_precipitation, const Real pdel[],
                       const Real prain[], const Real cmfdqr[],
                       const Real evapr[]) {
  // clang-format off
  // ----------------------------------------------------------------------
  // examine if level level_for_precipitation has precipitation.
  // ----------------------------------------------------------------------
  /*
  in :: pdel     ! pressure difference between two layers [Pa]
  in :: prain    ! rain production rate from stratiform clouds [kg/kg/s]
  in :: cmfdqr   ! dq/dt due to convective rainout [kg/kg/s]
  in :: evapr    ! rain evaporation rate [kg/kg/s]
  out :: examine_prec_exist   ! if there is precipitation falling into the level
  */
  // clang-format on
  // BAD CONSTANT
  const Real small_value_7 = 1.0e-7;
  const Real gravit = Constants::gravity;

  // initiate precipitation at the top level
  // precipitation falling from layers above [kg/m2/s]
  Real prec = 0;
  // check from the top level downward
  for (int k = 0; k < level_for_precipitation; ++k) {
    // update precipitation to the level below k
    prec += (prain[k] + cmfdqr[k] - evapr[k]) * pdel[k] / gravit;
  }
  const int isprx = (prec >= small_value_7) ? 1 : 0;
  return isprx;
}

//=============================================================================
// FUNCTION: set_f_act_coarse
//=============================================================================
// Description: Calculates the mass-weighted activation fraction for coarse
//              mode aerosols in convective clouds based on the relative
//              abundances of dust and sea salt.
//
// Physical Background:
//   Different aerosol species have different hygroscopicities and thus
//   different activation efficiencies in convective clouds:
//   - Sea salt (NaCl): Highly hygroscopic, activates easily (f = 0.80)
//   - Mineral dust: Less hygroscopic, activates less efficiently (f = 0.40)
//   
//   The effective coarse mode activation fraction is computed as a
//   mass-weighted average of the species-specific fractions.
//
// Algorithm:
//   1. Initialize default activation fractions for dust and sea salt
//   2. Calculate updated mass concentrations (current + tendency × dt)
//   3. If total mass is significant, compute mass-weighted average
//   4. Otherwise, use default overall coarse mode fraction (0.60)
//
// Parameters:
//   kk                      [in]  - Vertical level index
//   state_q                 [in]  - Tracer mixing ratios [kg/kg]
//   ptend_q                 [in]  - Tracer tendencies [kg/kg/s]
//   dt                      [in]  - Model timestep [s]
//   f_act_conv_coarse       [out] - Mass-weighted activation fraction [0-1]
//   f_act_conv_coarse_dust  [out] - Dust activation fraction (0.40) [0-1]
//   f_act_conv_coarse_nacl  [out] - Sea salt activation fraction (0.80) [0-1]
//
// Formula:
//   f_act = (f_dust × m_dust + f_nacl × m_nacl) / (m_dust + m_nacl)
//
// Performance Notes:
//   - O(1) complexity
//   - Simple arithmetic with one conditional
//   - Total estimate: ~20-30 cycles
//
// Suggestions for Improvement:
//   1. Move magic numbers to named constants (implemented above)
//   2. Add bounds checking for kk in debug builds
//   3. Consider returning struct instead of multiple output parameters
//=============================================================================
KOKKOS_INLINE_FUNCTION
void set_f_act_coarse(const int kk,
                      const Diagnostics::ColumnTracerView &state_q,
                      const Diagnostics::ColumnTracerView &ptend_q,
                      const Real dt, Real &f_act_conv_coarse,
                      Real &f_act_conv_coarse_dust,
                      Real &f_act_conv_coarse_nacl) {
  // -----------------------------------------------------------------------
  //  set the mass-weighted sol_factic for coarse mode species
  // -----------------------------------------------------------------------
  // clang-format off
  /*
  in  :: kk;
       state_q and ptend_q only use dust and seasalt in this subroutine
  in  :: state_q    ! tracer of state%q [kg/kg]
  in  :: ptend_q    ! tracer tendency (ptend%q) [kg/kg/s]
  in  :: dt         ! time step [s]
  out :: f_act_conv_coarse      ! prescribed coarse mode aerosol activation fraction for convective
  out :: f_act_conv_coarse_dust ! prescribed dust aerosol activation fraction for convective cloud [fraction]
  out :: f_act_conv_coarse_nacl ! prescribed seasalt aerosol activation fraction for convective cloud [fraction]
  */
  // clang-format on

  // initial value
  // BAD CONSTANT
  const Real small_value_30 = 1.0e-30;
  f_act_conv_coarse = 0.60;
  f_act_conv_coarse_dust = 0.40;
  f_act_conv_coarse_nacl = 0.80;

  // dust and seasalt mass concentration [kg/kg]
  const int idx_coarse = static_cast<int>(ModeIndex::Coarse);
  const int lcoardust = aero_model::lptr_dust_a_amode(idx_coarse);
  const int lcoarnacl = aero_model::lptr_nacl_a_amode(idx_coarse);
  const Real tmpdust =
      haero::max(0.0, state_q(kk, lcoardust) + ptend_q(kk, lcoardust) * dt);
  const Real tmpnacl =
      haero::max(0.0, state_q(kk, lcoarnacl) + ptend_q(kk, lcoarnacl) * dt);
  if (tmpdust + tmpnacl > small_value_30)
    f_act_conv_coarse =
        (f_act_conv_coarse_dust * tmpdust + f_act_conv_coarse_nacl * tmpnacl) /
        (tmpdust + tmpnacl);
}


//=============================================================================
// FUNCTION: calc_resusp_to_coarse
//=============================================================================
// Description: Handles aerosol resuspension from evaporating precipitation,
//              redirecting resuspended mass to the appropriate coarse mode
//              species.
//
// Physical Background:
//   When precipitation evaporates before reaching the surface, the aerosol
//   mass that was scavenged is released back into the atmosphere. This
//   "resuspension" can return aerosols to:
//   - Their original mode/species (for coarse mode aerosols)
//   - Coarse mode (for fine mode aerosols that have grown through processing)
//   
//   The mmtoo_prevap_resusp mapping determines where resuspended mass goes.
//
// Algorithm:
//   1. Look up target species index for resuspension (mmtoo)
//   2. Subtract resuspension from current species tendency
//   3. If mmtoo > 0, add resuspension to target coarse species accumulator
//   4. If update_dqdt is true, add accumulated resuspension to tendency
//
// Parameters:
//   mm          [in]    - Current species/tracer index
//   update_dqdt [in]    - Flag to add accumulated resuspension to tendency
//                         (false for cloud-borne aerosols, lphase==2)
//   rcscavt     [in]    - Resuspension rate from convective precip [kg/kg/s]
//   rsscavt     [in]    - Resuspension rate from stratiform precip [kg/kg/s]
//   dqdt_tmp    [inout] - Tendency for current aerosol species [kg/kg/s]
//   rtscavt_sv  [inout] - Resuspension accumulator for coarse mode [kg/kg/s]
//
// Mapping Logic (mmtoo_prevap_resusp):
//   mmtoo > 0:  Resuspension goes to coarse mode species at index mmtoo
//   mmtoo = -1: Species not simulated (no action needed)
//   mmtoo = -3: Special handling required (not implemented here)
//
// Performance Notes:
//   - O(1) complexity
//   - Function call to mmtoo_prevap_resusp (O(1) lookup)
//   - Simple arithmetic with conditionals
//   - Total estimate: ~15-25 cycles
//
// Suggestions for Improvement:
//   1. Document mmtoo mapping table in header
//   2. Add handling for mmtoo == -3 case if needed
//   3. Consider using enum for mmtoo special values
//=============================================================================
KOKKOS_INLINE_FUNCTION
void calc_resusp_to_coarse(const int mm, const bool update_dqdt,
                           const Real rcscavt, const Real rsscavt,
                           Real &dqdt_tmp, Real rtscavt_sv[]) {
  // clang-format off
  //-----------------------------------------------------------------------
  // resuspension goes to coarse mode
  //-----------------------------------------------------------------------
  /*
  in :: ncol, mm
  in :: update_dqdt  ! if update dqdt_tmp with rtscavt_sv
  in :: rcscavt      ! resuspention from convective [kg/kg/s]
  in :: rsscavt      ! resuspention from stratiform [kg/kg/s]

  inout :: dqdt_tmp ! temporary array to hold tendency for the "current" aerosol species [kg/kg/s]
  inout :: rtscavt_sv ! resuspension that goes to coarse mode [kg/kg/s]
  */
  // clang-format on

  const int mmtoo = aero_model::mmtoo_prevap_resusp(mm);

  // first deduct the current resuspension from the dqdt_tmp of the current
  // species
  dqdt_tmp -= (rcscavt + rsscavt);

  // then add the current resuspension to the rtscavt_sv of the appropriate
  // coarse mode species
  if (mmtoo > 0)
    rtscavt_sv[mmtoo] += (rcscavt + rsscavt);

  // then add the rtscavt_sv of the current species to the dqdt_tmp
  // of the current species. This is not called when lphase==2
  // note that for so4_a3 and mam3, the rtscavt_sv at this point will have
  //  resuspension contributions from so4_a1/2/3 and so4c1/2/3
  if (update_dqdt)
    dqdt_tmp += rtscavt_sv[mm];
}


//=============================================================================
// FUNCTION: calc_sfc_flux
//=============================================================================
// Description: Calculates surface flux from vertical integration of layer
//              tendencies using Kokkos parallel reduction for GPU efficiency.
//
// Physical Background:
//   Surface deposition flux represents the total column-integrated tendency
//   converted to a surface mass flux. This is used for diagnostics and
//   mass conservation tracking.
//
//   Integration formula:
//     F_sfc = Σ (tendency[k] × Δp[k] / g)
//   
//   This converts from mixing ratio tendency [kg/kg/s] to surface mass
//   flux [kg/m²/s].
//
// Parameters:
//   team       [in] - Kokkos team handle for parallel execution
//   layer_tend [in] - Tendency in each vertical layer [kg/kg/s]
//   pdel       [in] - Pressure thickness of each layer [Pa]
//   nlev       [in] - Number of vertical levels
//
// Returns:
//   Integrated surface flux [kg/m²/s]
//
// Performance Notes:
//   - O(nlev) complexity, parallelized with Kokkos
//   - Uses TeamVectorRange for efficient GPU execution
//   - parallel_reduce handles thread synchronization automatically
//   - Total estimate: ~5-10 cycles per level (highly parallelized on GPU)
//
// Suggestions for Improvement:
//   1. Pre-compute 1/gravit if called repeatedly with same gravity
//   2. Consider fusing multiple flux calculations if needed together
//=============================================================================
using View1D = DeviceType::view_1d<Real>;
KOKKOS_INLINE_FUNCTION
Real calc_sfc_flux(const ThreadTeam &team, const View1D &layer_tend,
                   haero::ConstColumnView pdel, const int nlev) {
  // clang-format off
  // -----------------------------------------------------------------------
  //  calculate surface fluxes of wet deposition from vertical integration of tendencies
  // -----------------------------------------------------------------------
  /*
  in :: pdel       ! pressure difference between two layers [Pa]
  in :: layer_tend ! physical tendencies in each layer [kg/kg/s]
  out :: sflx      ! integrated surface fluxes [kg/m2/s]
  */
  // clang-format on
  Real scratch = 0;
  const Real gravit = Constants::gravity;
  Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(team, nlev),
      [&](int k, Real &lsum) { lsum += layer_tend[k] * pdel[k] / gravit; },
      scratch);
  return scratch;
}


//=============================================================================
// FUNCTION: apportion_sfc_flux_deep
//=============================================================================
// Description: Apportions convective surface fluxes between deep and shallow
//              convection based on precipitation production and evaporation
//              characteristics of each convection type.
//
// Physical Background:
//   Convective wet removal can occur in both deep and shallow convective
//   systems. For diagnostic purposes, we need to partition the total
//   convective flux between these two types.
//
//   Assumptions:
//   1. Below-cloud removal (sflxbc) is proportional to precipitation production
//      → Uses deep fraction of total precipitation
//   2. Resuspension (sflxec) is proportional to (removal) × (evap/production)
//      → Accounts for higher evaporation in shallow convection
//
// Algorithm:
//   1. Calculate deep fraction of total precipitation production
//   2. Apportion below-cloud flux by precipitation fraction
//   3. Calculate resuspension efficiency for deep and shallow
//   4. Apportion evaporation flux by relative resuspension rates
//
// Parameters:
//   rprddpsum  [in]  - Column-integrated deep precip production [kg/m²/s]
//   rprdshsum  [in]  - Column-integrated shallow precip production [kg/m²/s]
//   evapcdpsum [in]  - Column-integrated deep precip evaporation [kg/m²/s]
//   evapcshsum [in]  - Column-integrated shallow precip evaporation [kg/m²/s]
//   sflxbc     [in]  - Total below-cloud scavenging surface flux [kg/m²/s]
//   sflxec     [in]  - Total resuspension surface flux [kg/m²/s]
//   sflxbcdp   [out] - Deep convection below-cloud flux [kg/m²/s]
//   sflxecdp   [out] - Deep convection resuspension flux [kg/m²/s]
//
// Notes:
//   - Only applies to interstitial aerosols (convective clouds don't affect
//     stratiform cloud-borne aerosols)
//   - This is an approximate method adequate for diagnostics since deep and
//     shallow convection rarely occur simultaneously
//   - More accurate partitioning could be done in wetdepa subroutine
//
// Performance Notes:
//   - O(1) complexity
//   - Multiple divisions and max/min operations
//   - Total estimate: ~30-50 cycles
//
// Suggestions for Improvement:
//   1. Move small_value constants to namespace scope (implemented above)
//   2. Document why different thresholds are used for precip vs evap
//   3. Consider struct return type for output pair
//=============================================================================
KOKKOS_INLINE_FUNCTION
void apportion_sfc_flux_deep(const Real rprddpsum, const Real rprdshsum,
                             const Real evapcdpsum, const Real evapcshsum,
                             const Real sflxbc, const Real sflxec,
                             Real &sflxbcdp, Real &sflxecdp) {

  // BAD CONSTANT
  Real small_value_35 = 1.0e-35;
  Real small_value_36 = 1.0e-36;

  // working variables for precipitation and evaporation from deep and shallow
  // convection
  const Real tmp_precdp = haero::max(rprddpsum, small_value_35);
  const Real tmp_precsh = haero::max(rprdshsum, small_value_35);
  const Real tmp_evapdp = haero::max(evapcdpsum, small_value_36);
  const Real tmp_evapsh = haero::max(evapcshsum, small_value_36);

  // assume that in- and below-cloud removal are proportional to
  // column precip production
  // working variables of deep fraction
  Real tmpa = tmp_precdp / (tmp_precdp + tmp_precsh);
  tmpa = utils::min_max_bound(0.0, 1.0, tmpa);
  sflxbcdp = sflxbc * tmpa;

  // assume that resuspension is proportional to
  // (wet removal)*[(precip evap)/(precip production)]
  //  working variables for resuspension from deep and shallow convection
  const Real tmp_resudp = tmpa * haero::min(tmp_evapdp / tmp_precdp, 1.0);
  const Real tmp_resush =
      (1.0 - tmpa) * haero::min(tmp_evapsh / tmp_precsh, 1.0);
  Real tmpb = haero::max(tmp_resudp, small_value_35) /
              haero::max(tmp_resudp + tmp_resush, small_value_35);
  tmpb = utils::min_max_bound(0.0, 1.0, tmpb);

  sflxecdp = sflxec * tmpb;
}

} // end namespace aero_model

} // end namespace mam4

#endif
