// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_AERO_MODES_HPP
#define MAM4XX_AERO_MODES_HPP

#include <haero/aero_species.hpp>
#include <haero/constants.hpp>
#include <haero/gas_species.hpp>
#include <haero/math.hpp>

#include "mam4_types.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace mam4 {

/// @struct Mode
/// This struct represents an aerosol particle mode and contains all associated
/// metadata. By definition, these metadata are immutable (constant in time).
/// The struct is not polymorphic, so don't derive any subclass from it.
///
/// This class represents the log-normal distribution that defines the mode via
/// the mean_std_dev member variable.  The other parameter necessary to define
/// the log-normal function is a variable (a function of mass- and number-
/// mixing ratios) and is not included in this class.
///
/// The member variables min_diameter and max_diameter do not define the bounds
/// of the log-normal distribution (which, matematically, are 0 and positive
/// infinity).  Rather, these min/max values are used to trigger a mass and
/// number redistribution elsewhere in the code; they signify the bounds beyond
/// which particles are considered to better belong in a different mode.
///
/// Variable nom_diameter is the nominal geometeric mean diameter [m]
/// of particles in a mode
///
/// Crystalization and deliquesence refer to the non-cloud water uptake process,
/// by which liquid water condenses into aerosol droplets.  They are relative
/// humidity values.  When the environmental relative humidity lies below the
/// cyrstalization point, water uptake does not occur.  When it lies between the
/// crystallization and deliquesence point, water uptake does occur, but not at
/// its maximum rate.   When the environmental relative humidty exceeds the
/// deliquescence_pt, particles achieve their maximum amount of liquid water.
///
struct Mode final {
  using Real = haero::Real;
  /// The minimum diameter for particles that belong to this mode.
  const Real min_diameter;
  /// The nominal diameter for particles that belong to this mode.
  const Real nom_diameter;
  /// The maximum diameter for particles that belong to this mode.
  const Real max_diameter;
  /// The geometric mean standard deviation for this mode.
  const Real mean_std_dev;
  /// The crystallization point [rel. humidity] for this mode.
  const Real crystallization_pt;
  /// The deliquescence point [rel. humidity] for this mode.
  const Real deliquescence_pt;
};

/// Mode indices in MAM4
enum class ModeIndex {
  Accumulation = 0,
  Aitken = 1,
  Coarse = 2,
  PrimaryCarbon = 3,
  None = 4, // invalid index
};

/// Map ModeIndex to string (for logging, e.g.)
/// This function cannot be called inside a GPU kernel,
/// but it's helpful to use with ekat::Logger statements
/// (which also cannot be called from inside a kernel)
std::string mode_str(const ModeIndex m);

static constexpr Real mam4_crystallization_rel_hum = 0.35;
static constexpr Real mam4_delequesence_rel_hum = 0.8;
static constexpr Real mam4_accum_min_diameter_m = 5.35e-8;
static constexpr Real mam4_accum_nom_diameter_m = 1.1e-7;
static constexpr Real mam4_accum_max_diameter_m = 4.4e-7;
static constexpr Real mam4_accum_mead_std_dev = 1.8;
static constexpr Real mam4_aitken_min_diameter_m = 8.7e-9;
static constexpr Real mam4_aitken_nom_diameter_m = 2.6e-8;
static constexpr Real mam4_aitken_max_diameter_m = 5.2e-8;
static constexpr Real mam4_aitken_mead_std_dev = 1.6;
static constexpr Real mam4_coarse_min_diameter_m = 1e-6;
static constexpr Real mam4_coarse_nom_diameter_m = 2e-6;
static constexpr Real mam4_coarse_max_diameter_m = 4e-6;
static constexpr Real mam4_coarse_mead_std_dev = 1.8;
static constexpr Real mam4_primary_carbon_min_diameter_m = 1e-8;
static constexpr Real mam4_primary_carbon_nom_diameter_m = 5e-8;
static constexpr Real mam4_primary_carbon_max_diameter_m = 1e-7;
static constexpr Real mam4_primary_carbon_mead_std_dev = 1.60000002384186;

/// A list of all modes within MAM4.
/// NOTE: MAM4 uses the same constant crystallization and deliquescence
/// NOTE: values for all modes & species.  See links for additional discussion:
/// NOTE:
/// https://eagles-project.atlassian.net/wiki/spaces/Computation/pages/1125515265/Aerosol+species+and+mode+data
/// NOTE:
/// https://eagles-project.atlassian.net/wiki/spaces/Computation/pages/354877515/Module+verifications
/// NOTE: These data are found on Anvil in
/// NOTE: /lcrc/group/acme/ccsm-data/inputdata/atm/cam/physprops/
KOKKOS_INLINE_FUNCTION const mam4::Mode &modes(const int i) {
  static const mam4::Mode M[4] = {
      // accumulation
      {mam4_accum_min_diameter_m, mam4_accum_nom_diameter_m,
       mam4_accum_max_diameter_m, mam4_accum_mead_std_dev,
       mam4_crystallization_rel_hum, mam4_delequesence_rel_hum},
      // aitken
      {mam4_aitken_min_diameter_m, mam4_aitken_nom_diameter_m,
       mam4_aitken_max_diameter_m, mam4_aitken_mead_std_dev,
       mam4_crystallization_rel_hum, mam4_delequesence_rel_hum},
      // coarse
      {mam4_coarse_min_diameter_m, mam4_coarse_nom_diameter_m,
       mam4_coarse_max_diameter_m, mam4_coarse_mead_std_dev,
       mam4_crystallization_rel_hum, mam4_delequesence_rel_hum},
      // primary carbon
      {mam4_primary_carbon_min_diameter_m, mam4_primary_carbon_nom_diameter_m,
       mam4_primary_carbon_max_diameter_m, mam4_primary_carbon_mead_std_dev,
       mam4_crystallization_rel_hum, mam4_delequesence_rel_hum}};
  return M[i];
};

/// Identifiers for aerosol species that inhabit MAM4 modes.
enum class AeroId {
  SOA = 0,  // secondary organic aerosol
  SO4 = 1,  // sulphate
  POM = 2,  // primary organic matter
  BC = 3,   // black carbon
  NaCl = 4, // sodium chloride
  DST = 5,  // dust
  MOM = 6,  // marine organic matter,
  None = 7  // invalid aerosol species
};

/// Map ModeIndex to string (for logging, e.g.)
/// This function cannot be called inside a GPU kernel,
/// but it's helpful to use with ekat::Logger statements
/// (which also cannot be called inside a kernel)
std::string aero_id_str(const AeroId aid);

/// Map ModeIndex to string (it is used by mam_coupling in emaxx)
/// give aerosol id return the aersol short name
/// this is name from e3sm
std::string aero_id_short_name(const AeroId aid);

/// Molecular weight of mam4 dust aerosol [kg/mol]
static constexpr Real mam4_molec_weight_dst = 0.135065;

/// Molecular weight of mam4 marine organic matter [kg/mol]
static constexpr Real mam4_molec_weight_mom = 250.093;

/// mam4 aerosol densities [kg/m3]
static constexpr Real mam4_density_soa = 1000.0;
static constexpr Real mam4_density_so4 = 1770.0;
static constexpr Real mam4_density_pom = 1000.0;
static constexpr Real mam4_density_bc = 1700.0;
static constexpr Real mam4_density_nacl = 1900.0;
static constexpr Real mam4_density_dst = 2600.0;
static constexpr Real mam4_density_mom = 1601.0;

/// mam4 aerosol hygroscopicities
static constexpr Real mam4_hyg_soa = 0.1;
static constexpr Real mam4_hyg_so4 = 0.507;
static constexpr Real mam4_hyg_pom = 1e-10;
static constexpr Real mam4_hyg_bc = 1e-10;
static constexpr Real mam4_hyg_nacl = 1.16;
static constexpr Real mam4_hyg_dst = 0.14;
static constexpr Real mam4_hyg_mom = 0.1;

/// A list of aerosol species in MAM4.
/**
  Note that in MAM4 fortran, molecular weights are given as g/mol, rather than
  kg/mol.

  Here and in Haero we use SI units for everything, so molecular weights
  are given as [kg/mol].

  When the variable is "universal" in the sense that it will be the same
  whether MAM4 is using or some other software package is using it, we
  use the external haero::Constants value, which is sourced to the latest
  NIST data available.  Additionally, this prepares Mam4xx to ultimately
  use an external source of constants with EAM.  Examples are the
  molecular weights of Carbon, Sulphate, and Sodium Chloride.

  Some of these constants are unique to mam4 -- these are listed here, with
  the prefix mam4_*. For example, its definition
  of primary carbon, dust, and marine organic matter are defined by choices
  of what those modes represent.  Other examples, such as the density of some
  substances, differ from the values provided by NIST; these, too, are listed
  here as mam4_* constants.
*/
KOKKOS_INLINE_FUNCTION AeroSpecies aero_species(const int i) {
  static const AeroSpecies species[7] = {
      AeroSpecies{Constants::molec_weight_c, mam4_density_soa,
                  mam4_hyg_soa}, // secondary organic aerosol
      AeroSpecies{Constants::molec_weight_so4, mam4_density_so4, mam4_hyg_so4},
      AeroSpecies{Constants::molec_weight_c, mam4_density_pom,
                  mam4_hyg_pom}, // primary organic matter
      AeroSpecies{Constants::molec_weight_c, mam4_density_bc,
                  mam4_hyg_bc}, // black carbon
      AeroSpecies{Constants::molec_weight_nacl, mam4_density_nacl,
                  mam4_hyg_nacl}, // sodium chloride
      AeroSpecies{mam4_molec_weight_dst, mam4_density_dst,
                  mam4_hyg_dst}, // dust
      AeroSpecies{mam4_molec_weight_mom, mam4_density_mom,
                  mam4_hyg_mom} // marine organic matter
  };
  return species[i];
}

//=============================================================================
// FUNCTION: mode_aero_species
//=============================================================================
// Description: Returns the aerosol species identifier (AeroId) for a given
//              species slot within a specified aerosol mode.
//
// Physical Background:
//   MAM4 organizes aerosol species into four modes based on particle size
//   and source characteristics. Each mode contains a specific subset of
//   the seven possible aerosol species, reflecting the physical processes
//   that form and transform aerosols in each size range.
//
// Mode-Species Distribution:
//   ┌─────────────────┬───────────────────────────────────────────────────────┐
//   │ Mode            │ Species (in order)                                    │
//   ├─────────────────┼───────────────────────────────────────────────────────┤
//   │ 0: Accumulation │ SO4, POM, SOA, BC, DST, NaCl, MOM  (7 species)        │
//   │ 1: Aitken       │ SO4, SOA, NaCl, MOM, None, None, None (4 species)     │
//   │ 2: Coarse       │ DST, NaCl, SO4, BC, POM, SOA, MOM  (7 species)        │
//   │ 3: PrimaryCarbon│ POM, BC, MOM, None, None, None, None (3 species)      │
//   └─────────────────┴───────────────────────────────────────────────────────┘
//
// Parameters:
//   modeNo    [in] - Mode index (0-3, corresponding to ModeIndex enum)
//   speciesNo [in] - Species slot index within the mode (0-6)
//
// Returns:
//   AeroId enum value for the species at that slot
//   AeroId::None if the slot is unused (species count < 7 for that mode)
//
// Usage Example:
//   AeroId species = mode_aero_species(0, 4);  // Returns AeroId::DST (dust in accum)
//   AeroId species = mode_aero_species(1, 5);  // Returns AeroId::None (unused slot)
//
// Performance Notes:
//   - O(1) array lookup
//   - Static constexpr array: zero initialization cost at runtime
//   - Total estimate: ~2-5 cycles
//
// Suggestions for Improvement:
//   1. Add bounds checking for modeNo and speciesNo in debug builds
//   2. Consider using ModeIndex enum instead of raw integer
//   3. Rename local array to avoid shadowing function name
//
// Related Functions:
//   - num_species_mode(): Returns count of active species per mode
//   - aerosol_index_for_mode(): Finds species index within a mode
//   - mode_contains_species(): Checks if mode contains a species
//   - lmassptr_amode(): Returns tracer array index for species mass
//=============================================================================
// A list of species within each mode for MAM4.
KOKKOS_INLINE_FUNCTION AeroId mode_aero_species(const int modeNo,
                                                const int speciesNo) {
  // A list of species within each mode for MAM4.
  static constexpr AeroId mode_aero_species[4][7] = {
      {// accumulation mode
       AeroId::SO4, AeroId::POM, AeroId::SOA, AeroId::BC, AeroId::DST,
       AeroId::NaCl, AeroId::MOM},
      {
          // aitken mode
          AeroId::SO4,
          AeroId::SOA,
          AeroId::NaCl,
          AeroId::MOM,
          AeroId::None,
          AeroId::None,
          AeroId::None,
      },
      {// coarse mode
       AeroId::DST, AeroId::NaCl, AeroId::SO4, AeroId::BC, AeroId::POM,
       AeroId::SOA, AeroId::MOM},
      {// primary carbon mode
       AeroId::POM, AeroId::BC, AeroId::MOM, AeroId::None, AeroId::None,
       AeroId::None, AeroId::None}};
  return mode_aero_species[modeNo][speciesNo];
}

//=============================================================================
// FUNCTION: num_species_mode
//=============================================================================
// Description: Returns the number of active (non-None) aerosol species
//              in the specified mode.
//
// Purpose:
//   Used to determine loop bounds when iterating over species within a mode,
//   avoiding unnecessary iterations over AeroId::None slots.
//
// Parameter:
//   i [in] - Mode index (0-3)
//
// Returns:
//   Number of active species in the mode (3, 4, or 7)
//
// Usage Example:
//   for (int s = 0; s < num_species_mode(imode); ++s) {
//       AeroId species = mode_aero_species(imode, s);
//       // Process species...
//   }
//
// Performance Notes:
//   - O(1) array lookup
//   - Static constexpr: zero runtime initialization cost
//
// Related:
//   - Used by index_ordering() to determine species type (mass vs number)
//   - nspec_amode in Fortran equivalent
//=============================================================================
KOKKOS_INLINE_FUNCTION int num_species_mode(const int i) {
  static constexpr int _num_species_mode[4] = {7, 4, 7, 3};
  return _num_species_mode[i];
}


//=============================================================================
// FUNCTION: aerosol_index_for_mode
//=============================================================================
// Description: Finds the index position of a given aerosol species within
//              a specified mode's species list.
//
// Purpose:
//   Maps from a species identifier (AeroId) to its position within a mode,
//   which is needed for accessing mode-specific species arrays.
//
// Algorithm:
//   Linear search through the mode's species list (max 7 iterations)
//
// Parameters:
//   mode    [in] - ModeIndex enum value (Accumulation, Aitken, Coarse, PrimaryCarbon)
//   aero_id [in] - AeroId enum value of the species to find
//
// Returns:
//   Index (0-6) of the species within the mode if found
//   -1 if the species is not present in the mode
//
// Usage Example:
//   int idx = aerosol_index_for_mode(ModeIndex::Coarse, AeroId::DST);  // Returns 0
//   int idx = aerosol_index_for_mode(ModeIndex::Aitken, AeroId::BC);   // Returns -1
//
// Performance Notes:
//   - O(n) linear search where n ≤ 7
//   - Worst case: 7 comparisons
//   - Consider binary search or hash map if called frequently
//
// Suggestions for Improvement:
//   1. Use num_species_mode() as loop bound instead of hardcoded 7
//   2. Consider precomputed inverse lookup table for O(1) access
//   3. Add early termination when AeroId::None is encountered
//=============================================================================
KOKKOS_INLINE_FUNCTION
int aerosol_index_for_mode(ModeIndex mode, AeroId aero_id) {
  int mode_index = static_cast<int>(mode);
  for (int s = 0; s < 7; ++s) {
    if (aero_id == mode_aero_species(mode_index, s)) {
      return s;
    }
  }
  return -1;
}


//=============================================================================
// FUNCTION: mode_contains_species
//=============================================================================
// Description: Convenience function to check whether a mode contains a
//              specific aerosol species.
//
// Purpose:
//   Provides a boolean interface for species membership queries, which is
//   more readable than checking aerosol_index_for_mode() != -1.
//
// Parameters:
//   mode    [in] - ModeIndex enum value
//   aero_id [in] - AeroId enum value to check
//
// Returns:
//   true  - Species is present in the mode
//   false - Species is not present in the mode
//
// Usage Example:
//   if (mode_contains_species(ModeIndex::Aitken, AeroId::BC)) {
//       // This branch is NOT taken (BC not in Aitken mode)
//   }
//
// Species Presence Matrix:
//   ┌─────────┬───────┬────────┬────────┬─────────────────┐
//   │ Species │ Accum │ Aitken │ Coarse │ Primary Carbon  │
//   ├─────────┼───────┼────────┼────────┼─────────────────┤
//   │ SO4     │  ✓    │   ✓    │   ✓    │       ✗         │
//   │ POM     │  ✓    │   ✗    │   ✓    │       ✓         │
//   │ SOA     │  ✓    │   ✓    │   ✓    │       ✗         │
//   │ BC      │  ✓    │   ✗    │   ✓    │       ✓         │
//   │ DST     │  ✓    │   ✗    │   ✓    │       ✗         │
//   │ NaCl    │  ✓    │   ✓    │   ✓    │       ✗         │
//   │ MOM     │  ✓    │   ✓    │   ✓    │       ✓         │
//   └─────────┴───────┴────────┴────────┴─────────────────┘
//
//=============================================================================
KOKKOS_INLINE_FUNCTION
bool mode_contains_species(ModeIndex mode, AeroId aero_id) {
  return -1 != aerosol_index_for_mode(mode, aero_id);
}


//=============================================================================
// ENUM CLASS: GasId
//=============================================================================
// Description: Identifiers for gas-phase species tracked in MAM4.
//              These gases participate in aerosol formation, growth,
//              and chemical transformations.
//
// Gas Species Overview:
//   ┌─────────┬────────────────────────┬─────────────────────────────────────┐
//   │ GasId   │ Name                   │ Role in Aerosol Processes           │
//   ├─────────┼────────────────────────┼─────────────────────────────────────┤
//   │ O3      │ Ozone                  │ Oxidant for SO2, DMS, VOCs          │
//   │ H2O2    │ Hydrogen Peroxide      │ Aqueous-phase oxidant for SO2       │
//   │ H2SO4   │ Sulfuric Acid          │ Nucleation, condensation growth     │
//   │ SO2     │ Sulfur Dioxide         │ Precursor to sulfate aerosol        │
//   │ DMS     │ Dimethyl Sulfide       │ Marine SO2 precursor                │
//   │ SOAG    │ SOA Precursor Gas      │ Lumped VOC oxidation products       │
//   │ None    │ Invalid/Placeholder    │ Array padding, error checking       │
//   └─────────┴────────────────────────┴─────────────────────────────────────┘
//
// Chemical Pathways:
//
//   DMS (ocean) → SO2 → H2SO4 → Sulfate aerosol (SO4)
//                  ↑        ↓
//                 O3     Nucleation (new particles)
//                  ↓        ↓
//                H2O2   Condensation (growth)
//                  ↓
//            Aqueous oxidation (in-cloud)
//
//   VOCs → SOAG → SOA (secondary organic aerosol)
//
// Integer Values:
//   Used for array indexing into gas tracer arrays
//   Values 0-5 are valid; 6 (None) indicates invalid/unused
//
// Usage Example:
//   int idx = static_cast<int>(GasId::H2SO4);  // Returns 2
//   Real h2so4_conc = gas_mixing_ratios[static_cast<int>(GasId::H2SO4)];
//=============================================================================
enum class GasId {
  O3 = 0,    // ozone
  H2O2 = 1,  // hydrogen peroxide
  H2SO4 = 2, // sulfuric acid
  SO2 = 3,   // sulfur dioxide
  DMS = 4,   // dimethyl sulfide
  SOAG = 5,  // secondary organic aerosol precursor
  None = 6,  // invalid gas id
};


/// Molecular weight of carbon dioxide [kg/mol]
static constexpr Real molec_weight_co2 = 0.0440095;
/// Molecular weight of methane @f$\text{CH}_4@f$
static constexpr Real molec_weight_ch4 = 0.0160425;
/// Molecular weight of trichlorofluoromethan @f$\text{CCl}_3\text{F}@f$
static constexpr Real molec_weight_ccl3f = 0.13736;
/// Molecular weight of dichlorofluoromethane @f$\texct{CHCl}_2F@f$
static constexpr Real molec_weight_chcl2f = 0.10292;
/// Molecular weight of hydrogen peroxide @f$\text{H}_2\text{O}_2@f$
static constexpr Real molec_weight_h2o2 = 0.034015;
/// Molecular weight of dimethylsulfide @f$\text{C}_2\text{H}_6\text{S}@f$
static constexpr Real molec_weight_dms = 0.06214;
/// Molecular weight of oxygen molecule @f$\text{O}_2@f$
static constexpr Real molec_weight_o2 = 0.0319988;
/// Molecular weight of nitrous oxide @f$\text{N}_2\text{O}@f$
static constexpr Real molec_weight_n2o = 0.044013;
/// Molecular weight of ozone @f$\text{O}_3@f$
static constexpr Real molec_weight_o3 = 0.0479982;
/// Molecular weight of sulfur dioxide @f$\text{SO}_2@f$
static constexpr Real molec_weight_so2 = 0.06407;


//=============================================================================
// FUNCTION: gas_species
//=============================================================================
// Description: Returns the GasSpecies structure containing molecular weight
//              and other properties for the specified gas species index.
//
// Physical Background:
//   MAM4 tracks multiple gas-phase species that participate in:
//   - Aerosol nucleation (H₂SO₄)
//   - Aerosol growth by condensation (H₂SO₄, SOAG)
//   - Oxidation chemistry (O₃, H₂O₂, OH)
//   - Sulfur cycle (SO₂, DMS, H₂SO₄)
//   - Climate forcing (CO₂, CH₄, N₂O, CFCs)
//
// Gas Species List (13 total):
//   ┌───────┬─────────────────────────────┬────────────────┬──────────────────┐
//   │ Index │ Species                     │ Formula        │ MW [kg/mol]      │
//   ├───────┼─────────────────────────────┼────────────────┼──────────────────┤
//   │   0   │ Ozone                       │ O₃             │ 0.0479982        │
//   │   1   │ Hydrogen peroxide           │ H₂O₂           │ 0.034015         │
//   │   2   │ Sulfuric acid               │ H₂SO₄          │ 0.098079         │
//   │   3   │ Sulfur dioxide              │ SO₂            │ 0.06407          │
//   │   4   │ Dimethylsulfide             │ (CH₃)₂S        │ 0.06214          │
//   │   5   │ SOA precursor gas           │ (lumped)       │ ~0.012 (carbon)  │
//   │   6   │ Oxygen                      │ O₂             │ 0.0319988        │
//   │   7   │ Carbon dioxide              │ CO₂            │ 0.0440095        │
//   │   8   │ Nitrous oxide               │ N₂O            │ 0.044013         │
//   │   9   │ Methane                     │ CH₄            │ 0.0160425        │
//   │  10   │ Trichlorofluoromethane      │ CCl₃F (CFC-11) │ 0.13736          │
//   │  11   │ Dichlorofluoromethane       │ CHCl₂F         │ 0.10292          │
//   │  12   │ Ammonia                     │ NH₃            │ 0.017031         │
//   └───────┴─────────────────────────────┴────────────────┴──────────────────┘
//
// Correspondence to GasId enum (first 6 species):
//   Index 0 → GasId::O3
//   Index 1 → GasId::H2O2
//   Index 2 → GasId::H2SO4
//   Index 3 → GasId::SO2
//   Index 4 → GasId::DMS
//   Index 5 → GasId::SOAG
//
// Parameter:
//   i [in] - Gas species index (0-12)
//
// Returns:
//   GasSpecies structure containing molecular weight [kg/mol]
//
// Usage Example:
//   GasSpecies so2 = gas_species(3);
//   Real mw_so2 = so2.molecular_weight;  // 0.06407 kg/mol
//
// Performance Notes:
//   - O(1) array lookup
//   - Static array: initialized once at first call
//   - Total estimate: ~2-5 cycles
//
// Suggestions for Improvement:
//   1. Add bounds checking for index i in debug builds
//   2. Consider using GasId enum instead of raw integer for type safety
//   3. Ensure species ordering matches GasId enum values
//   4. Move static array to namespace scope as constexpr if GasSpecies is literal
//   5. Fix typo in comment: "thrichlorofluoromethane" → "trichlorofluoromethane"
//
// Related:
//   - GasId enum for gas species identification
//   - Constants::molec_weight_h2so4 for sulfuric acid
//   - Constants::molec_weight_nh3 for ammonia
//   - Constants::molec_weight_c for carbon (used for SOAG)
//=============================================================================
/// A list of gas species in MAM4.
KOKKOS_INLINE_FUNCTION GasSpecies gas_species(const int i) {
  static const GasSpecies species[13] = {
      {molec_weight_o3},               // ozone
      {molec_weight_h2o2},             // hydrogen peroxide
      {Constants::molec_weight_h2so4}, // sulfuric acid
      {molec_weight_so2},              // sulfur dioxide
      {molec_weight_dms},              // dimethylsulfide
      {Constants::molec_weight_c},     // secondary organic aerosol precursor
      {molec_weight_o2},               // oxygen
      {molec_weight_co2},              // carbon dioxide
      {molec_weight_n2o},              // nitrous oxide
      {molec_weight_ch4},              // methane
      {molec_weight_ccl3f},            // thrichlorofluoromethane
      {molec_weight_chcl2f},           // dichlorofluoromethane
      {Constants::molec_weight_nh3}    // ammonia
  };
  return species[i];
}

} // namespace mam4

#endif
