#pragma once

#include <optional>
#include <string>
#include <utility>

#include "mesh.hh"
#include "solid_mechanics_model_cohesive.hh"
#include "fragment_manager.hh"
/// All CLI/runtime parameters
struct Args {
  std::string material_file = "AD995_cohesive_contact_m10_stable.dat";
  std::string mesh_file = "plate_0.01x0.01_npz1_P1.msh";
  akantu::Real strain_rate = 2.559e4;
  akantu::Real velocity = 10.0;
  akantu::Real safety_factor = 0.2;
  akantu::Real time = 0.0;
  akantu::Real kappa = 0.2;
  akantu::Real center_x = 0.0;
  akantu::Real center_y = 0.0;
  akantu::Real z_sign = 1.0;
  std::optional<akantu::Real> cutoff = std::nullopt;
  std::string shape = "gaussian";
  akantu::Real angle_xz = 0.0;
  akantu::Real angle_yz = 0.0;
  std::optional<std::string> mesh_conv = std::nullopt;
};

/// Parse `argv` into Args (simple, dependency-free)
Args parseArguments(int argc, char *argv[]);

/// Setup Paraview dumpers (bulk + cohesive elements)
void initParaviewDumpers(akantu::SolidMechanicsModelCohesive &model,
                         const std::string &outpath);

/// Prepare IO paths; clears/creates output directory. Returns {inpath,
/// outpath}.
std::pair<std::string, std::string> setupDir(const std::string &nname,
                                             const Args &args, int prank);

/// Apply a z-only Gaussian impact velocity field centered at (cx, cy).
void initImpactVelocityField(
    akantu::Mesh &mesh, akantu::SolidMechanicsModelCohesive &model,
    akantu::Real v0, akantu::Real kappa,
    std::pair<akantu::Real, akantu::Real> center = {0.0, 0.0},
    akantu::Real z_sign = 1.0,
    std::optional<akantu::Real> cutoff = std::nullopt,
    const std::string &shape = "gaussian",
    const std::pair<akantu::Real, akantu::Real> &angles = {0.0, 0.0});

/// Dump per-step data to HDF5 (fragments + energies)
void dumpResultsH5(akantu::SolidMechanicsModelCohesive &model,
	       akantu::FragmentManager & fragments,	int n,
                   akantu::Real dt, akantu::Real cumulative_work,
                   const std::string &h5_file = "../output/tmp/data.h5");

/// Save configuration file with all the initialization parameters for reproducibility
void saveConfigFile(const Args &args, const std::string &outpath);
