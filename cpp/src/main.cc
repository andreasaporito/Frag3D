// main.cc - Parallel version
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits.h>
#include <optional>
#include <string>
#include <unistd.h>
#include <utility>

#include "communicator.hh" // Add for parallel support
#include "mesh.hh"
#include "solid_mechanics_model_cohesive.hh"
// header-only helpers for HDF5
#include "h5_utils.hh"

namespace akantu {
using Real = double;
}
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

Args parseArguments(int argc, char *argv[]);

void initParaviewDumpers(akantu::SolidMechanicsModelCohesive &model,
                         const std::string &outpath);

std::pair<std::string, std::string> setupDir(const std::string &nname,
                                             const Args &args, int prank);

void initImpactVelocityField(
    akantu::Mesh &mesh, akantu::SolidMechanicsModelCohesive &model,
    akantu::Real v0, akantu::Real kappa,
    std::pair<akantu::Real, akantu::Real> center = {0.0, 0.0},
    akantu::Real z_sign = 1.0,
    std::optional<akantu::Real> cutoff = std::nullopt,
    const std::string &shape = "gaussian",
    const std::pair<akantu::Real, akantu::Real> &angles = {0.0, 0.0});

#include "fragment_manager.hh"

void dumpResultsH5(akantu::Mesh &mesh, akantu::SolidMechanicsModelCohesive &model,akantu::FragmentManager & fragments, int n,
                   akantu::Real dt, akantu::Real cumulative_work,
                   const std::string &h5_file = "../output/tmp/data.h5");

void saveConfigFile(const Args &args, const std::string &outpath);

// ---- tiny helpers
static inline std::string detect_hostname() {
  char buf[256] = {0};
  if (gethostname(buf, sizeof(buf)) == 0 && buf[0] != '\0')
    return std::string(buf);
  if (const char *h = std::getenv("HOSTNAME"))
    return std::string(h);
  if (const char *h = std::getenv("HOST"))
    return std::string(h);
  return {}; // unknown host
}

int main(int argc, char *argv[]) {
  using namespace akantu;

  // 1) args & dirs -----------------------------------------------------------
  Args args = parseArguments(argc, argv);

  static char aka_seed[] = "AKA_SEED=1";
  putenv(aka_seed);

  initialize(args.material_file, argc, argv);

  // Get the communicator and process rank
  const auto &comm = Communicator::getStaticCommunicator();
  Int prank = comm.whoAmI();
  Int psize = comm.getNbProc();

  // Only rank 0 prints initialization messages
  if (prank == 0) {
    const std::string host = detect_hostname();
    std::cout << "Running on host: '" << host << "'\n"
              << "Running with " << psize << " MPI processes\n"
              << "Reading material from: " << args.material_file << "\n"
              << std::flush;
  }

  const auto [inpath, outpath] = setupDir(detect_hostname(), args, prank);
  comm.barrier(); // ensure dir is created before proceeding
  saveConfigFile(args, outpath);

  // 2) mesh & model ----------------------------------------------------------
  const Int dim = 3;
  Mesh mesh(dim);

  // Only rank 0 reads the mesh file
  if (prank == 0) {
    const std::string mesh_path = args.mesh_file;
    mesh.read(mesh_path);
    std::cout << "Reading mesh from: " << mesh_path << "\n";
  }

  // Distribute mesh among all processors
  mesh.distribute();

  // Create the cohesive model
  SolidMechanicsModelCohesive model(mesh);

  // Initialize the model with explicit time integration
  model.initFull(_analysis_method = _explicit_lumped_mass,
                 _is_extrinsic = true);
  if (prank == 0) {
    std::cout << "Initialized model with cohesive elements...\n";
  }

  // Update automatic insertion
  model.updateAutomaticInsertion();

  akantu::FragmentManager fragment_manager(model);

  // 3) dumpers & ICs ---------------------------------------------------------
  initParaviewDumpers(model, outpath);

  // Example: center a Gaussian impact at (0,0) with width kappa = 0.2 * L/2
  // Adjust kappa/center/cutoff to your case.
  initImpactVelocityField(mesh, model,
                          /*v0=*/args.velocity,
                          /*kappa=*/args.kappa,
                          /*center=*/{args.center_x, args.center_y},
                          /*z_sign=*/args.z_sign,
                          /*cutoff=*/args.cutoff,
                          /*shape=*/args.shape,
                          {/*xy_angle=*/args.angle_xz, /*z_angle=*/args.angle_yz});

  // 4) time integration setup ------------------------------------------------
  Real dt = model.getStableTimeStep() * args.safety_factor;
  model.setTimeStep(dt);

  const Real T_end = args.time;
  const int n_steps = static_cast<int>(std::ceil(T_end / dt));

  if (prank == 0) {
    std::cout << "Stable time step: " << dt << "\n";
    std::cout << "Total steps: " << n_steps << " for T=" << T_end << "\n";
  }

  Real cumulative_work = 0.0;

  // 5) main loop -------------------------------------------------------------
  const int dump_stride_paraview = std::max(1, n_steps / 500);
  const int dump_stride_h5 = std::max(1, n_steps / 1000);

  for (int n = 0; n < n_steps; ++n) {
    // Check cohesive stress (parallel operation with ghost synchronization)
    model.checkCohesiveStress();
    // Solve one explicit step(parallel with automatic ghost communication)
    model.solveStep("explicit_lumped");

    // Dump results (parallel I/O)
    if (n % dump_stride_paraview == 0) {
      if (prank == 0) {
        std::cout << "Step " << n << " / " << n_steps << "\n" << std::flush;
      }

      // Each process writes its partition data
      model.dump();                    // bulk elements
      model.dump("cohesive elements"); // facet dumper (if configured)
    }

    if (n % dump_stride_h5 == 0 || n == n_steps - 1) {
      comm.barrier();
      //std::cout << "Writing this instead of writing into data.h5. \n";
      dumpResultsH5(mesh,model,fragment_manager, n, dt, cumulative_work, outpath + "data.h5");
    }

  }
  // Print final statistics from rank 0
  if (prank == 0) {
    std::cout << "Simulation completed successfully.\n";
  }

  // Finalize Akantu (as well as MPI)
  akantu::finalize();
  return EXIT_SUCCESS;
}
