#include "helpers.hh"
#include "h5_utils.hh"

#include "fragment_manager.hh"

#include <H5public.h>
#include <hdf5.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

/* -------------------------------------------------------------------------- */
/* parseArguments                                                             */
/* -------------------------------------------------------------------------- */

namespace {
static void print_help(const char *prog) {
  std::cout
      << "Usage: " << prog << " [options]\n\n"
      << "Options:\n"
      << "  --material_file, -mat <file>   Material file name (default: "
         "AD995_cohesive_contact_m10_stable.dat)\n"
      << "  --mesh_file,     -msh <file>   Mesh file name (default: "
         "plate_0.01x0.01_npz1_P1.msh)\n"
      << "  --strain_rate,   -sr  <real>   Strain rate (default: 2.559e4)\n"
      << "  --velocity,      -v   <real>   Initial velocity (default: 10.0)\n"
      << "  --safety_factor, -t   <real>   CFL safety factor (default: 0.2)\n"
      << "  --time,          -T   <real>   Final time (default: 0.0)\n"
      << "  --help,          -h            Show this message and exit\n";
}

inline akantu::Real to_real(const char *s) {
  return static_cast<akantu::Real>(std::atof(s));
}
} // namespace

Args parseArguments(int argc, char *argv[]) {
  Args args; // defaults defined in header

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need_value = [&](const char *name) {
      if (i + 1 >= argc) {
        std::cerr << "Error: option " << name << " requires a value.\n";
        print_help(argv[0]);
        std::exit(EXIT_FAILURE);
      }
    };

    if (a == "--help" || a == "-h") {
      print_help(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else if (a == "--material_file" || a == "-mat") {
      need_value(a.c_str());
      args.material_file = argv[++i];
    } else if (a == "--mesh_file" || a == "-msh") {
      need_value(a.c_str());
      args.mesh_file = argv[++i];
    } else if (a == "--strain_rate" || a == "-sr") {
      need_value(a.c_str());
      args.strain_rate = to_real(argv[++i]);
    } else if (a == "--velocity" || a == "-v") {
      need_value(a.c_str());
      args.velocity = to_real(argv[++i]);
    } else if (a == "--safety_factor" || a == "-t") {
      need_value(a.c_str());
      args.safety_factor = to_real(argv[++i]);
    } else if (a == "--time" || a == "-T") {
      need_value(a.c_str());
      args.time = to_real(argv[++i]);
    } else {
      std::cerr << "Warning: ignoring unknown option '" << a << "'.\n";
    }
  }

  return args;
}

/* -------------------------------------------------------------------------- */
/* initParaviewDumpers                                                        */
/* -------------------------------------------------------------------------- */

void initParaviewDumpers(akantu::SolidMechanicsModelCohesive &model,
                         const std::string &outpath) {
  using namespace akantu;
  model.setBaseName("tension");
  model.setDirectory(outpath);

  // Bulk
  model.addDumpField("displacement");
  model.addDumpField("external_force");
  model.addDumpField("internal_force");
  model.addDumpField("velocity");
  // If your Akantu expects "gradu" instead of "grad_u", switch this.
  model.addDumpField("grad_u");
  model.addDumpField("stress");

  // Cohesive facets
  model.setBaseNameToDumper("cohesive elements", "cohesive");
  model.addDumpFieldToDumper("cohesive elements", "displacement");
  model.addDumpFieldToDumper("cohesive elements", "damage");
  model.addDumpFieldToDumper("cohesive elements", "tractions");
  model.addDumpFieldToDumper("cohesive elements", "opening");
}

/* -------------------------------------------------------------------------- */
/* setupDir                                                                   */
/* -------------------------------------------------------------------------- */

std::pair<std::string, std::string> setupDir(const std::string &nname,
                                             const Args &args, int prank) {

  namespace fs = std::filesystem;

  const fs::path inpath = fs::absolute(fs::path(__FILE__))
                              .parent_path()
                              .parent_path()
                              .parent_path();

  if (prank == 0) {
    std::cout << "Input path: " << inpath << "\n";
  }

  fs::path outpath;
  if (nname == "lsmspc19") {
    outpath = inpath / "output" / "local";
  } else {
    outpath = fs::path("/scratch/saporito/Frag3D");
  }

  auto fmt = [](akantu::Real x) {
    std::ostringstream oss;
    oss.setf(std::ios::scientific);
    oss.precision(1);
    oss << x;
    return oss.str();
  };

  outpath /= ("impact_vel_" + fmt(args.velocity) + "_safety_factor_" +
              fmt(args.safety_factor) + "_time" + fmt(args.time)) +
             "kappa40";

  if (prank == 0) {
    try {
      if (fs::exists(outpath))
        fs::remove_all(outpath);
      fs::create_directories(outpath);
      std::cout << "Created output directory: " << outpath << "\n";
    } catch (const std::exception &e) {
      std::cerr << "Error preparing output directory '" << outpath
                << "': " << e.what() << "\n";
      std::exit(EXIT_FAILURE);
    }
  }

  return {(inpath.string() + fs::path::preferred_separator),
          (outpath.string() + fs::path::preferred_separator)};
}

/* -------------------------------------------------------------------------- */
/* initImpactVelocityField                                                    */
/* -------------------------------------------------------------------------- */

void initImpactVelocityField(akantu::Mesh &mesh,
                             akantu::SolidMechanicsModelCohesive &model,
                             akantu::Real v0, akantu::Real kappa,
                             std::pair<akantu::Real, akantu::Real> center,
                             akantu::Real z_sign,
                             std::optional<akantu::Real> cutoff) {
  using namespace akantu;

  auto &vel = model.getVelocity(); // Array<Real> [nb_nodes x dim]
  auto &nodes = mesh.getNodes();   // Array<Real> [nb_nodes x dim]

  const auto &lower = mesh.getLowerBounds(); // Vector<Real>
  const auto &upper = mesh.getUpperBounds(); // Vector<Real>
  const Real L = upper(0) - lower(0);

  const Real sigma = kappa * L / 2.0;
  const Real inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);

  const Real cx = center.first;
  const Real cy = center.second;

  const UInt nb_nodes = mesh.getNbNodes();
  const UInt dim = mesh.getSpatialDimension();
  AKANTU_DEBUG_ASSERT(dim >= 2, "Expected spatial dimension >= 2");

  for (UInt i = 0; i < nb_nodes; ++i) {
    const Real dx = nodes(i, 0) - cx;
    const Real dy = nodes(i, 1) - cy;
    const Real r2 = dx * dx + dy * dy;

    Real vz = 0.0;
    if (!cutoff || r2 <= (*cutoff) * (*cutoff)) {
      vz = z_sign * v0 * std::exp(-r2 * inv_two_sigma2);
    }

    // enforce z-only impact
    vel(i, 0) = 0.0;
    vel(i, 1) = 0.0;
    vel(i, 2) = vz;
  }
  // Eccentricity (change cx and cy in dx, dy), Direction of velocity (change vx or/and vy to nonzero),
  // different options for shape
  // Synchronize velocities across ghost nodes
  // model.synchronize(SynchronizationTag::_velocity);
}

/* -------------------------------------------------------------------------- */
/* dumpResultsH5                                                              */
/* -------------------------------------------------------------------------- */

void dumpResultsH5(akantu::Mesh &mesh, akantu::SolidMechanicsModelCohesive &model, int n,
                   akantu::Real dt, akantu::Real cumulative_work,
                   const std::string &h5_file) {
  using namespace akantu;

  // Energies (if your Akantu expects enums, replace string keys accordingly)
  const Real epot = model.getEnergy("potential");
  const Real ekin = model.getEnergy("kinetic");
  const Real edis = model.getEnergy("dissipated");
  const Real erev = model.getEnergy("reversible");
  const Real econ = model.getEnergy("cohesive contact");

  const Real work = cumulative_work;
  const Real total_energy = epot + ekin + edis + erev + econ - work;

  // //Mass: [nb_frag x 1] -> 1D
  // const auto &mass = fragments.getMass();
  // std::vector<double> frag_mass;
  // frag_mass.reserve(nb_frag);
  // for (int i = 0; i < nb_frag; ++i)
  //  frag_mass.push_back(mass(i, 0));

  // //Velocity: [nb_frag x dim]
  // const auto &vel = fragments.getVelocity();
  // const auto dim = static_cast<int>(vel.getNbComponent());
  // std::vector<double> frag_vel;
  // frag_vel.resize(nb_frag * dim);
  // for (int i = 0; i < nb_frag; ++i)
  //  for (int d = 0; d < dim; ++d)
  //    frag_vel[i * dim + d] = vel(i, d);

  const auto &comm = Communicator::getStaticCommunicator();
  Int prank = comm.whoAmI();

  if (prank == 0) {
    // Fragments
    akantu::FragmentManager fragments(model);
    fragments.computeAllData();
    // We store the number of fragments (UInt getNbFragment() const method)
    const int nb_frag = static_cast<int>(fragments.getNbFragment());
    // We store the mass of fragments (const Array<Real> &getMass() const method)
    const auto &frag_mass = fragments.getMass();
    // We store the velocity of fragments (const Array<Real> &getVelocity() const method)
    const auto &frag_vel = fragments.getVelocity();

    // HDF5 write with small retry (file contention)
    for (int attempt = 0; attempt < 5; ++attempt) {
      hid_t fid = h5util::open_or_create_file(h5_file);
      if (fid < 0) {
        std::cerr << "HDF5: failed to open file (attempt " << attempt + 1
                  << ")\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }

      const std::string step_name = "step_" + std::to_string(n);
      hid_t gid = h5util::recreate_group(fid, step_name);
      if (gid < 0) {
        std::cerr << "HDF5: failed to create group '" << step_name << "'\n";
        H5Fclose(fid);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }

      if (!frag_mass.empty())
         h5util::write_dataset_1d(gid, "fragment_mass", frag_mass.data(),
                                  static_cast<hsize_t>(frag_mass.size()));
      if (!frag_vel.empty())
         h5util::write_dataset_2d(gid, "fragment_velocity", frag_vel.data(),
                                  static_cast<hsize_t>(nb_frag),
                                  static_cast<hsize_t>(mesh.getSpatialDimension()));

      h5util::write_attr_int(gid, "nb_fragments", nb_frag);
      h5util::write_attr_double(gid, "time", static_cast<double>(n * dt));
      h5util::write_attr_double(gid, "epot", static_cast<double>(epot));
      h5util::write_attr_double(gid, "ekin", static_cast<double>(ekin));
      h5util::write_attr_double(gid, "edis", static_cast<double>(edis));
      h5util::write_attr_double(gid, "erev", static_cast<double>(erev));
      h5util::write_attr_double(gid, "econ", static_cast<double>(econ));
      h5util::write_attr_double(gid, "work", static_cast<double>(work));
      h5util::write_attr_double(gid, "total_energy",
                                static_cast<double>(total_energy));

      H5Gclose(gid);
      H5Fflush(fid, H5F_SCOPE_GLOBAL);
      H5Fclose(fid);
      return; // success
    }

    std::cerr
        << "Oh Oh, file is locked or unavailable after multiple retries.\n";
  }
}
