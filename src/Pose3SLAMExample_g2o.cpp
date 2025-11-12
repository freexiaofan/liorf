/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file Pose3SLAMExample_g2o.cpp
 * @brief A 3D Pose SLAM example that reads input from g2o, and initializes the Pose3 using InitializePose3
 * Syntax for the script is ./Pose3SLAMExample_g2o input.g2o output.g2o [timestamps.tum]
 * The first pose is fixed during optimization. If timestamps.tum is provided, those timestamps will be used in output.
 * @date Aug 25, 2014
 * @author Luca Carlone
 */

#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

using namespace std;
using namespace gtsam;

// Function to read timestamps from TUM file
map<Key, double> readTUMTimestamps(const string& tumFile) {
    map<Key, double> timestamps;
    ifstream file(tumFile);
    
    if (!file.is_open()) {
        cerr << "Warning: Cannot open TUM file for timestamps: " << tumFile << endl;
        return timestamps;
    }
    
    cout << "Reading timestamps from TUM file: " << tumFile << endl;
    
    string line;
    Key currentKey = 0;
    
    while (getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        istringstream iss(line);
        double timestamp, tx, ty, tz, qx, qy, qz, qw;
        
        if (iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
            timestamps[currentKey] = timestamp;
            currentKey++;
        }
    }
    
    file.close();
    cout << "Read " << timestamps.size() << " timestamps from TUM file" << endl;
    return timestamps;
}

// Function to save trajectory in TUM format with optional timestamps
bool saveTUMTrajectory(const Values& values, const string& filename, const map<Key, double>& timestamps = map<Key, double>()) {
    cout << "Saving trajectory in TUM format to: " << filename << endl;
    
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open file: " << filename << endl;
        return false;
    }
    
    // Write TUM format header
    file << "# TUM trajectory format" << endl;
    file << "# timestamp tx ty tz qx qy qz qw" << endl;
    
    // Sort poses by key for consistent output
    map<Key, Pose3> sorted_poses;
    for (const auto& key_value : values) {
        if (values.exists<Pose3>(key_value.key)) {
            sorted_poses[key_value.key] = values.at<Pose3>(key_value.key);
        }
    }
    
    cout << "Writing " << sorted_poses.size() << " poses to TUM file..." << endl;
    bool using_external_timestamps = !timestamps.empty();
    if (using_external_timestamps) {
        cout << "Using external timestamps from TUM file" << endl;
    } else {
        cout << "Using key values as timestamps" << endl;
    }
    
    for (const auto& pair : sorted_poses) {
        const Pose3& pose = pair.second;
        Vector3 translation = pose.translation();
        gtsam::Quaternion rotation = pose.rotation().toQuaternion();
        
        // Use external timestamp if available, otherwise use key as timestamp
        double timestamp;
        if (using_external_timestamps && timestamps.find(pair.first) != timestamps.end()) {
            timestamp = timestamps.at(pair.first);
        } else {
            timestamp = static_cast<double>(pair.first);
        }
        
        // Write in TUM format: timestamp tx ty tz qx qy qz qw
        file << fixed << setprecision(3)
             << timestamp << " "
             << setprecision(6) 
             << translation.x() << " " << translation.y() << " " << translation.z() << " "
             << rotation.x() << " " << rotation.y() << " " << rotation.z() << " " << rotation.w()
             << endl;
    }
    
    file.close();
    cout << "Successfully saved TUM trajectory with " << sorted_poses.size() << " poses!" << endl;
    return true;
}

int main(const int argc, const char* argv[]) {
  // Check arguments
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " input.g2o output.g2o [timestamps.tum]" << endl;
    cout << "  input.g2o      - Input g2o graph file" << endl;
    cout << "  output.g2o     - Output optimized g2o file" << endl;
    cout << "  timestamps.tum - Optional TUM file with timestamps to use in output" << endl;
    // /home/tyjt/Desktop/ros_ws/devel/lib/liorf/liorf_Pose3SLAMExample_g2o  /mnt/nvme0n1p2/data/data_liorf/graph.g2o /mnt/nvme0n1p2/data/data_liorf/graph_opt.g2o   /mnt/nvme0n1p2/data/data_liorf/geo_key_pose_opt.tum
    return -1;
  }
  

  // Read graph from file
  string g2oFile = argv[1];

  NonlinearFactorGraph::shared_ptr graph;
  Values::shared_ptr initial;
  bool is3D = true;
  boost::tie(graph, initial) = readG2o(g2oFile, is3D);

  // Add strong prior on the first key to prevent it from moving during optimization
  // Using very small variances to effectively fix the first pose
  auto priorModel = noiseModel::Diagonal::Variances(
      (Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished());
  Key firstKey = 0;
  Pose3 firstPose;
  for (const auto key_value : *initial) {
    std::cout << "Adding strong prior to first pose (key=" << key_value.key << ") to fix it during optimization" << std::endl;
    firstKey = key_value.key;
    firstPose = key_value.value.cast<Pose3>();
    // Use the actual initial pose instead of identity
    graph->addPrior(firstKey, firstPose, priorModel);
    std::cout << "First pose fixed at: " << std::endl;
    firstPose.print("  First Pose: ");
    break;
  }

  std::cout << "Optimizing the factor graph" << std::endl;
  GaussNewtonParams params;
  params.setVerbosity("TERMINATION");  //  show info about stopping conditions
  GaussNewtonOptimizer optimizer(*graph, *initial, params);
  Values result = optimizer.optimize();
  std::cout << "Optimization complete" << std::endl;

  std::cout << "initial error=" << graph->error(*initial) << std::endl;
  std::cout << "final error=" << graph->error(result) << std::endl;

  // Verify first pose hasn't moved
  if (result.exists(firstKey)) {
    Pose3 optimizedFirstPose = result.at<Pose3>(firstKey);
    std::cout << "\n=== First Pose Verification ===" << std::endl;
    std::cout << "Original first pose:" << std::endl;
    firstPose.print("  Original: ");
    std::cout << "Optimized first pose:" << std::endl;
    optimizedFirstPose.print("  Optimized: ");
    
    // Check if they are approximately equal
    double translation_diff = (optimizedFirstPose.translation() - firstPose.translation()).norm();
    double rotation_diff = optimizedFirstPose.rotation().between(firstPose.rotation()).matrix().trace();
    std::cout << "Translation difference: " << translation_diff << std::endl;
    std::cout << "Rotation trace difference: " << rotation_diff << std::endl;
    std::cout << "First pose " << (translation_diff < 1e-10 ? "successfully fixed" : "moved during optimization") << std::endl;
    std::cout << "==============================\n" << std::endl;
  }

  if (argc < 3) {
    result.print("result");
  } else {
    const string outputFile = argv[2];
    std::cout << "Writing results to file: " << outputFile << std::endl;
    
    // Write the complete graph (including constraints) with optimized values
    std::cout << "Including " << graph->size() << " factors (constraints + priors) in output" << std::endl;
    writeG2o(*graph, result, outputFile);
    std::cout << "Successfully written optimized graph with all constraints to: " << outputFile << std::endl;
    
    // Read timestamps from TUM file if provided
    map<Key, double> external_timestamps;
    if (argc >= 4) {
        string timestampFile = argv[3];
        std::cout << "Reading timestamps from: " << timestampFile << std::endl;
        external_timestamps = readTUMTimestamps(timestampFile);
    }
    
    // Save TUM trajectory with timestamps
    string tumFile = outputFile.substr(0, outputFile.find_last_of('.')) + "_trajectory.tum";
    if (saveTUMTrajectory(result, tumFile, external_timestamps)) {
        std::cout << "TUM trajectory saved to: " << tumFile << std::endl;
        if (!external_timestamps.empty()) {
            std::cout << "Used " << external_timestamps.size() << " external timestamps" << std::endl;
        }
    } else {
        std::cerr << "Failed to save TUM trajectory!" << std::endl;
    }
  }
  return 0;
}
