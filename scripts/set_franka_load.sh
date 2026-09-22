#!/usr/bin/env bash
# Declare the tool mass to the Franka, so its internal gravity compensation
# accounts for the FT sensor + coupling + gripper.
#
# WHY: control/robot_server.py runs the impedance controllers with
# `use_gravity_compensation: false` -- crisp adds no gravity term of its own
# and relies entirely on what the robot has been told it is carrying. With
# the load undeclared the arm sits below every commanded pose. Measured on
# this cell with deploy/deploy_spacemouse.py --test: a -5.6 mm z bias on all
# six probes, i.e. ~2.8 N, i.e. ~0.29 kg unaccounted for -- almost exactly
# the FT 300's 0.3 kg.
#
# The values are the stack from the flange: Robotiq FT 300 (0.300 kg,
# 37.5 mm) + coupling (~0.1 kg) + Robotiq 2F-85 (0.900 kg, ~162 mm). They are
# datasheet figures, NOT measured on this robot -- see the check at the end.
#
# libfranka refuses setLoad while a controller is running, so deactivate
# first. The service reports success:false, error:"command exception error"
# if you forget.
set -euo pipefail

MASS="${MASS:-1.300}"
COM="${COM:-[0.0, 0.0, 0.0855]}"
# About the COM. Cylinder r=40mm h=210mm; only mass and COM matter for
# gravity, the inertia refines the dynamics terms.
INERTIA="${INERTIA:-[0.00527, 0.0, 0.0, 0.0, 0.00527, 0.0, 0.0, 0.0, 0.00104]}"
SERVICE="${SERVICE:-/service_server/set_load}"

echo "Declaring load: mass=${MASS} kg  com=${COM}"
echo "service: ${SERVICE}"
echo
echo "If this reports 'command exception error', a controller is still"
echo "active. Deactivate it, rerun, then reactivate:"
echo "    ros2 control switch_controllers --deactivate cartesian_impedance_controller"
echo

ros2 service call "${SERVICE}" franka_msgs/srv/SetLoad \
  "{mass: ${MASS}, center_of_mass: ${COM}, load_inertia: ${INERTIA}}"

echo
echo "Now re-measure:  python -m deploy.deploy_spacemouse --test --yes"
echo "The z bias should collapse from ~-5.6 mm toward the ~+/-2 mm stiction floor."
