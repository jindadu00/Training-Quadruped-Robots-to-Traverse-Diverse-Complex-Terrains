# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg, dir_log = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True, dir_log=dir_log)
    log_file_path = os.path.join(dir_log, "rewards_settings.txt")
    print("-----------------------------------")
    with open(log_file_path, "w") as f:
        def print_and_log(message):
            print(message)
            f.write(message + "\n")
        print_and_log("termination: " + str(env_cfg.rewards.scales.termination))
        print_and_log("tracking_lin_vel: " + str(env_cfg.rewards.scales.tracking_lin_vel))
        print_and_log("tracking_ang_vel: " + str(env_cfg.rewards.scales.tracking_ang_vel))
        print_and_log("lin_vel_z: " + str(env_cfg.rewards.scales.lin_vel_z))
        print_and_log("ang_vel_xy: " + str(env_cfg.rewards.scales.ang_vel_xy))
        print_and_log("orientation: " + str(env_cfg.rewards.scales.orientation))
        print_and_log("torques: " + str(env_cfg.rewards.scales.torques))
        print_and_log("dof_vel: " + str(env_cfg.rewards.scales.dof_vel))
        print_and_log("dof_acc: " + str(env_cfg.rewards.scales.dof_acc))
        print_and_log("base_height: " + str(env_cfg.rewards.scales.base_height))
        print_and_log("feet_air_time: " + str(env_cfg.rewards.scales.feet_air_time))
        print_and_log("collision: " + str(env_cfg.rewards.scales.collision))
        print_and_log("feet_stumble: " + str(env_cfg.rewards.scales.feet_stumble))
        print_and_log("action_rate: " + str(env_cfg.rewards.scales.action_rate))
        print_and_log("stand_still: " + str(env_cfg.rewards.scales.stand_still))
        print_and_log("dof_pos_limits: " + str(env_cfg.rewards.scales.dof_pos_limits))
        print_and_log("base_height_target: " + str(env_cfg.rewards.base_height_target))
        print_and_log("base_height_target: " + str(env_cfg.rewards.limbo))
    print("-----------------------------------")

if __name__ == '__main__':
    args = get_args()
    train(args)
