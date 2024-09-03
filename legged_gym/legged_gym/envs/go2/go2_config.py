from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2RoughCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 238
        symmetric = False  #True :  set num_privileged_obs = None;    false: num_privileged_obs = observations + 187 ,set "terrain.measure_heights" to true
        num_privileged_obs = 238
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 200 # episode length in seconds


        history_encoding = True
        reorder_dofs = True
        # additional visual inputs 
        include_foot_contacts = True
        
        randomize_start_pos = False
        randomize_start_vel = False
        randomize_start_yaw = False
        rand_yaw_range = 1.2
        randomize_start_y = False
        rand_y_range = 0.5
        randomize_start_pitch = False
        rand_pitch_range = 1.6

        contact_buf_len = 100

        next_goal_threshold = 0.2
        reach_goal_delay = 0.1
        num_future_goal_obs = 2
        coordinates = [
            [6.0, 7.0, 3.0],
            [6.0, 7.0, 0.0],  # 第一列的三维坐标
            [7.0, 6.0, 0.0],  # 第二列的三维坐标
            [8.0, 7.0, 0.0],  # 第三列的三维坐标
            [9.0, 6.0, 0.0],  # 第四列的三维坐标
            [9.0, 7.0, 0.0],  # 第四列的三维坐标
        ]
        lookat_id=0


    class terrain( LeggedRobotCfg.env ):
        mesh_type = 'competition' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.25 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 12.
        terrain_width = 12.
        num_rows= 9 # number of terrain rows (levels)
        num_cols = 1 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0,1, 0, 0, 0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        #TODO modify the number of goals
        num_goals = 8

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
    class commands( LeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.8, 0.8] # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
            ang_vel_yaw = [0., 0.]    # min max [rad/s]
            heading = [0, 0]

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 50.}  # [N*m/rad]
        damping = {'joint': 2}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True
    class domain_rand:
        randomize_friction = False
        friction_range = [0.2, 1.5]
        randomize_base_mass = False
        added_mass_range = [-4., 4.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

        randomize_base_com = False
        added_com_range = [-0.15, 0.15]

        randomize_motor = False
        motor_strength_range = [0.8, 1.2]
        action_buf_len = 8

    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = -0.00
            ang_vel_xy = -0.0
            orientation = -0.0
            torques = -0.000
            dof_vel = -0.0
            dof_acc = -0.0
            base_height = -0.000
            feet_air_time = 0.0
            collision = -0.0
            feet_stumble = -0.0
            action_rate = -0.0
            stand_still = -0.0
            dof_pos_limits = -0.0
            stagnation=-0.0
            limbo = -0.0
            goal_pos = 0.0
            out_mid = -0.0
            move_back = -0.0

            # tracking rewards
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5
            # regularization rewards
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.
            dof_acc = -2.5e-7
            collision = -10.
            action_rate = -0.1
            delta_torques = -1.0e-7
            torques = -0.00001
            hip_pos = -0.5
            dof_error = -0.04
            feet_stumble = -1
            # feet_edge = -1

            # termination = -0.0
            # tracking_lin_vel = 2.0
            # tracking_ang_vel = 0.5
            # lin_vel_z = -0.001
            # ang_vel_xy = -0.05
            # orientation = -1.0
            # torques = -0.0002
            # dof_vel = -0.0
            # dof_acc = -2.5e-07
            # base_height = -0.0001
            # feet_air_time = 1.0
            # collision = -1.0
            # feet_stumble = -0.0
            # action_rate = -0.01
            # stand_still = -0.0
            # dof_pos_limits = -10.0
            # stagnation=-0.0
            # limbo = -0.0
            # goal_pos = 0.0
            # out_mid = -0.0
            # move_back = -0.0

# step 1 
# negtive reward -> -0.001



# tracking_lin_vel 0.02
# dof_pos_limits -0.1   -10  -> -1 : -0.01 


# reward 100
# tracking_lin_vel 0.9

# base_height = -0.01 -> base_height = -0.05
# orientation =-0.0001  orientation= -0.1

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.3
        max_contact_force = 100. # forces above this value are penalized
        goal_position_x = 46 #TODO

class Go2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = ''
        model_dir = None
  
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 48 # per iteration
        max_iterations = 6000 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations

        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt