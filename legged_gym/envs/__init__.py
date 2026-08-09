from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.Go2_MoB.GO2_JUMP.go2_jump_env import GO2_JUMP_Robot
from legged_gym.envs.Go2_MoB.GO2_JUMP.GO2_JUMP_config import GO2_JUMP_Cfg_Yu,GO2_JUMP_PPO_Yu


from legged_gym.envs.Go2_MoB.GO2_Trot.GO2_Trot import GO2_Trot_Robot
from legged_gym.envs.Go2_MoB.GO2_Trot.GO2_Trot_config import GO2_Trot_Cfg_Yu,GO2_Trot_PPO_Yu

from legged_gym.envs.Go2_MoB.GO2_Trot.GO2_Stairs import GO2_Stairs_Robot
from legged_gym.envs.Go2_MoB.GO2_Trot.GO2_Stairs_config import GO2_Stairs_Cfg_Yu,GO2_Stairs_PPO_Yu

from legged_gym.envs.GO2_Flip.GO2_BackFlip.GO2_BackFlip_env import Go2_BackFlip
from legged_gym.envs.GO2_Flip.GO2_BackFlip.GO2_BackFlip_Config import GO2_BackFlip_Cfg_Yu, GO2_BackFlip_PPO_Yu


from .base.legged_robot import LeggedRobot
from .GO2_Stand.GO2_Handstand.Go2_handstand import Go2_stand
from .GO2_Stand.GO2_Leftstand.Go2_handstand import Go2_stand_Robot
from .GO2_Stand.GO2_Fronthandstand.Go2_Fronthandstand import Go2_Fronthandstand

from legged_gym.envs.GO2_Stand.GO2_Handstand.Go2_handstand_Config import GO2Cfg_Handstand,GO2CfgPPO_Handstand
from .GO2_Stand.GO2_Leftstand.Go2_handstand_Config import GO2Cfg_Handstand_Command,GO2CfgPPO_Handstand_Command
from .GO2_Stand.GO2_Fronthandstand.Go2_Fronthandstand_Config import GO2Cfg_Fronthandstand,GO2CfgPPO_Fronthandstand

from legged_gym.utils.task_registry import task_registry


from legged_gym.envs.GO2_Flip.GO2_Spring_Jump.GO2_Spring_Jump_env import GO2_Spring_Jump_Robot
from legged_gym.envs.GO2_Flip.GO2_Spring_Jump.GO2_Spring_Jump_Config import GO2_Spring_Jump_Cfg_Yu, GO2_Spring_Jump_PPO_Yu

from legged_gym.envs.GO2_Mulitpolicy.GO2_TrottoHandstand.GO2_TrottoHandstand  import GO2_TrottoHandstand_Robot
from legged_gym.envs.GO2_Mulitpolicy.GO2_TrottoHandstand.GO2_TrottoHandstand_configT import GO2_TrottoHandstand_Cfg,GO2_TrottoHandstand_PPO

from legged_gym.envs.GO2_Mulitpolicy.GO2_StairstoHandstand.GO2_StairstoHandstand  import GO2_StairstoHandstand_Robot
from legged_gym.envs.GO2_Mulitpolicy.GO2_StairstoHandstand.GO2_StairstoHandstand_configS import GO2_StairstoHandstand_Cfg,GO2_StairstoHandstand_PPO

from legged_gym.envs.GO2_Mulitpolicy.GO2_TrottoJump.GO2_TrottoJump import GO2_TrottoJump_Robot
from legged_gym.envs.GO2_Mulitpolicy.GO2_TrottoJump.GO2_TrottoJump_configT import GO2_TrottoJump_Cfg, GO2_TrottoJump_PPO

from legged_gym.envs.GO2_Mulitpolicy.GO2_JumptoHandstand.GO2_JumptoHandstand  import GO2_JumptoHandstand_Robot
from legged_gym.envs.GO2_Mulitpolicy.GO2_JumptoHandstand.GO2_JumptoHandstand_configJ import GO2_JumptoHandstand_Cfg,GO2_JumptoHandstand_PPO


from legged_gym.envs.GO2_Policyconnection.GO2_TrotConHandstand.GO2_TrotConHandstand import GO2_TrotConHandstand_Robot
from legged_gym.envs.GO2_Policyconnection.GO2_TrotConHandstand.GO2_TrotConHandstand_config import GO2_TrotConHandstand_Cfg, GO2_TrotConHandstand_PPO

from legged_gym.envs.GO2_Policyconnection.GO2_JumpConHandstand.GO2_JumpConHandstand import GO2_JumpConHandstand_Robot
from legged_gym.envs.GO2_Policyconnection.GO2_JumpConHandstand.GO2_JumpConHandstand_config import GO2_JumpConHandstand_Cfg, GO2_JumpConHandstand_PPO


task_registry.register( "go2_trot", GO2_Trot_Robot, GO2_Trot_Cfg_Yu(), GO2_Trot_PPO_Yu())
task_registry.register( "go2_stairs", GO2_Stairs_Robot, GO2_Stairs_Cfg_Yu(), GO2_Stairs_PPO_Yu())
task_registry.register( "go2_jump", GO2_JUMP_Robot, GO2_JUMP_Cfg_Yu(), GO2_JUMP_PPO_Yu())
task_registry.register( "go2_handstand", Go2_stand, GO2Cfg_Handstand(), GO2CfgPPO_Handstand())
task_registry.register( "go2_fronthandstand", Go2_Fronthandstand, GO2Cfg_Fronthandstand(), GO2CfgPPO_Fronthandstand())
task_registry.register( "go2_handstand_command", Go2_stand_Robot, GO2Cfg_Handstand_Command(), GO2CfgPPO_Handstand_Command())
task_registry.register( "go2_spring_jump", GO2_Spring_Jump_Robot, GO2_Spring_Jump_Cfg_Yu(), GO2_Spring_Jump_PPO_Yu())
task_registry.register( "go2_backflip", Go2_BackFlip, GO2_BackFlip_Cfg_Yu(), GO2_BackFlip_PPO_Yu())
task_registry.register( "go2_trottohandstand", GO2_TrottoHandstand_Robot, GO2_TrottoHandstand_Cfg(), GO2_TrottoHandstand_PPO())
task_registry.register( "go2_stairstohandstand", GO2_StairstoHandstand_Robot, GO2_StairstoHandstand_Cfg(), GO2_StairstoHandstand_PPO())
task_registry.register( "go2_trottojump", GO2_TrottoJump_Robot, GO2_TrottoJump_Cfg(), GO2_TrottoJump_PPO())
task_registry.register( "go2_jumptohandstand", GO2_JumptoHandstand_Robot, GO2_JumptoHandstand_Cfg(), GO2_JumptoHandstand_PPO())
task_registry.register( "go2_trotconhandstand", GO2_TrotConHandstand_Robot, GO2_TrotConHandstand_Cfg(), GO2_TrotConHandstand_PPO())
task_registry.register( "go2_jumpconhandstand", GO2_JumpConHandstand_Robot, GO2_JumpConHandstand_Cfg(), GO2_JumpConHandstand_PPO())
print("注册的任务:  ",task_registry.task_classes)
