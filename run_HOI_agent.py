import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import warnings
warnings.filterwarnings("ignore")
import autogen
from typing import Callable, Dict, Literal, Optional, Union
import json
import io
from autogen.runtime_logging import log_new_agent, logging_enabled
from autogen import ConversableAgent
from HOI_agent.prompt.single_agent_prompt import TASK_PROMPT, USER_PROMPT
import contextlib
from HOI_agent.protomotios_generate_multi_object import protomotion_model
from HOI_agent.utils.log_func import episode_logs
from HOI_agent.utils.config_utils import parse_opt
import torch
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

def task_completed():
    print("Taskexec")

class ExecutorAgent(ConversableAgent):
    def __init__(
            self,
            name: str,
            system_message: str,
            execute_model,
            llm_config: Optional[Union[Dict, Literal[False]]] = None,
            is_termination_msg: Optional[Callable[[Dict], bool]] = None,
            max_consecutive_auto_reply: Optional[int] = None,
            human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "NEVER",
            description: Optional[str] = None,
            **kwargs,
        ):
        super().__init__(
                name,
                system_message,
                is_termination_msg,
                max_consecutive_auto_reply,
                human_input_mode,
                llm_config=llm_config,
                description=description,
                **kwargs,
            )
        self.external_functions = {
        "generate_motion": execute_model.generate_human_object,
        "generate_interaction": execute_model.generate_human_object,
        "task_completed": task_completed,
        # "detect_object": execute_model.detect_object
        }
        if logging_enabled():
            log_new_agent(self, locals())

        if description is None:
            self.description = self.system_message

    def run_code(self, code, **kwargs):
        exec_context = {}
        if self.external_functions:
            exec_context.update(self.external_functions)
        exec_context.update(kwargs)
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        exit_code = 0

        try:
            # Redirect stdout and stderr to capture outputs
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                for c in code.split("\n\n"):
                    exec(c, exec_context) 
                # exec('\ndetect_object()\n', exec_context)
        except Exception as e:
            exit_code = 1
            stderr_capture.write(f"Error: {e}")

        print(code)
        print(kwargs)
        
        # Get the captured output and error strings
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        # Close the StringIO objects
        stdout_capture.close()
        stderr_capture.close()

        logs = f"{stderr_output}\n{stdout_output}"
        tag = "None"
        return exit_code, logs, f"python:{tag}"

def create_multi_agent(custom_client=None, execute_model=None, max_round=10, llm_config=False):    
    # create user_proxy
    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message="""Give the task, and 
        send instructions to 
        the planner to execute the task in the environment.""",
        code_execution_config=False,

        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "taskexec" in msg["content"].lower()
    )

    planner = autogen.ConversableAgent(
        name="Planner",
        system_message=TASK_PROMPT,
        description="Planner. Given a task, determine what "
        "steps are needed to complete the task. "
        "After each step is done by others, check the progress and "
        "instruct the remaining steps",
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "taskexec" in msg["content"].lower()
    )

    # create executor
    executor_agent = ExecutorAgent(
        name="Executor",
        system_message="""System message""",
        execute_model=execute_model,
        code_execution_config={
            "last_n_messages": 3,
            "use_docker": False},
        human_input_mode="NEVER",
        llm_config=llm_config,
        default_auto_reply= "Please continue. If everything is done, execute task_completed().",
        is_termination_msg=lambda msg: "taskexec" in msg["content"].lower()
    )


    groupchat = autogen.GroupChat(
        agents=[user_proxy, planner, executor_agent], 
        messages=[], 
        max_round=max_round,
        allowed_or_disallowed_speaker_transitions={
            user_proxy: [planner],
            planner: [executor_agent],
            executor_agent: [planner],
        },
        speaker_transitions_type="allowed",
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        name='Supervisor',
        system_message=USER_PROMPT,
        llm_config=llm_config,
    )
    
    return user_proxy, manager

used_data = {
    'sub17_floorlamp_001_pidx_0.json',
    'sub16_clothesstand_004_pidx_10.json',
}

def run_agent(opt):
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    if opt.executor == 'maskedmimic':
        execute_model = protomotion_model(device, opt.save_folder, batch_size=1, last_k_frame=opt.last_k_frame, vis_fps=opt.fps, opt=opt)
    else:
        raise Exception

    llm_config = {"model": opt.llm_model_name, "api_key": OPENAI_API_KEY, "cache_seed": None}
    max_round = 20
    user_proxy, manager = create_multi_agent(execute_model=execute_model, max_round=max_round, llm_config=llm_config)
    
    json_data_folder = os.path.join(opt.data_folder, "environment_state")
    data_list = os.listdir(json_data_folder)
    for data_name in data_list:

        if data_name not in used_data:
            continue
        
        state = json.load(open(os.path.join(json_data_folder, data_name), 'r'))
        data_name, pidx = data_name.replace('.json', '').split('_pidx_')
        scene_name = json_data_folder.split('/')[-2]
        execute_model.set_data_name(data_name, pidx, scene_name)
        
        # set up logger
        agent_logger, _, cleanup = episode_logs(save_folder=opt.save_folder, scene_name=scene_name, seq_name=data_name, pidx=pidx)

        # parse scene info
        target_pos = state['target']
        if opt.navigation_only:
            if len(target_pos) == 1:
                command = f"Walk next to {target_pos[0]}."
            else:
                command = f"Walk next to {target_pos[0]} and {target_pos[1]}."
        elif opt.multi_object_together:
            if len(target_pos) == 1:
                command = f"Pick up {state['objects'][0]['name']} and {state['objects'][1]['name']} at the same time, and put them next to {target_pos[0]}."
            else:
                command = f"Pick up {state['objects'][0]['name']} and {state['objects'][1]['name']} at the same time, and put them next to {target_pos[0]} and {target_pos[1]}."
        else:
            if len(target_pos) == 1:
                command = f"Pick up {state['objects'][0]['name']}, and put it next to {target_pos[0]}."
            else:
                command = f"Pick up {state['objects'][0]['name']}, and put it next to {target_pos[0]} and {target_pos[1]}."
        CURRENT_HUMAN_ENVIRONMENT = ""
        ee_start_pelvis_position = state['human']['pelvis']
        ee_start_left_hand_position = state['human']['left_hand']
        ee_start_right_hand_position = state['human']['right_hand']

        ee_start_orientation = state['human']['orientation']
        CURRENT_HUMAN_ENVIRONMENT += f"pelvis position: {ee_start_pelvis_position}\n"
        CURRENT_HUMAN_ENVIRONMENT += f"left hand position: {ee_start_left_hand_position}\n"
        CURRENT_HUMAN_ENVIRONMENT += f"right hand position: {ee_start_right_hand_position}\n"
        CURRENT_HUMAN_ENVIRONMENT += f"orientation: {ee_start_orientation}\n"

        CURRENT_STATE_ENVIRONMENT = ""
        for obj in state['objects']:
            CURRENT_STATE_ENVIRONMENT += f"***{obj['name']}***:\n" + f"position: {obj['position']}\n" + f"orientation: {obj['orientation']}\n" + f"sizes: {obj['sizes']}\n"
        task_prompt = USER_PROMPT.replace("[HUMAN_STATE]", str(CURRENT_HUMAN_ENVIRONMENT)).replace("[INSERT TASK]", command).replace("[STATE]", CURRENT_STATE_ENVIRONMENT)
        
        agent_logger.info(f"Task Descreption: {command}")

        user_proxy.initiate_chat(
            manager, 
            message=task_prompt
        )

        execute_model.get_human_object_mesh()
        if not opt.save_obj_only:
            execute_model.render_video()
        cleanup()

if __name__ == "__main__":
    opt = parse_opt()

    run_agent(opt)
    