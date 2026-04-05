import torch
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig

from verl import DataProto
import redis
import json
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        self.task_ready = False
        try:
            self.r = redis.StrictRedis(host='localhost', port=6390, db=0, decode_responses=True)
        except:
            print("*****Redis Error*****")

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [
            resp.split('</execute>')[0] + '</execute>' 
            if '</execute>' in resp 
            else resp
            for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def new_update_rolling_state(self, rollings, cur_responses, next_obs, raw_chats) -> Dict:
        """Update rolling state with new responses and observations."""
        assistant_tokens = pad_sequence([
            self._format_role_message("assistant", r, False) for r in cur_responses
        ], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        user_tokens = pad_sequence([
            self._format_role_message("user", obs, True) for obs in next_obs
        ], batch_first=True, padding_value=self.tokenizer.pad_token_id)

        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            assistant_tokens,
            user_tokens
        ])
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })

    def new_update_right_side(self, right_side: Dict, cur_responses: str, next_obs: str = None) -> Dict:
        """Update right side state."""
        assistant_tokens = pad_sequence([
            self._format_role_message("assistant", r, False) for r in cur_responses
        ], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        if next_obs:
            user_tokens = pad_sequence([
                self._format_role_message("user", obs, False) for obs in next_obs
            ], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        if next_obs != None:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                assistant_tokens,
                user_tokens
            ], pad_to_left=False)
        else:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                assistant_tokens,
            ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        return {'responses': responses[:, :max_len]}

    def _format_role_message(self, role: str, content: str, add_prompt: bool) -> torch.Tensor:
        """Format single message with chat template and tokenize."""
        msg = [{"role": role, "content": content}]
        text = self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=add_prompt)
        target = "<|im_start|>system\nYou are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.<|im_end|>\n"
        text = text.replace(target, "")
        target = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        text = text.replace(target, "")
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """

        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus

        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()

        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        padding_size = num_gpus - remainder
        padded_batch = {}

        for k, v in active_batch.batch.items():
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}

        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output


    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor, task_ids, raw_chats, user_task, ports, training=False) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []]}

        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        self.first_done_lens = {}
        prev_valid_lens = torch.zeros(gen_batch.batch['input_ids'].shape[0])
        advantage_weights = []
        for step in range(self.config.max_turns):
            true_indices = torch.nonzero(active_mask, as_tuple=False).view(-1).tolist()
            print("not done idx", true_indices)
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            next_obs, dones, step_weight = self.execute_predictions(
                task_ids, responses_str, self.tokenizer.pad_token, active_mask, user_task=user_task
            )
            def inplace_check_tasks(tasks, group_size=8, threshold=6):
                num_groups = len(tasks) // group_size
                for i in range(num_groups):
                    start, end = i * group_size, (i + 1) * group_size
                    group = tasks[start:end]
                    if sum(group) >= threshold:
                        tasks[start:end] = [1] * group_size
            if training and step >= 30:
                inplace_check_tasks(dones)
            rollings = self.new_update_rolling_state(
                rollings,
                responses_str,
                next_obs,
                raw_chats
            )
            original_right_side = self.new_update_right_side(
                original_right_side,
                responses_str,
                next_obs
            )

            input_ids = original_right_side['responses']
            pad_token_id = self.tokenizer.pad_token_id
            valid_lens = (input_ids != pad_token_id).sum(dim=1)
            curr_step_lens = (valid_lens - prev_valid_lens).long()
            prev_valid_lens = valid_lens
            over_limit_indices = (curr_step_lens == 0).nonzero(as_tuple=True)[0].tolist()
            for idx in over_limit_indices:
                dones[idx] = 1
            B, _ = input_ids.shape
            step_weight_token_level = []
            for b in range(B):
                if b in true_indices:
                    print("Sample {}, cur_length {}".format(b, curr_step_lens[b]))
                    w = torch.ones(curr_step_lens[b].item(), dtype=torch.float32) * step_weight[b]
                else:
                    w = torch.ones(0, dtype=torch.float32) * step_weight[b]
                step_weight_token_level.append(w)
            advantage_weights.append(step_weight_token_level)

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())

            for i, done in enumerate(dones):
                if done and i not in self.first_done_lens:
                    cur_len = (input_ids[i] != pad_token_id).sum().item()
                    self.first_done_lens[i] = cur_len
                    print("{} task done, response len:{}".format(i, cur_len))
            print("\n\n")
            if sum(dones) > 30 and training and step >= 30:
                break
        final_weight_per_sample = []

        B = len(advantage_weights[0])

        for b in range(B):
            sample_weights = [advantage_weights[step][b] for step in range(len(advantage_weights))]
            final_weight_per_sample.append(torch.cat(sample_weights, dim=0))

        final_weight = torch.nn.utils.rnn.pad_sequence(final_weight_per_sample, batch_first=True, padding_value=0)
        print(f"final_weight shape: {final_weight.shape}")
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        print(len(active_num_list))

        return self._compose_final_output(original_left_side, original_right_side, meta_info, self.first_done_lens, final_weight)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            first_done_lens: Dict[int, int],
                            advantage_weight: torch.Tensor) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']

        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)

        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)

        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)

        if first_done_lens:
            new_responses = []
            for i, res in enumerate(final_output.batch['responses']):
                if i in first_done_lens:
                    valid_len = first_done_lens[i]
                    new_responses.append(res[:valid_len])
                else:
                    new_responses.append(res)

            max_len = max([r.size(0) for r in new_responses])
            print('max_len from final output: ', max_len)
            padded_responses = []
            for r in new_responses:
                pad_len = max_len - r.size(0)
                if pad_len > 0:
                    r = F.pad(r, (0, pad_len), value=self.tokenizer.pad_token_id)
                padded_responses.append(r)
            new_responses = torch.stack(padded_responses, dim=0)

            final_output.batch['responses'] = new_responses

            final_output.batch['input_ids'] = torch.cat([
                left_side['input_ids'],
                new_responses
            ], dim=1)
            final_output.batch['attention_mask'] = torch.cat([
                self.tensor_fn.create_attention_mask(left_side['input_ids']),
                self.tensor_fn.create_attention_mask(new_responses)
            ], dim=1)

            final_output.batch['position_ids'] = self.tensor_fn.create_position_ids(
                final_output.batch['attention_mask']
            )

        final_output.batch['advantage_weight'] = advantage_weight
        print("shape of final output")
        print(final_output.batch['input_ids'].shape)
        print(final_output.batch['attention_mask'].shape)
        print(final_output.batch['position_ids'].shape)
        print(final_output.batch['advantage_weight'].shape)
        advantage_weight = final_output.batch['advantage_weight']
        target_len = final_output.batch['position_ids'].shape[1]

        if advantage_weight.shape[1] < target_len - 2500:
            print("advantage_weight padding")
            pad_len = target_len - advantage_weight.shape[1] - 2500
            advantage_weight = F.pad(advantage_weight, (0, pad_len), value=0)
            final_output.batch['advantage_weight'] = advantage_weight
        return final_output

    def execute_predictions(self, task_ids, predictions: List[str], pad_token: str, active_mask=None, execute=True, user_task=None) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones = [], []
        max_output_len = self.config.max_obs_length

        queries = [(action, content) for action, content in zip(cur_actions, contents)]
        if execute:
            execute_results = self.batch_execute(queries)
        else:
            execute_results = [''] * sum([1 for action in cur_actions])
        suboptimals = []
        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                suboptimals.append(0)
            else:
                if action == 'execute':
                    try:
                        result = execute_results.pop(0)
                    except IndexError:
                        result = {"output": "Execution failed. no results available", 
                        "is_completed": False, "is_positive": False}
                    if isinstance(result, dict):
                        output = result.get("output", "")
                        is_completed = result.get("is_completed", False)
                        is_positive = result.get("is_positive", False)
                    else:  # Compatible with plain string results.
                        output = result
                        is_completed = False
                        is_positive = False
                    output_ids = self.tokenizer(output, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                    if output_ids.shape[0] > max_output_len:
                        output_ids = output_ids[:max_output_len]
                        output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                        output += "\n[Environment feedback too long, truncated.]"
                    if "Execution failed" in output:
                        suboptimals.append(-0.2)

                    elif is_positive:
                        suboptimals.append(0.2)
                    else:
                        suboptimals.append(0)
                    next_obs.append(f"<environment>{output} {user_task[i]}</environment>")
                    dones.append(1 if is_completed else 0)
                else:
                    next_obs.append(f'<environment> Your previous action is invalid. \
If code execution is needed, put your code for this step in a single ```python``` block. </environment>')
                    dones.append(0)
                    suboptimals.append(-0.2)
        if execute:
            assert len(execute_results) == 0

        return next_obs, dones, suboptimals

    def task_init(self, ports, task_ids):
        self.ports = ports[:len(task_ids)]
        init_message = {port: {"task_id": task_id, "state": "init"} for port, task_id in zip(self.ports, task_ids)}
        for port in self.ports:
            self.r.rpush(f"task_queue_{port}", json.dumps(init_message[port]))
        self.task_ready = True

    def score_check(self, ports, task_ids):
        check_message = {port:{"task_id": task_id, "state": "check"} for port, task_id in zip(self.ports, task_ids)}
        execute_ports = set()
        for port in self.ports[:len(check_message)]:
            self.r.rpush(f"task_queue_{port}", json.dumps(check_message[port]))
            execute_ports.add(port)
        received_ports = set()

        self.task_ready = False
        received = 0
        check_result = []
        while received < len(self.ports):
            res = self.r.blpop("result_queue", timeout=15)
            if res is None:
                received += 1
                continue
            try:
                _, result = res
                mid_result = json.loads(result)
                received_ports.add(int(mid_result["port"]))
                check_result.append(mid_result)
            except Exception as e:
                print(f"Failed to parse result: {res}, error: {e}")
            finally:
                received += 1
        missing_ports = execute_ports - received_ports
        if missing_ports:
            print(f"[Warning] Missing result from ports: {missing_ports}")
            print("execute_ports: ", execute_ports)
            print("received_ports: ", received_ports)
        for port in missing_ports:
            check_result.append({"port": port, "all_count": 1, "pass_count": 0})
        check_result = sorted(check_result, key=lambda x: int(x["port"]))
        assert len(check_result) == len(self.ports)
        check_array = np.array([[item["all_count"], item["pass_count"]] for item in check_result], dtype=object)
        return check_array

    def batch_execute(self, queires):
        execute_results = []
        execute_count = 0
        execute_ports = set()
        received_ports = set()
        for idx, (action, content) in enumerate(queires):
            if action == 'execute':
                execute_count += 1
                info = {"state": "working", 'code': content}
                self.r.rpush(f"task_queue_{self.ports[idx]}", json.dumps(info))
                execute_ports.add(self.ports[idx])
        received = 0
        while received < execute_count:
            res = self.r.blpop("result_queue", timeout=15)
            if res is None:
                received += 1
                continue
            try:
                _, result = res
                mid_result = json.loads(result)
                received_ports.add(int(mid_result["port"]))
                execute_results.append(mid_result)
            except Exception as e:
                print(f"Failed to parse result: {res}, error: {e}")
            finally:
                received += 1
        missing_ports = execute_ports - received_ports
        if missing_ports:
            print(f"[Warning] Missing result from ports: {missing_ports}")
            print("execute_ports: ", execute_ports)
            print("received_ports: ", received_ports)
        for port in missing_ports:
            print("port {} missed!".format(port))
            result = {"port": port, "output": "code execute timeout, please retry", "is_completed": False, "is_positive": False}
            execute_results.append(result)

        execute_results = sorted(execute_results, key=lambda x: int(x["port"]))
        if len(execute_results) != execute_count:
            print('execute_results', execute_results)
            print('execute_count', execute_count)
        assert len(execute_results) == execute_count
        outputs = [{"output": item["output"], "is_completed": item["is_completed"], "is_positive": item["is_positive"]}
                   for item in execute_results]
        return outputs

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []

        for prediction in predictions:
            if isinstance(prediction, str):
                pattern = r"(?<=```python\n)([\s\S]*?)(?=\n```)"
                matches = re.findall(pattern, prediction, re.DOTALL)
                if len(matches) == 1:
                    content = matches[0].strip()
                    action = "execute"
                else:
                    content = ''
                    action = "error"
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")

            actions.append(action)
            contents.append(content)

        return actions, contents
