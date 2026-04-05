import redis
from appworld import AppWorld
import re
import json
import sys
import os
import time

def worker(port):
    r = redis.StrictRedis(host='localhost', port=6390, db=0, decode_responses=True)
    is_init = False
    while True:
        _, task = r.blpop(f"task_queue_{port}")
        task = json.loads(task)
        if task['state']=='init':
            AppWorld.init_defaults.timeout_seconds = 10
            is_init = True
            with AppWorld(task_id=task['task_id'], 
                          remote_environment_url="http://127.0.0.1:"+port, experiment_name=str(port)) as world:
                print("port {} task {} inited".format(port, task['task_id']))

                print(world.task.instruction)  # Keep the worker alive after initialization.
                is_completed = False
                while True:
                    _, task = r.blpop(f"task_queue_{port}")
                    task = json.loads(task)

                    if task['state'] == 'working':
                        positive = False
                        start_time = time.time()
                        print("CODE:\n", task['code'])
                        
                        try:
                            output = world.execute(task['code'])
                        except:
                            output = "Execution failed. Code illegal, please check whether there are infinite loops in the code."
                        print("ENV output:\n", output)
                        print("%"*30)
                        if world.task_completed():
                            is_completed = True

                        result = {"port": port, "output": output, "is_completed": is_completed, "is_positive": positive}
                        r.rpush("result_queue", json.dumps(result))
                        print("time cost: ", time.time()-start_time)
        
                    if task['state'] == 'check':
                        output = world.evaluate()
                        result = {"port": port,"all_count": output.fail_count+output.pass_count,"pass_count":output.pass_count}
                        r.rpush("result_queue",json.dumps(result))
                        break
        print("#"*30)         
        

if __name__=='__main__':
    if len(sys.argv) > 1:
        port = sys.argv[1]  # Read the command-line argument.
    else:
        port = '8002'  # Default port.
    print("Current working directory:", os.getcwd())
    print("port: ", port)
    worker(port)
