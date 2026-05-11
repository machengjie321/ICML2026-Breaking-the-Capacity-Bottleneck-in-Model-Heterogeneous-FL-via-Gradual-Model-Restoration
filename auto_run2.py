import os
import subprocess
import time
import shlex
import logging
from collections import OrderedDict
import threading
import psutil
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetCount
)

# ========== Logger Setup ==========
log_file_path = os.path.join(os.path.dirname(__file__), 'task_manager1.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),  # Overwrite log on each run
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== Resource Checks ==========
def get_system_free_memory():
    memory = psutil.virtual_memory()
    return memory.available / 1024**2, memory.total / 1024**2  # MB

def get_max_free_memory():
    try:
        nvmlInit()
        gpu_count = nvmlDeviceGetCount()
        max_free, total = 0, 0
        for i in [0,1,2]:
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            max_free = max(max_free, info.free / 1024**2)
            total += info.total / 1024**2
        nvmlShutdown()
        return max_free, total
    except Exception as e:
        logger.warning(f"Failed to check GPU memory: {e}")
        return 0, 0

def is_process_running(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True

# ========== Task Manager ==========
class TaskManager:
    def __init__(self, tasks, memory_threshold=12*1024, usage_threshold=0.98):
        self.tasks = tasks
        self.running_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.memory_threshold = memory_threshold
        self.usage_threshold = usage_threshold
    
    def _schedule_log_deletion(self, log_file, delay=60):
        """在指定秒数后删除日志文件"""
        def delayed_delete():
            try:
                # 给系统一点时间关闭文件句柄
                time.sleep(1) 
                if os.path.exists(log_file):
                    os.remove(log_file)
                    logger.info(f"Deleted log file after task failure/kill: {log_file}")
            except Exception as e:
                logger.error(f"Error deleting log file {log_file}: {e}")

        # 使用线程避免阻塞主监控循环
        timer = threading.Timer(delay, delayed_delete)
        timer.start()
        
    def check_and_run_task(self):
        free_gpu_mem, total_gpu_mem = get_max_free_memory()
        free_sys_mem, total_sys_mem = get_system_free_memory()

        if (
            free_gpu_mem > self.memory_threshold and
            free_sys_mem > 2000 and
            len(self.running_tasks) < 6 and
            self.tasks
        ):
            task = self.tasks.pop(0)
            log_file = f"log_{int(time.time())}.log"
            
            try:
                # 注意：这里使用 'w' 模式打开文件，Popen 启动后文件句柄会传给子进程
                with open(log_file, 'w') as f:
                    process = subprocess.Popen(
                        shlex.split(task),
                        stdout=f,
                        stderr=subprocess.STDOUT
                    )
                    pid = process.pid
                    self.running_tasks.append((process, task, pid, log_file))
                    logger.info(f"Started task {pid}: {task}, logging to {log_file}, running: {len(self.running_tasks)}")
                    
                    # 启动成功后的等待（原逻辑）
                    time.sleep(60)

            except Exception as e:
                logger.error(f"Failed to start task: {task}. Error: {e}")
                self.failed_tasks.append(task)
                
                # --- 新增逻辑：1分钟后删除日志文件 ---
                def delayed_delete(file_path):
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logger.info(f"Deleted failure log file: {file_path}")
                    except Exception as del_e:
                        logger.error(f"Error deleting log file {file_path}: {del_e}")

                # 创建并启动定时器，60秒后执行
                timer = threading.Timer(60, delayed_delete, args=[log_file])
                timer.start()
                
    def monitor_tasks(self):
        free_gpu_mem, total_gpu_mem = get_max_free_memory()
        free_sys_mem, total_sys_mem = get_system_free_memory()

        gpu_usage = 1 - free_gpu_mem / total_gpu_mem if total_gpu_mem > 0 else 1
        sys_usage = 1 - free_sys_mem / total_sys_mem if total_sys_mem > 0 else 1

        if gpu_usage > self.usage_threshold or sys_usage > 0.95:
            if self.running_tasks:
                process, task, pid, log_file = self.running_tasks.pop(-1)
                try:
                    os.kill(pid, 9)
                    self.tasks.insert(0, task)
                    logger.warning(f"Killed task {pid}: {task} due to high resource usage, running: {len(self.running_tasks)}")
                    self._schedule_log_deletion(log_file, delay=60)
                    
                except OSError:
                    logger.warning(f"Failed to kill task {pid}: {task}")

    def clean_completed_tasks(self):
        for process, task, pid, log_file in list(self.running_tasks):
            retcode = process.poll()
            if retcode is not None:
                if retcode == 3:
                    self.completed_tasks.append(task)
                    logger.info(f"[✔] Task completed successfully {pid}: {task}")
                else:
                    self.failed_tasks.append(task)
                    self.tasks.insert(0, task)
                    logger.warning(f"[✘] Task failed {pid}: {task} with code {retcode}")
                self.running_tasks.remove((process, task, pid, log_file))

    def run(self):
        logger.info("[🔁] Starting TaskManager loop...")
        last_status_time = time.time()

        while self.tasks or self.running_tasks:
            self.check_and_run_task()
            self.monitor_tasks()
            self.clean_completed_tasks()

            now = time.time()
            if now - last_status_time >= 600:
                logger.info(f"\n[📊] Status:")
                logger.info(f"Completed ({len(self.completed_tasks)}):")
                
                for t in self.completed_tasks[-5:]:
                    logger.info(f" ✔ {t}")
                    
                logger.info(f"Running ({len(self.running_tasks)}):") 
                for t in self.running_tasks:
                    logger.info(f"   {t}")
                
                
                logger.info(f"Pending ({len(self.tasks)}):")
                for t in self.tasks[:5]:
                    logger.info(f" ⏳ {t}")
                last_status_time = now

            time.sleep(5)

        if self.failed_tasks:
            logger.warning("\n[⚠] The following tasks failed and were retried:")
            for t in self.failed_tasks:
                logger.warning(f"- {t}")

# ========== Task Parsing ==========
def convert_to_task_list(file_path):
    task_list = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if line.startswith("nohup"):
                    line = line.split(">", 1)[0].replace("nohup ", "").rstrip("&&")
                task_list.append(line)
    return task_list

# ========== Main Entry ==========
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)

    file_path = os.path.join("autorun", "femnist.sh")
    
    # file_path = os.path.join("autorun", "femnist.sh")
    # file_path = os.path.join("autorun", "femnist_cs.sh")
    task_list = convert_to_task_list(file_path)

    manager = TaskManager(task_list)
    manager.run()
