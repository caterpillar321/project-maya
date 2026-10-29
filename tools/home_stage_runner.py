"""Start the paired home stage engines; optionally run the unchanged Claude qshort harness."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

HOME=Path.home();ROOT=HOME/'src/project-maya';RUN=Path(sys.argv[1]);FIXED=int(sys.argv[2])
RUN.mkdir(parents=True,exist_ok=False)
HELPER='user@192.168.0.76'
GATE=Path(os.environ['MAYA_STAGE_TEST_GATE'])
gate=json.load((GATE/'status.json').open())
CUT=gate['cut']; FRONT_CPU=gate['front_cpu']; MAIN_RAM=gate['main_ram']
main_env=json.load((GATE/'environment.json').open())
helper_env=json.load((GATE/'front-environment.json').open())
if gate['phase']!='passed': raise RuntimeError('GPU correctness gate not passed')
HARNESS=HOME/'fn/qshort.py'
if os.environ.get('MAYA_STAGE_SKIP_HARNESS') != '1' and hashlib.sha256(HARNESS.read_bytes()).hexdigest()!='278b008e7e6f889407f257b57725057fb61839a8dba025fbf09a38d1dc653a64':
    raise RuntimeError('original harness changed')
state={'phase':'starting','fixed':FIXED,'production_restored':False,'cut':CUT,'front_cpu':FRONT_CPU,'gate':str(GATE)}
server=None;front=None

def save(): (RUN/'status.json').write_text(json.dumps(state,indent=2)+'\n')
def remote(code,data=None):
    p=subprocess.run(['ssh','-o','BatchMode=yes',HELPER,'python3 -c '+shlex.quote(code)],input=json.dumps(data),text=True,capture_output=True,timeout=45)
    if p.returncode: raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout)
def note(s): print(time.strftime('%H:%M:%S'),s,flush=True)

try:
    if subprocess.check_output(['ss','-ltnpH','sport = :18100'],text=True).strip():raise RuntimeError('API already occupied')
    helper_env.update(MAYA_GLM_MTP_FIXED=str(FIXED), MAYA_GLM_SLOT_RAM_GB='0')
    for e in (main_env,helper_env):
        for k in list(e):
            if k.endswith(('_DIAG','_REFERENCE','_HELPER_CHECK')):e.pop(k)
    main_env.update(json.loads(os.environ.get('MAYA_STAGE_BENCH_EXTRA','{}')))
    helper_env.update(json.loads(os.environ.get('MAYA_STAGE_BENCH_FRONT_EXTRA','{}')))
    (RUN/'helper-environment.json').write_text(json.dumps(helper_env,indent=2)+'\n')
    args=[gate['front']['args'][0],'--pack','/mnt/d/GLM-5.3-Flash-BF16-IQ4_XS/pack',
          '--dial','192.168.0.4:18802','--cut',str(CUT),'--context','262144','--profile','/home/user/fn-bin/profile-mix4agent.bin']
    front=remote(r'''
import json,os,pathlib,subprocess,sys
d=json.load(sys.stdin);r=pathlib.Path('/home/user/fn-bin')/d['name'];r.mkdir(exist_ok=False)
e={k:v for k,v in os.environ.items() if not k.startswith(('MAYA_','STRATA_'))};e.update(d['env'])
log=(r/'worker.log').open('w');p=subprocess.Popen(d['args'],env=e,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
s={'pid':p.pid,'args':d['args'],'log':str(r/'worker.log')};(r/'state.json').write_text(json.dumps(s));print(json.dumps(s))
''',{'name':RUN.name,'args':args,'env':helper_env})
    cfg=json.load(Path(os.environ.get('MAYA_STAGE_CONFIG_TEMPLATE',str(HOME/'fn/codex-split-net-copy-20261029/resident-pipe.json'))).open())
    cfg['exe']=os.environ.get('MAYA_STAGE_BENCH_BINARY',str(HOME/'fn/bin-test/maya-remote-stage-ranked-20261029'))
    cfg['log']=str(RUN/'engine.log')
    cfg['env']=dict(main_env, MAYA_GLM_MTP_FIXED=str(FIXED), MAYA_GLM_SLOT_RAM_GB=os.environ.get('MAYA_STAGE_SLOT_RAM_GB','0'))
    if os.environ.get('MAYA_STAGE_ROUTE_CAPTURE') == '1':
        cfg['env']['MAYA_GLM_ROUTE_LOG']=str(RUN/'routes')
    if os.environ.get('MAYA_STAGE_TRACE') == '1':
        cfg['env'].update(MAYA_GLM_WIN_TRACE='1', MAYA_GLM_WIN_TRACE_EVERY='16', MAYA_GLM_WIN_TRACE_LIMIT='6', MAYA_GLM_TIER_AUDIT='1')
    for i,a in enumerate(cfg['args']):
        if a=='--prompt-cache-every':cfg['args'][i+1]='0'
    (RUN/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    server_env={k:v for k,v in os.environ.items() if not k.startswith(('MAYA_','STRATA_'))}
    server_log=(RUN/'server.log').open('w')
    server=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(ROOT/'serve/server.py'),'--engine','strata',
                             '--config',str(RUN/'config.json'),'--port','18100'],env=server_env,
                            stdout=server_log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    state.update(server_pid=server.pid,front=front);save();note('development server loading')
    deadline=time.monotonic()+420
    while time.monotonic()<deadline:
        if server.poll() is not None:raise RuntimeError('server exited; inspect server.log')
        try:
            with urllib.request.urlopen('http://127.0.0.1:18100/health',timeout=2) as r:health=json.load(r)
            if health.get('status')=='ok':break
        except Exception:pass
        time.sleep(2)
    else:raise TimeoutError('model startup timed out')
    if os.environ.get('MAYA_STAGE_SKIP_HARNESS') == '1':
        state['phase']='development-ready';save();note('development engine ready; no benchmark requested')
    else:
        state['phase']='qshort-running';save();note('original qshort harness starting')
        def cpu_seconds(pid):
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            return (int(fields[11])+int(fields[12])) / os.sysconf('SC_CLK_TCK')
        main_pids=[int(a[1]) for line in subprocess.check_output(['ps','-eo','pgid=,pid='],text=True).splitlines() if len(a:=line.split())==2 and int(a[0])==server.pid]
        def front_cpu_seconds():
            return remote('import json,os,pathlib,sys; d=json.load(sys.stdin); f=pathlib.Path("/proc/"+str(d["pid"])+"/stat").read_text().rsplit(")",1)[1].split(); print(json.dumps((int(f[11])+int(f[12]))/os.sysconf("SC_CLK_TCK")))',front)
        cpu_start=sum(cpu_seconds(pid) for pid in main_pids)
        front_cpu_start=front_cpu_seconds()
        started=time.monotonic()
        out=(RUN/'qshort.txt').open('w');errors=(RUN/'qshort-stderr.txt').open('w')
        p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(HARNESS),str(RUN/'engine.log'),'chat','code'],
                           stdout=out,stderr=errors,start_new_session=True)
        offset=0
        while p.poll() is None:
            text=(RUN/'qshort.txt').read_text()
            if len(text)>offset: note(text[offset:].strip());offset=len(text)
            if time.monotonic()-started>600:
                os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);raise TimeoutError('qshort timeout')
            time.sleep(1)
        text=(RUN/'qshort.txt').read_text()
        if len(text)>offset:note(text[offset:].strip())
        elapsed=time.monotonic()-started
        state.update(returncode=p.returncode,harness_wall_seconds=elapsed,
                     main_cpu_percent=100*(sum(cpu_seconds(pid) for pid in main_pids)-cpu_start)/elapsed,
                     front_cpu_percent=100*(front_cpu_seconds()-front_cpu_start)/elapsed)
        if p.returncode:raise RuntimeError('qshort failed')
        state['phase']='complete-development-kept';save()
        subprocess.run(['scp',HELPER+':'+front['log'],str(RUN/'worker.log')],check=True)
except BaseException as e:
    state.update(phase='failed',error=str(e));save();note(str(e))
    if server is not None and server.poll() is None:
        os.killpg(server.pid,signal.SIGTERM)
        try:server.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
    if front:
        remote(r'''
import json,os,pathlib,signal,sys
d=json.load(sys.stdin);p=pathlib.Path('/proc')/str(d['pid'])/'cmdline'
if p.exists() and p.read_bytes():
 if [x.decode() for x in p.read_bytes().split(b'\0') if x]!=d['args']:raise RuntimeError('worker PID changed')
 os.killpg(d['pid'],signal.SIGTERM)
print('{}')
''',front)
    raise
