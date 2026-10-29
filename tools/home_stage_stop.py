import json,os,signal,subprocess,sys,time
from pathlib import Path
run=Path(sys.argv[1]); state=json.load((run/'status.json').open());pid=state['server_pid']
p=Path(f'/proc/{pid}/cmdline')
if p.exists() and p.read_bytes():
    args=[x.decode() for x in p.read_bytes().split(b'\0') if x]
    if str(run/'config.json') not in args or not any(x.endswith('/serve/server.py') for x in args):
        raise RuntimeError('server PID/config changed')
    group=os.getpgid(pid)
    if group==os.getpgrp():raise RuntimeError('refusing own group')
    os.killpg(group,signal.SIGTERM)
    for _ in range(30):
        ps=subprocess.check_output(['ps','-eo','pgid=,stat='],text=True)
        if not any(int(a[0])==group and not a[1].startswith('Z') for l in ps.splitlines() if len(a:=l.split())==2):break
        time.sleep(1)
    else:raise RuntimeError('server did not exit')
time.sleep(3)
state['phase']='complete-development-stopped-for-next-test'
(run/'status.json').write_text(json.dumps(state,indent=2)+'\n')
print('Stopped only the recorded development server; worker receives EOF.')
